from itertools import combinations
from math import sqrt, pi, ceil
from typing import Union, Tuple, List

import numpy as np
from scipy.spatial.distance import cdist
from shapely import Polygon
from skimage.draw import polygon2mask
from skimage.measure import label, regionprops, find_contours
from skimage.segmentation import find_boundaries


def auto_contrast_3d(input_im: np.ndarray, low: float, high: float) -> np.ndarray:
    """
    Apply auto-contrast adjustment to a 3D image.

    Auto-contrast is a technique to enhance the contrast of an image by redistributing pixel values within a specified
    range. This function calculates the lower and upper limits for pixel values based on the input fractions `low`
    and `high`, and then scales the image to fit within this new range. The result is an image with improved contrast.

    Parameters:
        input_im (np.ndarray): The 3D input image.
        low (float): The lower fraction for determining the lower limit of pixel values. Should be in the range [0, 1].
        high (float): The upper fraction for determining the upper limit of pixel values. Should be in the range [0, 1].

    Returns:
        np.ndarray: The auto-contrast adjusted 3D image.

    Note:
        - Ensure that the input image does not have an unsigned integer data type (e.g., uint8, uint16), as this can lead
          to incorrect results. If necessary, the function will temporarily convert the image to 'float64' for
          accurate calculations.
        - The `low` and `high` fractions determine the range of pixel values to consider for the contrast adjustment.
          A lower `low` value and a higher `high` value will result in a wider range of pixel values being used for
          contrast scaling.
    """
    if input_im.any:
        # Make sure input_im.dtype is not an unsigned integer!
        # Otherwise, the autocontrast calculation would yield erroneous results.
        if any(np.issubdtype(input_im.dtype, uint) for uint in [np.uint8, np.uint16, np.uint32, np.uint64]):
            input_im = input_im.astype('float64')
        # Sort list of pixel values
        px_values = sorted(input_im.reshape(input_im.size))
        # Get lower and upper limit of pixel values (using input fractions)
        v_min = px_values[ceil(low * input_im.size)]
        v_max = px_values[ceil(high * input_im.size) - 1]  # -1 to account for python indexing starting at 0
        # Adjust image contrast (just copied formula from Jiamin Liu's original Matlab script)
        output_image = (input_im - v_min) / (v_max - v_min)
        # Set all values below v_min to 0 and all values above v_max to 1
        output_image = np.where(output_image > 0, output_image, 0)
        output_image = np.where(output_image < 1, output_image, 1)
    else:
        output_image = input_im
    return output_image


def handle_overlaps(domes: np.ndarray = None,
                    flats: np.ndarray = None,
                    spheres: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Handle overlapping features in segmentation maps.

    This function takes segmentation masks for domes, flats, and spheres and handles overlapping features
    to ensure that each feature type remains distinct. In cases of incomplete input data (e.g., if there is no
    vesicle map to potentially overwrite dome features), masks are left as they are.

    Parameters:
        domes (numpy.ndarray, optional): Segmentation mask for domes.
        flats (numpy.ndarray, optional): Segmentation mask for flats.
        spheres (numpy.ndarray, optional): Segmentation mask for spheres.

    Returns:
        Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]: A tuple containing the updated segmentation masks for
        domes, flats, and spheres, respectively.
    """
    if domes is not None and spheres is not None:
        d_tmp = domes.astype(bool)
        v_tmp = spheres.astype(bool)
        # If dome and sphere --> keep sphere
        d_tmp = d_tmp ^ (d_tmp & v_tmp)
        if domes.dtype == bool:
            domes = d_tmp
        else:
            domes = d_tmp * domes.max()
    if flats is not None and domes is not None:
        f_tmp = flats.astype(bool)
        d_tmp = domes.astype(bool)
        # If flat and dome --> keep dome
        f_tmp = f_tmp ^ (d_tmp & f_tmp)
        if flats.dtype == bool:
            flats = f_tmp
        else:
            flats = f_tmp * flats.max()
    if spheres is not None and flats is not None:
        v_tmp = spheres.astype(bool)
        f_tmp = flats.astype(bool)
        # If sphere and flat --> keep flat
        v_tmp = v_tmp ^ (v_tmp & f_tmp)
        if spheres.dtype == bool:
            spheres = v_tmp
        else:
            spheres = v_tmp * spheres.max()

    return domes, flats, spheres


def separate_mask_features(mask: np.ndarray, exclude_alpha: bool = False) -> np.ndarray:
    """
    Separate features of different classes in a multi-class mask by setting boundary pixels to zero.

    This function takes a multi-class mask, where each class is represented by a unique integer value,
    and separates the features of different classes by setting boundary pixels around each feature to zero.
    This creates a 2-pixel separation between features, ensuring that they are not connected.

    Optionally,the alpha-channel in RGB masks can be excluded from processing.

    Parameters:
        mask (numpy.ndarray): A multi-class mask where each class is represented by an RGB channel.
        exclude_alpha (bool, optional): Whether to exclude the alpha-channel if the mask is in RGB format.
            Default is False.

    Returns:
        numpy.ndarray: The modified mask with boundary pixels set to zero.

    """
    # Optionally, exclude alpha-channel in rgb masks
    if exclude_alpha and (mask.shape[2] > 3):
        mask = mask[:, :, :-1]
    # Preallocate array for labels
    y, x, _ = mask.shape
    labeled_mask = np.zeros((y, x), dtype=int)
    # Create labels for features of all classes
    for d in range(mask.shape[2]):
        # Create labels for this class
        tmp = label(mask[:, :, d])
        if not d == 0:
            # Add number of last label (to get continuous label numbers through all classes)
            bool_index = tmp != 0
            tmp[bool_index] = tmp[bool_index] + labeled_mask.max()
        # Add labels from this class to summary label array
        labeled_mask = labeled_mask + tmp
    # Find boundaries between features of different class (e.g. domes & vesicles)
    # Attention! The input array HAS TO BE uint8 to avoid a bug in skimage.segmentation.find_boundaries
    # For documentation on this bug see:
    # https://forum.image.sc/t/segmentation-find-boundaries-different-results-for-different-platforms/35932/3
    boundaries = find_boundaries(labeled_mask.astype(np.uint8), connectivity=2, mode='outer', background=0)
    # Set values at boundaries between features to zero
    if mask.dtype is bool:
        mask[boundaries[:, :, np.newaxis].repeat(3, axis=2)] = False
    else:
        mask[boundaries[:, :, np.newaxis].repeat(3, axis=2)] = 0
    return mask


def apply_cellmask(cellmask: np.ndarray, segmask: np.ndarray, remove_edge_features: bool = False) -> np.ndarray:
    """
    Apply a cell mask to a segmentation mask.

    Pixels outside the cell are set to 255 in the segmentation mask.
    Optionally, remove features connected to the cell edge.

    Parameters:
        cellmask (np.ndarray): NxMx1 binary mask. Pixels with value 255 are considered outside the cell.
        segmask (np.ndarray): NxMx3 RGB segmentation mask.
        remove_edge_features (bool): Optional. If True, remove features connected to the cell edge. Default is False.

    Returns:
        np.ndarray: NxMx3 RGB segmentation mask with pixels outside the cell set to 255.
    """
    # Remove features connected to cell edge
    if remove_edge_features:
        # Label connected components in the z-projected binary mask (contains all features + cellmask in segmask)
        labeled_mask = label(np.any(np.concatenate((segmask.astype(bool), np.expand_dims(cellmask, axis=2)),
                                                   axis=2), axis=2))
        # Check which label contains cellmask
        label_id = labeled_mask[np.unravel_index(np.argmax(cellmask.flatten()), cellmask.shape)]
        # Set all pixel in segmask that are part of this label to zero
        segmask[labeled_mask == label_id] = 0

    # Set all pixels in segmask to 255 if they are outside the cell
    if cellmask.dtype == bool:
        segmask[cellmask] = 255
    elif cellmask.dtype == np.uint8:
        segmask[np.squeeze(cellmask) == 255, :] = 255
    else:
        raise (NotImplementedError('Cellmask dtype not supported.'))

    return segmask


def border_pixel_to_zero(im: np.ndarray, pad: bool = False) -> np.ndarray:
    """
    Set the pixels at the edges of a given 2D array to zero (or False in case of a bool array).

    Parameters:
        im (np.ndarray):
            The input 2D array.
        pad (bool, optional):
            If True, pad the input array with a border of zero pixels. If False, modify the input array in-place
            (default is False).

    Returns:
        np.ndarray:
            The modified 2D array with the pixels at the edges set to zero (or False).
    """
    if pad:
        im = np.pad(im, pad_width=1, mode='constant', constant_values=0)
    else:
        im[0, :] = im[-1, :] = im[:, 0] = im[:, -1] = 0
    return im


def polygon_dilation(binary_mask: np.ndarray[bool, 2], dist: Union[float, int]) -> np.ndarray[bool, 2]:
    """
    Dilate a binary mask using the buffer method of a shapely Polygon object.

    Parameters:
        binary_mask (np.ndarray[bool]):
            The input binary mask to be dilated.
        dist (Union[float, int]):
            The distance [pixels] by which to dilate the mask. It can be a float or an integer.

    Returns:
        np.ndarray[bool]:
            The dilated binary mask.

    Notes:
        - This function utilizes the buffer method of a shapely Polygon object to perform dilation.
        - It is particularly efficient for large masks with a single feature and dilation by a large amount.
    """
    # Get contour of the binary mask
    contour = find_contours(image=binary_mask)[0]
    # Create a shapely Polygon object from the contour
    polygon = Polygon(contour)
    # Dilate the polygon using the buffer method
    dilated_polygon = polygon.buffer(distance=dist)
    # Convert the dilated polygon back to a binary mask
    dilated_mask = polygon2mask(image_shape=binary_mask.shape, polygon=dilated_polygon.exterior.coords)
    return dilated_mask


def shapes_to_mask(mask_shape: Tuple[int, int],
                   shapes_layer: 'napari.types.LayerData') -> np.ndarray:
    """
    Generate a binary mask based on the provided ShapesData and the dimensions of the original image.

    Args:
        mask_shape (Tuple): Shape of the original image used to determine the dimensions of the mask.
        shapes_layer (LayerData): A ShapesData object containing the regions of interest.

    Returns:
        np.ndarray: A binary mask that represents the regions of interest within the dimensions of the original image.

    Notes:
        - This function creates a binary mask based on the shapes within a ShapesData layer.
        - The shape and dimensions of the mask match those of the original image.
        - If no shapes are present, the mask is filled with `False` values.
        - The built-in Napari function `to_masks` is very heavy in memory usage as it converts each shape into a
          separate mask. --> Used 'to_labels' instead and then converted to binary mask.
    """
    if not shapes_layer.data:
        mask = np.full(mask_shape, False)
    else:
        # use built-in Napari function to convert shapes into labels (i.e., integer values)
        labels = shapes_layer.to_labels(mask_shape)
        # Convert to binary mask
        mask = labels != 0
    return mask


def mask_to_shapes(mask: np.ndarray, rho: float = 1/10) -> Tuple[List[np.ndarray], List[str]]:
    """
    Convert a 2D boolean mask into a list of polygon shapes.

    Parameters:
        mask (np.ndarray):
            A 2D boolean array representing the mask, where True values indicate the presence of a structure.
        rho (float, optional):
            A density parameter controlling the number of vertices for each polygon shape. Higher values result
            in more vertices. Default is 1/10.

    Returns:
        List[np.ndarray]:
            A list of polygon shapes, where each shape is represented as a NumPy array of coordinates.
            Each shape corresponds to a detected structure in the mask.
        List[str]: A list of strings indicating the shape type for each shape. All shapes in this function are polygons.

    Notes:
        - The function calculates the number of vertices for each structure based on the line density and the
          estimated circumference of the individual structures, assuming a circular shape.
          --> n = ceil(2 * rho * sqrt(area_convex * pi))
          --> This assumption is not really valid for structures containing holes, but it works well enough.
        - Structures touching the border of the mask are correctly converted to shapes by setting a row of
          pixels touching the border to zero.
    """
    # Make sure mask is boolean
    mask = mask.astype(bool)
    # Preallocate shapes
    shapes = []
    if not mask.any():
        # Return if mask doesn't contain any structures
        return shapes
    else:
        # set one line of pixels touching the border to zero
        # --> this ensures that features touching the border are correctly converted to shapes
        mask = border_pixel_to_zero(mask)
        # Get properties of existing structures
        prop = regionprops(label(mask))
        # Loop over all structures
        for ind, struct in enumerate(prop):
            # Calculate number of vertices for each structure.
            # Based on the line density line_density and on the estimated circumference of
            # the individual structures (assumes circular shape).
            n = ceil(2 * rho * sqrt(struct.area_convex * pi))
            # Calculate offset from global image for this structure
            offset = np.array((struct.bbox[0] - 1, struct.bbox[1] - 1)).reshape(1, 2)  # -1 to account padding with zeros (below)
            # Get contours for this structure
            contours = find_contours(border_pixel_to_zero(im=struct.image, pad=True))
            # Select evenly spaced vertices for each contour
            for c, contour in enumerate(contours):
                idx = np.round(np.linspace(0, len(contour) - 1, n)).astype(int)
                contours[c] = contour[idx]
            # Combine contours if necessary
            if len(contours) == 1:
                contour = contours[0]
            elif len(contours) == 2:
                contour = connect_polygons(poly1=contours[0], poly2=contours[1])
            else:
                # Get number of contours and prepare matrix for pairwise minimum distances and corresponding points
                num_contours = len(contours)
                connection_matrix = np.full((num_contours, num_contours, 3), np.nan)
                # Calculate pairwise minimum distances and corresponding points for all contours
                for i, j in combinations(range(num_contours), 2):
                    if i != j:
                        pt_ij, pt_ji, distance = find_closest_pts(poly1=contours[i], poly2=contours[j])
                        # Store in connection matrix
                        connection_matrix[i, j, 0] = distance  # minimum distance between contour i and contour j
                        connection_matrix[j, i, 0] = distance  # minimum distance between contour j and contour i
                        connection_matrix[i, j, 1:] = pt_ij  # closest point from contour i to contour j
                        connection_matrix[j, i, 1:] = pt_ji  # closest point from contour j to contour i
                # Loop trough connection_matrix to correctly connect contours
                ind1 = 0  # start with outer contour
                contour = contours[ind1]
                while np.any(~np.isnan(connection_matrix)):
                    # Find inner contour that is closest to (current) outer contour
                    # (--> defined by distance in connection_matrix)
                    valid_indices = np.where(~np.isnan(connection_matrix[ind1, :, 0]))
                    ind2 = valid_indices[0][np.argmin(connection_matrix[ind1, valid_indices, 0])]
                    # Connect to form new outer contour
                    contour = connect_polygons(poly1=contour, poly2=contours[ind2],
                                               pt1=connection_matrix[ind1, ind2, 1:],
                                               pt2=connection_matrix[ind2, ind1, 1:])
                    # Remove "used" values from connection_matrix
                    connection_matrix[ind1, :, :] = np.nan
                    connection_matrix[:, ind1, :] = np.nan
                    # Update index for outer contour
                    ind1 = ind2
            # Add contour to shapes (apply offset to get global coordinates)
            if len(contour) > 2:
                shapes.append(contour + offset)
        # All shapes are polygons when created from an RGB mask
        # --> Create a list of the same length as shapes containing the string 'polygon'
        shape_types = ['polygon' for _ in range(len(shapes))]
        # Return the shapes with reduced number of vertices
        return shapes, shape_types


def find_closest_pts(poly1: np.ndarray, poly2: np.ndarray) -> (np.ndarray, np.ndarray, float):
    """
    Find the closest points between two polygons and their respective Euclidean distance.

    Calculates the pairwise distances between all vertices of two input polygons and identifies
    the points with the smallest distance.

    Parameters:
        poly1 (np.ndarray): The first polygon as an Nx2 numpy array of (x, y) coordinates.
        poly2 (np.ndarray): The second polygon as an Mx2 numpy array of (x, y) coordinates.

    Returns:
        Tuple[np.ndarray, np.ndarray, float]: A tuple containing the closest point of poly1,
        the closest of from poly2, and the Euclidean distance between them.

    Example:
        poly1 = np.array([[1, 2], [3, 4], [5, 6]])
        poly2 = np.array([[2, 3], [4, 5], [6, 7]])
        closest_pt1, closest_pt2, distance = find_closest_pts(poly1, poly2)
    """
    # Calculate pairwise distances between all vertices of the two polygons
    distances = cdist(poly1, poly2)
    # Find the indices of the smallest distance in the distance matrix
    min_indices = np.unravel_index(np.argmin(distances), distances.shape)

    # Get the closest points and the distance between them
    closest_point_poly1 = poly1[min_indices[0]]
    closest_point_poly2 = poly2[min_indices[1]]
    distance = distances[min_indices]
    # Return the closest points and the distance between them
    return closest_point_poly1, closest_point_poly2, distance


def connect_polygons(poly1: np.ndarray, poly2: np.ndarray,
                     pt1: np.ndarray = None, pt2: np.ndarray = None) -> np.ndarray:
    """
    Connect two polygons with optional connection points to form a single closed polygon.

    This function connects two input polygons, `poly1` and `poly2`, to create a single polygon. You can also provide
    optional points `pt1` and `pt2` to ensure that the polygons are connected at specific points.

    Parameters:
    - poly1 (np.ndarray): The first polygon as an Nx2 numpy array of (x, y) coordinates.
    - poly2 (np.ndarray): The second polygon as an Nx2 numpy array of (x, y) coordinates.
    - pt1 (np.ndarray, optional): Optional point to be used as the connection point of `poly1`. Defaults to None.
    - pt2 (np.ndarray, optional): Optional point to be used as the connection point of `poly2`. Defaults to None.

    Returns:
    - np.ndarray: The connected polygon as an Nx2 numpy array of (x, y) coordinates.
    """
    # Prepare polygons for connection if necessary
    for pt, poly in zip([pt1, pt2], [poly1, poly2]):
        if pt is not None:
            # Check if pt is already at start and end of poly
            if np.all(poly[0, :] == pt) and np.all(poly[-1, :] == pt):
                continue
            else:
                # Find row-index of pt in poly
                pt_index = np.where((poly == pt).all(axis=1))[0][0]
                # Rearrange poly to have pt at start and end
                poly = np.concatenate((np.roll(poly[:-1, :], -pt_index, axis=0), [pt]), axis=0)
                if pt is pt1:
                    poly1 = poly
                elif pt is pt2:
                    poly2 = poly
    # Connect the two polygons
    connected_poly = np.concatenate((poly1, poly2, [poly1[0, :]]), axis=0)
    return connected_poly


def intersect2d(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Find row intersection between two 2D NumPy arrays, a and b.

    Parameters:
        a (np.ndarray): The first 2D NumPy array.
        b (np.ndarray): The second 2D NumPy array.

    Returns:
        np.ndarray: A new 2D NumPy array containing the rows that are present in both a and b.

    Reference:
        Found at https://gist.github.com/Robaina/b742f44f489a07cd26b49222f6063ef7
    """
    return np.array([x for x in set(tuple(x) for x in a) & set(tuple(x) for x in b)])


def divide(a: Union[int, float], b: Union[int, float]):
    """
    Divides a by b and returns NaN in case of division by zero.

    Parameters:
        a (Union[int, float]): The numerator.
        b (Union[int, float]): The denominator.

    Returns:
        Union[float, np.nan]: The result of dividing a by b. If b is zero, returns NaN.
    """
    if b == 0:
        return np.nan
    else:
        return a / b


def list_to_array(input_list: List[np.ndarray]) -> np.ndarray:
    """
    Convert a list of 2D NumPy arrays into a 3-dimensional NumPy array. Used for saving feature shapes as numpy arrays.

    Parameters:
        input_list (List[np.ndarray]): A list containing 2D NumPy arrays with shape (N, 2).

    Returns:
        np.ndarray: A 3-dimensional NumPy array of shape (max_length, 2, X) where X is the number of arrays
                   in the input_list. The arrays in the input_list are padded with NaN values to have the same
                   length (max_length) along the first axis. Any "empty" slots in the arrays are filled with NaN values.
    """
    if not input_list:
        # Return empty array if input_list is empty
        return np.empty((0, 2, 0))
    else:
        # Get the largest array among all arrays in the list
        max_length = max(arr.shape[0] for arr in input_list)
        # Pad the arrays with NaN values to make them have the same length
        padded_list = []
        for arr in input_list:
            pad_length = max_length - arr.shape[0]
            padded_arr = np.pad(arr, ((0, pad_length), (0, 0)), mode='constant', constant_values=np.nan)
            padded_list.append(padded_arr)
        # Stack the padded arrays along a new axis to create a 3-dimensional array
        stacked_arrays = np.stack(padded_list, axis=2)
        return stacked_arrays


def array_to_list(stacked_array: np.ndarray) -> List[np.ndarray]:
    """
    Convert a 3-dimensional NumPy array back to a list of 2D NumPy arrays.

    This function is used after loading numpy arrays to restore the original list
    of 2D arrays and remove any padding with NaN values, making the data suitable for
    display in Napari.

    Parameters:
        stacked_array (np.ndarray):
            A 3-dimensional NumPy array of shape (max_length, 2, X), where X is the number
            of arrays in the original input_list. The arrays should have been padded with NaN values.

    Returns:
        List[np.ndarray]:
            A list containing 2D NumPy arrays with shape (N, 2), where N varies for each array.
            The function removes the padding with NaN values and restores the original arrays.
            If the input array is empty, an empty list is returned.
    """

    restored_list = []
    if stacked_array.any():
        # Remove the padding with NaN values and restore the original list arrays
        for x in range(stacked_array.shape[2]):
            array_without_nan = stacked_array[:, :, x][~np.isnan(stacked_array[:, :, x]).any(axis=1)]
            restored_list.append(array_without_nan)
    return restored_list


if __name__ == '__main__':
    pass
