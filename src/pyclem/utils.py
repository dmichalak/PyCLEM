import configparser
import os
import re
import itertools
from math import sqrt, pi, ceil
from pathlib import Path
from typing import Union, Tuple, List

import numpy as np
import pandas as pd
import tifffile
from PIL import Image
from scipy.ndimage import binary_fill_holes
from scipy.spatial.distance import cdist
from shapely import Polygon
from skimage.draw import polygon2mask
from skimage.io import imread, imsave
from skimage.measure import label, regionprops, find_contours
from skimage.morphology import disk, binary_closing, remove_small_objects
from skimage.segmentation import find_boundaries
from skimage.transform import rescale, resize

from pyclem.file_io import get_files


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


def mask_to_shapes(mask: np.ndarray, rho: float = 1/10) -> pd.DataFrame:
    """
    Convert a 2D boolean mask into a DataFrame containing polygon shapes.

    This function takes a 2D boolean array representing the mask, where True values indicate the presence of a structure.
    It then converts these structures into polygon shapes, where each shape is represented as a NumPy array of coordinates.
    Each shape corresponds to a detected structure in the mask.

    Parameters:
        mask (np.ndarray):
            A 2D boolean array representing the mask, where True values indicate the presence of a structure.
        rho (float, optional):
            A density parameter controlling the number of vertices for each polygon shape. Higher values result
            in more vertices. Default is 1/10.

    Returns:
        pd.DataFrame:
            A DataFrame containing the polygon shapes detected in the mask. Each row represents a vertex of a shape.
            The DataFrame has columns: 'index' (shape index), 'shape-type' (always 'polygon'), 'vertex-index'
            (index of the vertex within its shape), 'axis-0' (x-coordinate), and 'axis-1' (y-coordinate).

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
    if not mask.any():
        # Return empty data frame if mask doesn't contain any structures
        shape_data = pd.DataFrame({new_list: np.asarray([])
                                   for new_list in ['index', 'shape-type', 'vertex-index', 'axis-0', 'axis-1']})
        return shape_data
    else:
        # Preallocate shapes
        shapes = []
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
            offset = (np.array((struct.bbox[0] - 1, struct.bbox[1] - 1))  # -1 to account padding with zeros (below)
                      .reshape(1, 2))
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
                for i, j in itertools.combinations(range(num_contours), 2):
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
            # Note: Up to here, contours were closed polygons (i.e., last point = first point)
            #       --> Remove last point to avoid duplicate points in the final shape
            if len(contour) > 2:
                shapes.append(contour[:-1, :] + offset)

        # Prepare arrays with information for every shape vertex (e.g., shape type, vertex-index, ...)
        shape_types = np.concatenate([['polygon' for vertice in shape] for shape in shapes], axis=0)
        vertex_indices = np.concatenate([[i for i in range(len(shape))] for shape in shapes], axis=0)
        shape_indices = np.concatenate([[shape_num for vertice in shape] for (shape_num, shape) in enumerate(shapes)],
                                       axis=0)
        shapes = np.concatenate(shapes, axis=0)
        # Store all information in Pandas DataFrame
        shape_data = pd.DataFrame({new_list: entry for new_list, entry in
                                   zip(['index', 'shape-type', 'vertex-index', 'axis-0', 'axis-1'],
                                       [shape_indices, shape_types, vertex_indices, shapes[:, 0], shapes[:, 1]])})
        return shape_data


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


# Todo: Refactor and move to file_io.py
def remove_duplicate_files_from_list(files: List[Union[str, Path]]) -> List[Path]:
    """
    Removes duplicates from a list of files. Only the base filename is considered for this meaning if there are files
    with the same name in different subfolders, all but one will be removed from the list.
    :param files: list of filenames (str or Path objects)
    :return: list of filenames (str or Path objects)
    """
    unique_files = set()
    duplicate_files = []
    for file in files:
        # Make sure file is a Path() object
        file = Path(file)
        # Get the filename without the path
        file_name = file.name
        # Check if the filename is already in the set of unique files
        if file_name in unique_files:
            duplicate_files.append(file)
        else:
            unique_files.add(file_name)
    # Remove the duplicate files from the original list
    for duplicate_file in duplicate_files:
        files.remove(duplicate_file)

    return files


def is_isolated(label_tile: np.ndarray[int, 2], central_feat_nr: int, min_dist: Union[float, int]) -> bool:
    """
    Check if a labeled mask tile contains an isolated feature at the center.

    Parameters:
        label_tile (np.ndarray):
            NxN numpy array, representing an integer labeled mask tile containing features. Here, features are not
            distinguished by their class, but only by their label number.
        central_feat_nr (int):
            The label number of the central feature to be checked for isolation.
        min_dist (Union[float, int]):
            The minimum distance threshold [pixels] for determining if a tile contains an isolated feature.
            It can be a float or an integer.

    Returns:
        bool:
            True if the tile contains an isolated feature at the center, False otherwise.

    Notes:
        - This function checks if any other features are within min_dist pixels of the edge
          of the central feature in the given labeled mask tile.
    """
    # get binary mask containing all features except the central one
    other_features = (label_tile != central_feat_nr) & (label_tile != 0)
    if not np.any(other_features):
        return True
    # get binary mask containing only the central feature
    central_feature = label_tile == central_feat_nr
    # dilate central feature by min_dist pixels
    central_feature = polygon_dilation(binary_mask=central_feature, dist=min_dist)
    # check if any other feature overlaps with the dilated central feature
    # if not, the central feature in this tile is defined as isolated
    return not np.any(central_feature & other_features)


def adjust_px_size(image: np.ndarray,
                   px_size_new: Union[int, float],
                   px_size_old: Union[int, float],
                   channel_axis: int = 2) -> np.ndarray:
    """
    Rescales an image from its original pixel size to a new pixel size.

    This function rescales the input image from its original pixel size, specified by `px_size_old`, to a new pixel
    size, specified by `px_size_new`. The rescaling is performed using anti-aliasing if the image data type is not boolean.

    Parameters:
        image (np.ndarray): The input image to be rescaled.
        px_size_new (Union[int, float]): The target pixel size in the same units as `px_size_old`. This is the
            desired pixel size for the output image.
        px_size_old (Union[int, float]): The original pixel size in the same units as `px_size_new`. This is the
            pixel size of the input image.
        channel_axis (int, optional): The axis that represents channels or color channels in the image. Default is 2.

    Returns:
        np.ndarray: The rescaled image with the new pixel size.

    Raises:
        NotImplementedError: If the input image has a dimensionality that is not supported by this function.

    Note:
        - The function performs rescaling based on the provided pixel sizes while considering the `channel_axis`.
        - If the image data type is boolean, anti-aliasing is not applied.
        - The output image is returned with `preserve_range=True`, meaning the pixel values are preserved without
          scaling to the full range of the data type.
        - The function raises a `NotImplementedError` for image dimensionalities that are not supported.
    """
    scaling_factor = px_size_old / px_size_new
    anti_alias = not image.dtype == bool
    if image.ndim == 3:
        scaled_image = rescale(image, scaling_factor, anti_aliasing=anti_alias, channel_axis=channel_axis,
                               preserve_range=True)
    elif image.ndim == 2:
        scaled_image = rescale(image, scaling_factor, anti_aliasing=anti_alias, preserve_range=True)
    else:
        raise NotImplementedError('Images of this dimensionality (np.ndim = ' + str(image.ndim) +
                                  ') cannot yet be processed with "adjust_px_size"')

    return scaled_image


def mrcnn_preprocess(files: Union[list, str, Path],
                     tile_size: int, tile_overlap: int,
                     px_size_mrcnn: float, contrast_limits: Tuple[float, float],
                     parent_folder: Union[str, Path] = Path, tile_dir: str = Path):
    """
    Prepare EM-Montage for automated segmentation with MaskRCNN
    """
    # Make sure that files is iterable
    if not isinstance(files, list):
        files = list([files])
    # Loop over all files
    for file in files:
        print('Splitting image into tiles for  ' + str(file))
        # make file a absolute Path object
        file = Path(parent_folder, file)
        # Open image as numpy array
        im = imread(file)
        # Adjust pixel spacing for further processing in Mask-RCNN
        px_size_old = mrcnn_get_2d_pixelsize(filename=file)
        im = adjust_px_size(image=im, px_size_new=px_size_mrcnn, px_size_old=px_size_old)
        # Prepare new folder for tiles of this image
        base_filename = file.stem
        im_dir = Path(parent_folder, tile_dir, base_filename)
        im_dir.mkdir(parents=True, exist_ok=True)
        # Split image into smaller tiles and save them individually in new folder
        mrcnn_split_image(input_im=im, xy_size=tile_size, overlap=tile_overlap, cont_limit=contrast_limits,
                          tile_dir=im_dir)

        # Get path of image tiles relative to original image
        rel_path = os.path.relpath(im_dir, os.path.dirname(file))
        # Save tile_size, tile_overlap and relative path of image tiles in protocol .txt file
        fn_protocol = Path(file.parent, file.stem + '_segprotocol.txt')
        config = configparser.ConfigParser()
        if fn_protocol.exists():
            config.read(fn_protocol)
        if not config.has_section('PREPROCESS'):
            config.add_section('PREPROCESS')
        config.set('PREPROCESS', 'tile_size', str(tile_size))
        config.set('PREPROCESS', 'tile_overlap', str(tile_overlap))
        config.set('PREPROCESS', 'px_size_mrcnn', str(px_size_mrcnn))
        config.set('PREPROCESS', 'px_size_original', str(px_size_old))
        config.set('PREPROCESS', 'auto_contrast', str(contrast_limits))
        config.set('PREPROCESS', 'rel_path', str(rel_path))
        with open(fn_protocol, 'w') as configfile:
            config.write(configfile)


def mrcnn_split_image(input_im: np.ndarray,
                      xy_size: int, overlap: int,
                      cont_limit: Tuple[float, float] = None,
                      tile_dir: Union[str, Path] = Path):
    """
    Splits image into overlapping tiles of shape tile_size x tile_size.
    Contrast in tiles is automatically adjusted.
    """
    # Get image size
    imsize = input_im.shape
    imsize_x = imsize[1]
    imsize_y = imsize[0]
    # If splitting a mask, preallocate alpha channel to save computation time
    if input_im.ndim == 3:
        alpha = np.ones((xy_size, xy_size, 1)) * 255
    # Prepare counter
    count_x = 1
    # for xx in range(0, imsize_x - size_xy + 1, size_xy - overlap):
    for xx in range(0, imsize_x - (xy_size - overlap) + 1, xy_size - overlap):
        count_y = 1
        # for yy in range(0, imsize_y - size_xy + 1, size_xy - overlap):
        for yy in range(0, imsize_y - (xy_size - overlap) + 1, xy_size - overlap):
            # Beginning and end of each tile
            x_start = xx
            x_end = min([imsize_x, xx + xy_size])
            y_start = yy
            y_end = min([imsize_y, yy + xy_size])
            # Crop out tile
            if input_im.ndim == 2:  # for EM-image
                tile = input_im[y_start:y_end, x_start:x_end]
                # if not tile.shape == (600, 600):
                #     print('Debug stop! tile.shape is not 600x600.')
                # if (count_x == 5) & (count_y == 5):
                #     print('Debug stop! Check contrast at this tile!')
                # Adjust contrast
                tile = auto_contrast_3d(input_im=tile, low=cont_limit[0], high=cont_limit[1])
                # Transform tile to uint8 RGB image
                tile_uint8 = np.uint8(tile * 255)
                save_tile = np.stack((tile_uint8, tile_uint8, tile_uint8), axis=2)
                # Generate save name
                save_name = Path(tile_dir, 'x' + f'{count_x:03}' + 'y' + f'{count_y:03}' + '.tif')
            elif input_im.ndim == 3:  # for mask
                tile = input_im[y_start:y_end, x_start:x_end, :]
                # Add alpha channel set to 1 (used later to make zero values transparent in Napari)
                tile = np.concatenate((tile, alpha), axis=-1)
                # Set alpha channel to 0 wherever the RGB values are (0,0,0)
                tile[tile[:, :, :-1].sum(axis=-1) == 0, -1] = 0
                save_tile = np.uint8(tile)
                # Generate save name
                save_name = Path(tile_dir, 'x' + f'{count_x:03}' + 'y' + f'{count_y:03}' + '_mask.tif')
            # Save tile
            imsave(save_name, save_tile, check_contrast=False)
            # Increase counters to correctly label tile files
            count_y += 1
        count_x += 1


def mrcnn_postprocess(files, px_size_mrcnn: float, tile_size: int, tile_overlap: int,
                      min_feature_size: int, parent_folder: Union[str, Path], result_folder: str = Path(),
                      line_density: float = 1/10):
    """
    Assembles full mask from image tiles as output by MaskRCNN-segmentation (PR_segmentation.py).
    """
    # Make sure that files is iterable
    if not isinstance(files, list):
        files = list([files])
    # Loop over all files to process
    for file in files:
        print('Reassembling mask for ' + str(file))
        # Make file an absolute Path object
        file = Path(parent_folder, file)
        # Get base-filename
        base_filename = file.stem
        # Assemble name of folder containing the segmentation results
        seg_dir = Path(parent_folder, result_folder, base_filename)
        # Get pixel size of original image
        px_size_original = mrcnn_get_2d_pixelsize(file)
        # Get dimensions of original Image
        im_size_original = mrcnn_get_xy_shape(filename=file)
        # Calculate dimensions of scaled image (skimage.transform.rescale also uses np.round to get scaled shape)
        scaling_factor = px_size_original/px_size_mrcnn  # see also adjust_px_size for comparison
        im_size_scaled = calculate_output_shape_rescale(orig_shape=im_size_original, scale=scaling_factor)
        # Assemble full mask from tiles
        mask = mrcnn_assemble_mask(im_size=im_size_scaled, xy_size=tile_size, overlap=tile_overlap,
                                   min_feature=min_feature_size, seg_dir=seg_dir, tile_ext='_mrcnn_seg.jpg')
        # Resize assembled mask back to original size
        mask = resize(image=mask, output_shape=im_size_original+(3,))
        # Save mask as RGB image with raw EM-image
        mask = np.uint8(255 * mask)
        save_name = file.with_name(file.stem + '_segmask.tif')
        imsave(save_name, mask, check_contrast=False)

        # Get path of image tiles relative to original image
        rel_path = os.path.relpath(seg_dir, os.path.dirname(file))
        # Save tile_size, tile_overlap and relative path of image tiles in protocol .txt file
        # for later use in CheckSegmentationGui.py
        fn_protocol = Path(file.parent, file.stem + '_segprotocol.txt')
        config = configparser.ConfigParser()
        if fn_protocol.exists():
            config.read(fn_protocol)
        if not config.has_section('POSTPROCESS'):
            config.add_section('POSTPROCESS')
        config.set('POSTPROCESS', 'tile_size', str(tile_size))
        config.set('POSTPROCESS', 'tile_overlap', str(tile_overlap))
        config.set('POSTPROCESS', 'rel_path', str(rel_path))
        with open(fn_protocol, 'w') as configfile:
            config.write(configfile)


def mrcnn_assemble_mask(im_size: Tuple[int, int],
                        xy_size: int, overlap: int,
                        min_feature: Union[int, float],
                        seg_dir: Union[str, Path],
                        tile_ext: str) -> np.ndarray:
    """
    Assembling full masks from mask tiles.
    """
    # Make sure seg_dir is a Path object
    seg_dir = Path(seg_dir)
    # Get image size to calculate range of for-loops
    imsize_y, imsize_x = im_size
    # Pre-allocate arrays for assembled segmentation data
    dome = np.zeros(im_size).astype(bool)
    flat = np.zeros(im_size).astype(bool)
    sphere = np.zeros(im_size).astype(bool)

    # Prepare counter and structuring element
    count_x = 1
    struct_el = disk(2)
    # Reassemble three logical masks for flats, domes and spheres from individual tiles
    # for xx in range(0, imsize_x - size_xy + 1, size_xy - overlap):
    for xx in range(0, imsize_x - (xy_size - overlap) + 1, xy_size - overlap):
        count_y = 1
        # for yy in range(0, imsize_y - size_xy + 1, size_xy - overlap):
        for yy in range(0, imsize_y - (xy_size - overlap) + 1, xy_size - overlap):
            # Beginning and end of each tile
            x_start = xx
            x_end = min([imsize_x, xx + xy_size])
            y_start = yy
            y_end = min([imsize_y, yy + xy_size])
            # Load segmentation mask for current tile
            fn = 'x' + f'{count_x:03}' + 'y' + f'{count_y:03}' + tile_ext
            # Load mask and split into logical masks for the different feature types.
            tmp = imread(fname=seg_dir.joinpath(fn))
            if tmp.ndim > 3:
                tmp = tmp[:, :, :2]
            dome_local, flat_local, sphere_local = mrcnn_split_mask_to_rgb(tmp)
            if dome_local.any():
                # smooth features and close small holes between features or border
                dome_local = binary_closing(image=dome_local, footprint=struct_el)
                # Remove features that are touching the border
                dome_local = clear_border(dome_local)
            if flat_local.any():
                # smooth features and close small holes between features or border
                flat_local = binary_closing(image=flat_local, footprint=struct_el)
                # Remove features that are touching the border
                flat_local = clear_border(flat_local)
            if sphere_local.any():
                # smooth features and close small holes between features or border
                sphere_local = binary_closing(image=sphere_local, footprint=struct_el)
                # Remove features that are touching the border
                sphere_local = clear_border(sphere_local)
            # Add features from current tile to full masks
            dome[y_start:y_end, x_start:x_end] = dome[y_start:y_end, x_start:x_end] | dome_local
            flat[y_start:y_end, x_start:x_end] = flat[y_start:y_end, x_start:x_end] | flat_local
            sphere[y_start:y_end, x_start:x_end] = sphere[y_start:y_end, x_start:x_end] | sphere_local
            # Increase counters
            count_y += 1
        count_x += 1

    # Handle overlapping areas that contain more than one feature of different classes
    mask_dome, mask_flat, mask_sphere = handle_overlaps(domes=dome, flats=flat, spheres=sphere)
    # Stack to build (still boolean) RGB mask
    mask = np.stack((mask_dome, mask_flat, mask_sphere), axis=2)
    # Make sure that features of different class (e.g. domes & vesicles) do not touch
    # I intentionally did not remove features that form a closed loop with another feature of a different class.
    # There are (rare) cases were this is actually correct (e.g. dome inside a flat).
    mask = separate_mask_features(mask)
    # Remove features below the size threshold
    for d in range(mask.shape[2]):
        mask[:, :, d] = remove_small_objects(mask[:, :, d], min_size=min_feature, connectivity=1)
    # Do NOT fill holes in features, as this will lead to overlapping features!
    return mask


def mrcnn_prep_trainingdata(files, parent_dir: Union[str, Path], train_dir: str,
                            contrast_limits: Tuple[float], train_ext: tuple,
                            px_size_mrcnn: float, offset: int, offset_step: int, max_tile_size: int,
                            max_size_border_feature: int):
    # Loop over all files
    for file in files:
        print('Extracting training features for ' + str(file))
        file = Path(parent_dir, file)
        # Open EM-image
        im = imread(file)
        im_size_original = im.shape
        px_size_original = mrcnn_get_2d_pixelsize(file)
        # Open mask
        mask_fn = file.replace(train_ext[0], train_ext[1])
        mask = imread(mask_fn)
        # Rescale image and mask (to enable better feature recognition with Mask-RCNN)
        im = adjust_px_size(image=im, px_size_new=px_size_mrcnn, px_size_old=px_size_original)
        mask = adjust_px_size(image=mask, px_size_new=px_size_mrcnn, px_size_old=px_size_original)
        # Transform mask from RGB image to a single layer uint8 array
        # 1 = red = domes; 2 = green = flat; 3 = blue = vesicle
        mask, _ = mrcnn_transform_rgb_to_class(mask)

        # Prepare directory for saving features found in this file
        parent_dir, filename = os.path.split(file)
        base_filename, ext = os.path.splitext(filename)

        tile_dir = Path(parent_dir, train_dir, base_filename)
        tile_dir.mkdir(parents=True, exist_ok=True)

        # Extract features from mask in form of cropped tiles (they vary in size).
        # They same tiles are also extracted from the corresponding EM-image.
        # Tiles are saved in subdirectory train_dir
        mrcnn_extract_features(feature_dir=tile_dir,
                               input_mask=mask, input_im=im,
                               offset=offset, offset_step=offset_step, max_tile_size=max_tile_size,
                               max_size_border_feature=max_size_border_feature,
                               low_limit=contrast_limits[0], upp_limit=contrast_limits[1])
    # Get parent directory
    parent = Path(os.path.split(files[0])[0])
    # Get file names of saved feature tiles
    im_tiles, _ = get_files(r'feature.*\.tif', parent.joinpath(train_dir))
    # Store tile pairs in training list
    train_list = open(parent.joinpath(train_dir, 'training_list.txt'), 'a')
    for entry in im_tiles:
        im_tile = str(parent.joinpath(train_dir, entry))
        mask_tile = str(im_tile).replace('.tif', '.png')
        train_list.write(im_tile + ',' + mask_tile + '\n')
    train_list.close()


def mrcnn_transform_rgb_to_class(input_mask: np.ndarray) -> (np.ndarray, np.ndarray):
    # Extract color channels from RGB mask and generate binary feature masks
    red, green, blue = mrcnn_split_mask_to_rgb(input_mask)
    # Generate binary mask (without out-of-cell areas)
    bin_mask = red | green | blue
    # Generate integer mask.
    # 1=red, 2=green, 3=blue, 0=black
    int_mask = red * 1 + green * 2 + blue * 3
    # Transform to uint8 array
    output_mask = np.uint8(int_mask)
    return output_mask, bin_mask


def mrcnn_split_mask_to_rgb(mask: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
    """
    Extract individual color layers from various mask formats used in the mrcnn segmentation pipeline.
    The color layers are exported as boolean masks.
    Additionally, white (out of cell) areas are excluded. At this point this is of use only when preparing training
    data from manually created segmentation masks.

    Necessary because the MaskRCNN algorithm saves the resulting masks as a bit of a weird *.jpg,
    where red, green and blue are encoded as values ~50, ~100 and ~150, respectively.
    In contrast, CheckSegmentationGui saves masks as "normal" *.png files the third dimension as channel axis.
    """
    # RGB mask with third dimension as channel-axis (e.g. *_mask.tif ... masks for checking segmentation manually)
    if mask.ndim == 3 and mask.shape[2] in [3, 4]:
        red = (mask[:, :, 0] != 0) & (mask[:, :, 1] == 0) & (mask[:, :, 2] == 0)
        green = (mask[:, :, 1] != 0) & (mask[:, :, 0] == 0) & (mask[:, :, 2] == 0)
        blue = (mask[:, :, 2] != 0) & (mask[:, :, 0] == 0) & (mask[:, :, 1] == 0)
    # RGB mask with first dimension as channel-axis
    elif mask.ndim == 3 and mask.shape[0] in [3, 4]:
        red = (mask[0, :, :] != 0) & (mask[1, :, :] == 0) & (mask[2, :, :] == 0)
        green = (mask[1, :, :] != 0) & (mask[0, :, :] == 0) & (mask[2, :, :] == 0)
        blue = (mask[2, :, :] != 0) & (mask[0, :, :] == 0) & (mask[1, :, :] == 0)
    # RGB mask with third dimension as channel-axis (e.g. *_mrcnn_seg.jpg ... output of MaskRCNN segmentation)
    elif mask.ndim == 2:
        red = (mask >= 25) & (mask < 75)
        green = (mask >= 75) & (mask < 125)
        blue = mask >= 125
    else:
        raise TypeError('The used mask-type is not implemented in mrcnn_split_to_rgb!')
    return red, green, blue


def mrcnn_extract_features(feature_dir: Union[Path, str],
                           input_mask: np.ndarray, input_im: np.ndarray,
                           offset: int, offset_step: int, max_tile_size: int,
                           max_size_border_feature: int, low_limit: float,
                           upp_limit: float):
    # Todo: Completely revisit this function!
    # Give each feature an ID number
    label_mask = label(input_mask, connectivity=1)
    # Get statistics about all features, importantly including centroids and bounding boxes
    stats = regionprops(label_mask)
    # Get size of whole mask
    imsize_x, imsize_y = input_mask.shape
    # Loop over all features
    for feature_nr, feature in enumerate(stats):
        for current_offset in np.arange(0, max_tile_size / 2 - offset, offset_step).astype(int) + offset:
            # Get bounding box for this feature from stats and add offset
            (x_min, y_min, x_max, y_max) = feature.bbox
            x_start = max(1, x_min - current_offset)
            x_end = min(x_max + current_offset, imsize_x)
            y_start = max(1, y_min - current_offset)
            y_end = min(y_max + current_offset, imsize_y)
            # Crop mask
            mask_tmp = input_mask[x_start:x_end, y_start:y_end]
            # Remove features that are touching the border from mask
            mask_feature = clear_border(mask_tmp)
            mask_border = mask_tmp - mask_feature
            # Exit loop if the features removed at border within the accepted size limit.
            # Also exit loop if the next iteration would lead to a tile size bigger than the maximum tile size.
            if x_end - x_start > max_tile_size - offset_step \
                    or y_end - y_start > max_tile_size - offset_step \
                    or sum(sum(mask_border)) < max_size_border_feature:
                # Save mask tile
                feature_name = 'feature_' + f'{feature_nr + 1:03}' + '.png'
                imsave(Path(feature_dir, feature_name), mask_feature, check_contrast=False)
                break
        # Crop corresponding image
        # noinspection PyUnboundLocalVariable
        im_crop = input_im[x_start:x_end, y_start:y_end]
        # Adjust contrast
        im_crop = auto_contrast_3d(im_crop, low_limit, upp_limit)
        # Transform cropped image tile to uint8 RGB image
        im_crop_uint8 = np.uint8(im_crop * 255)
        im_feature = np.stack((im_crop_uint8, im_crop_uint8, im_crop_uint8), axis=2)
        # Save image tile
        feature_name = 'feature_' + f'{feature_nr + 1:03}' + '.tif'
        imsave(Path(feature_dir, feature_name), im_feature)


def mrcnn_get_2d_pixelsize(filename: Union[str, Path]):
    """
    Get the 2D pixel size in standard SI units (meters) from a TIFF file, as saved by ImageJ or OME-TIFF.
    This function is compatible with Python versions < 3.8, as used in the Mask R-CNN environment.

    This function reads the xy pixel size (physical pixel size) from the TIFF file metadata, considering the X and Y
    resolutions. It checks if the x and y pixel sizes match and make sense. If not, a ValueError is raised.

    Implementation based on information found in the tifffile library (https://pypi.org/project/tifffile) and in the
    plant-seg repository on GitHub (https://github.com/hci-unihd/plant-seg#citation).

    Parameters:
        filename (Union[str, Path]): Path to the TIFF file.

    Returns:
        float: The 2D pixel size in standard SI units (meters). If pixel size information is missing or
               unrealistic, a ValueError is raised.
    """
    # Sub-function to parse X and Y resolution from TIFF metadata
    def _xy_voxel_size(tags, key):
        assert key in ['XResolution', 'YResolution']
        if key in tags:
            num_pixels, units = tags[key].value
            return units / num_pixels
        # return default
        return 1.

    # Open TIFF file
    with tifffile.TiffFile(filename) as tiff:
        # Get pixel size for OME-TIFF files
        if tiff.ome_metadata is not None:
            # Use regular expressions to extract PhysicalSizeX and PhysicalSizeY
            physical_size_x_match = re.search(r'PhysicalSizeX="([^"]+)"', tiff.ome_metadata)
            physical_size_y_match = re.search(r'PhysicalSizeY="([^"]+)"', tiff.ome_metadata)
            # Check if matches were found and extract the values as floats
            if physical_size_x_match:
                x = float(physical_size_x_match.group(1))
            else:
                x = None
            if physical_size_y_match:
                y = float(physical_size_y_match.group(1))
            else:
                y = None
        # Get pixel size for ImageJ-TIFF files
        elif tiff.imagej_metadata is not None:
            # Get tags
            tags = tiff.pages[0].tags
            # parse X, Y resolution
            y = _xy_voxel_size(tags, 'YResolution')
            x = _xy_voxel_size(tags, 'XResolution')

    # Check if x and y pixel sizes make sense
    if x is None or y is None:
        raise ValueError('Pixel spacing missing in the metadata for ' + str(filename))
    elif x != y:
        raise ValueError('Pixel spacing in x and y does not match for ' + str(filename))
    elif x == y == 1:
        raise ValueError('Pixel spacing in x and y is 1 (likely missing) for ' + str(filename))
    else:
        px_size = x
    # Translate pixel sizes from µm to standard SI units (meters)
    px_size_si = px_size * 1e-06
    if px_size < 1e-10:
        raise ValueError('Pixel size for the file ' + str(filename) +
                         '\nwas read from metadata as: ' + str(px_size) +
                         '\nAssuming units of µm and multiplying by 1e-06 yields a pixel size of ' +
                         str(px_size_si) + ' m (SI units)' + '\nThis is outside a reasonable range!')
    # Return 2d pixel size
    return px_size_si


def mrcnn_get_xy_shape(filename: Union[str, Path] = Path()) -> Tuple:
    # Set the size-limit to None (unlimited)
    Image.MAX_IMAGE_PIXELS = None
    with Image.open(filename) as im:
        # Get shape of the image without loading it
        shape = im.size
        # Rename output to prevent confusions with the different convention used in PILLOW and sci-kit image
        # PILLOW uses shape (x, y, z) and sci-kit image uses shape (y, x, z) = (row, col, z)
        y = shape[1]
        x = shape[0]
    return y, x


def calculate_output_shape_rescale(orig_shape: Tuple, scale: Union[int, float],
                                   channel_axis: int = None):
    """
    Calculate the new shape of a rescaled image without actually performing the rescaling.

    This function is a simplified version of skimage.transform.rescale and is used to determine the new shape
    of an image after rescaling. It is designed for cases where you want to calculate the output shape
    before applying the actual rescaling operation, which can be useful for memory and computation time savings.

    Parameters:
        orig_shape (Tuple): The original shape of the image as a tuple (height, width, [channels]).
        scale (Union[int, float]): The scaling factor to apply to the image dimensions. It can be a single
            value for uniform scaling or a sequence of values for non-uniform scaling.
        channel_axis (int, optional): The axis that represents channels in the image. If provided, this axis
            will not be scaled. Set to None for single-channel images. Default is None.

    Returns:
        Tuple: The new shape of the rescaled image as a tuple (new_height, new_width, [channels]).

    Note:
        This function is based on a simplified version of skimage.transform.rescale. The original source code
        can be found at https://github.com/scikit-image/scikit-image/blob/main/skimage/transform/_warps.py
        (lines 275 ff., as of May 18, 2023).
    """
    scale = np.atleast_1d(scale)
    multichannel = channel_axis is not None
    if len(scale) > 1:
        if ((not multichannel and len(scale) != len(orig_shape)) or
                (multichannel and len(scale) != len(orig_shape) - 1)):
            raise ValueError("Supply a single scale, or one value per spatial "
                             "axis")
        if multichannel:
            scale = np.concatenate((scale, [1]))
    output_shape = np.maximum(np.round(scale * orig_shape), 1)
    if multichannel:  # don't scale channel dimension
        output_shape[-1] = orig_shape[-1]
    return tuple(output_shape.astype(int))


def clear_border(im: np.ndarray):
    """
    Remove features that are directly adjacent to the border of a 2D binary image.

    This function processes a binary image and removes features (connected components)
    that touch the image border. It ensures that only features completely contained
    within the image area are retained.

    Parameters:
        im (numpy.ndarray): A 2D binary image where features to be cleared are represented as True (1) values,
            and the background is represented as False (0) values.

    Returns:
        numpy.ndarray: A modified binary image with features touching the border removed.

    Note:
        This function creates a padded copy of the input image, inverts it, and fills any holes
        to identify features connected to the border. It then removes these features from the
        original image.
    """
    # Get shape of given 2D array
    d1, d2 = im.shape
    # Create Boolean duplicate of im, padded with a line of True all around
    im2 = np.ones([d1 + 2, d2 + 2])
    im2 = im2.astype(bool)
    if im.dtype == 'bool':
        im2[1:-1, 1:-1] = im
    else:
        im2[1:-1, 1:-1] = im != 0
    # Invert and fill holes to keep only features that are connected to border
    im2 = binary_fill_holes(np.invert(im2))
    # Delete features connected to border from original image
    if im.dtype == 'bool':
        im = im & im2[1:-1, 1:-1]
    else:
        im = im * im2[1:-1, 1:-1].astype(im.dtype)
    return im


if __name__ == '__main__':
    pass
