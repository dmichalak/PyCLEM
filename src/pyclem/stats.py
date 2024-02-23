import re
from math import ceil, floor
from pathlib import Path
from typing import Union, Tuple, List

import numpy as np
import pandas as pd
from aicsimageio import AICSImage
from skimage.measure import label, regionprops
from tifffile import tifffile

from pyclem.io import get_files
from pyclem.utils import divide, intersect2d, polygon_dilation


def get_mrcnn_stats(file: Union[str, Path] = Path(), conf_int: float = 0.05) -> pd.DataFrame:
    """
    Generates statistics on how well MaskRCNN segmented clathrin when compared to manual segmentation.
    """
    # Initialize counters for summary statistics p, tp, fp and fn
    # Index in list corresponds to three structure types [domes, flats, spheres]
    p = [0, 0, 0]
    tp = [0, 0, 0]
    fp = [0, 0, 0]
    fn = [0, 0, 0]
    # Get ai-proposed and ground truth (manually edited) masks
    file_mask_old = Path(file.parent, file.stem + '_segmask.tif')
    file_mask_new = Path(file.parent, file.stem + '_segmask_check.tif')
    # Load arrays and transform to binary masks (to save memory)
    old = np.squeeze(AICSImage(file_mask_old).data).astype(bool)
    new = np.squeeze(AICSImage(file_mask_new).data).astype(bool)
    # Remove cellmask by setting "outside" pixels to False
    if np.any(np.all(new == [True, True, True], axis=2)):
        old[np.all(new == (True, True, True), axis=2)] = False
        new[np.all(new == (True, True, True), axis=2)] = False

    # Loop over structure types (domes, flats, spheres)
    for ind in range(3):
        # Match structures of ai-proposed mask (old) with structures in ground truth mask (new) to get tp and fp
        tp[ind], fp[ind] = match_masks(mask_a=old[:, :, ind], mask_b=new[:, :, ind], conf=conf_int)
        # Reverse matching to get fn
        _, fn[ind] = match_masks(mask_a=new[:, :, ind], mask_b=old[:, :, ind], conf=conf_int)
        # Calculate ground truth number of positives (could also be derived directly from new but this is faster)
        p[ind] = tp[ind] + fn[ind]

    # Calculate true positive ration TPR = tp/p and false discovery rate FDR = fp/tp+fp
    # see Powers (2011), Int. J. Mach. Learn. Tech.
    tpr = [divide(a=tp_i, b=p_i) for tp_i, p_i in zip(tp, p)]
    fdr = [divide(a=fp_i, b=tp_i + fp_i) for fp_i, tp_i in zip(fp, tp)]
    # Summarize in pandas data frame
    stats = pd.DataFrame(data={'p': p, 'tp': tp, 'fp': fp, 'fn': fn, 'TPR': tpr, 'FDR': fdr},
                         index=['domes', 'flats', 'spheres'])
    return stats


def match_masks(mask_a: np.ndarray, mask_b: np.ndarray, conf: float) -> (int, int, int, int):
    """
    Find the number of features in mask A that have an equivalent in mask B and the number of features in mask A
    that don't have an equivalent in mask B.

    Parameters:
        mask_a (np.ndarray): Binary mask array representing features A.
        mask_b (np.ndarray): Binary mask array representing features B.
        conf (float): Confidence level used to determine equivalent features based on the overlapping area.

    Returns:
        tuple: A tuple containing two integers, (num_a_in_b, num_a_not_in_b), where:
               - num_a_in_b (int): The number of features in mask A that have an equivalent in mask B.
               - num_a_not_in_b (int): The number of features in mask A that don't have an equivalent in mask B.
    """
    # label mask A to get number of features and define properties of individual features
    mask_a_label, num_a = label(mask_a, return_num=True)
    prop_a = list(regionprops(mask_a_label))
    # Clear memory
    del mask_a_label, mask_a
    # label mask B to get number of features and define properties of individual features
    mask_b_label, num_b = label(mask_b, return_num=True)
    prop_b = list(regionprops(mask_b_label))
    # Clear memory
    del mask_b_label, mask_b

    # Loop over features in mask
    match = [False for n in range(num_a)]
    for a, feat_a in enumerate(prop_a):
        # Get pixel coordinates for this feature
        coords_a = feat_a.coords
        # Loop over ground truth features
        for b, feat_b in enumerate(prop_b):
            # Get pixel coordinates ground truth feature
            coords_b = feat_b.coords
            # Get overlapping coordinates
            overlap = intersect2d(coords_a, coords_b)
            if not not overlap.any():
                # Calculate area of overlapping region
                area_overlap = len(overlap)
                # Get area of ground truth feature
                area_b = feat_b.area
                # Check if overlapping area is within confidence interval of ground truth area
                if area_overlap in pd.Interval(floor((1 - conf) * area_b), ceil((1 + conf) * area_b)):
                    # Mark this feature as matched in list A
                    match[a] = True
                    # Remove matched feature from list B
                    prop_b.pop(b)
                    # Break out of loop B and continue with next feature in list A
                    break
    # return number of features in a that have an equivalent in b and number of those that don't
    return sum(match), sum([not elem for elem in match])


def get_feature_stats(file: Union[str, Path],
                      mask_suffix: str = '_segmask_check.tif',
                      feature_class: Tuple[str, str, str] = ('dome', 'flat', 'sphere')
                      ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Get statistics on the number of features in a mask file.

    Parameters:
        file (Union[str, Path]): Path to the image file.
        mask_suffix (str, optional): Suffix for the mask file. Default is '_segmask_check.tif'.
        feature_class (Tuple[str, str, str], optional): Tuple of feature classes. Default is ('dome', 'flat', 'sphere').

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing two DataFrames:
            1. Summary statistics for the image.
            2. Detailed statistics for individual features.
    """
    # Make sure fn is a Path object
    file = Path(file)
    # Load mask
    try:
        mask = np.squeeze(AICSImage(file.with_name(file.stem + mask_suffix)).data)
    except FileNotFoundError:
        raise FileNotFoundError(f'Could not find mask file (suffix: {mask_suffix}) for {file.name}.\n'
                                f'Make sure it is in the same folder as the EM image file!')

    # Initialize dictionary to store sum statistics and list for individual feature statistics
    stats = {}
    stats_detail = []
    # Get pixel size from image metadata
    stats['px_size'] = AICSImage(file).physical_pixel_sizes.X

    # Get size of analysis area (cell)
    if np.any(np.all(mask == [255, 255, 255], axis=2)):
        stats['cell_area'] = np.sum(~np.all(mask == [255, 255, 255], axis=2))
        # Remove cellmask for remaining analysis
        mask[np.all(mask == [255, 255, 255], axis=2)] = 0
    elif file.with_name(file.stem + '_cellmask.tif').exists():
        cellmask = np.squeeze(AICSImage(file.with_name(file.stem + '_cellmask.tif')).data)
        stats['cell_area'] = np.sum(~cellmask)
    else:  # If no cellmask is available, set cell area to whole image area
        stats['cell_area'] = mask.size / 3

    # Loop over feature classes to get individual feature statistics
    for f, feat_class in enumerate(feature_class):
        features = regionprops(label(mask[:, :, f]))
        for feat in features:
            stats_detail.append(
                pd.DataFrame({'class': feat_class, 'size': feat.area}, index=[0])
            )
    # Transform list of individual feature statistics to pandas data frame
    stats_detail = pd.concat(stats_detail, ignore_index=True)

    # Get sum statistics
    for f, feat_class in enumerate(feature_class):
        # Get number of features
        stats[f'{feat_class}_num'] = stats_detail[stats_detail['class'] == feat_class]['size'].count()
        # Get area covered by this feature class
        stats[f'{feat_class}_area'] = stats_detail[stats_detail['class'] == feat_class]['size'].sum()
        # Get average feature size
        stats[f'{feat_class}_avg_size'] = stats_detail[stats_detail['class'] == feat_class]['size'].mean()
    # Get total number of features
    stats['total_num'] = stats_detail['size'].count()
    # Get total area covered by features
    stats['total_area'] = stats_detail['size'].sum()
    # Transform dictionary to pandas data frame
    stats = pd.DataFrame(stats, index=[file.name])

    return stats, stats_detail


def get_brightness_stats(fn: Union[str, Path],
                         image_modes: Union[str, List[str]],
                         stack_pattern: str = r'.*_.*_stack.*\.tif$',
                         psf_dist: Union[float, int] = 0.2,
                         iso_dist: Union[float, int] = 0.5,
                         save_bkg_mask: bool = False) -> pd.DataFrame:
    """
    Extract brightness statistics from stacks of cropped images.

    Parameters:
        fn (Union[str, Path]): Path to the sample directory containing the stacks.
        image_modes (Union[str, List[str]]): Imaging modes of fluorescent images to consider.
        stack_pattern (str, optional): Regular expression pattern for stack files. Default is r'.*_.*_stack.*\.tif$'.
        psf_dist (Union[float, int], optional): Distance for dilating masks (in µm). Default is 0.2.
        iso_dist (Union[float, int], optional): Minimal distance between central feature and closest neighbor to
                                                define central feature as isolated (in µm]. Default is 0.5.
        save_bkg_mask (bool, optional): Whether to save background masks. Default is False.

    Returns:
        pd.DataFrame: DataFrame containing brightness statistics for each feature.

    Raises:
        FileNotFoundError: If no matching stack files are found.
    """
    # Make sure fn is a Path object
    fn = Path(fn)
    # Get pixel size from image metadata and translate distances to pixels
    px_size = AICSImage(fn).physical_pixel_sizes.X
    psf_dist = int(psf_dist / px_size)
    iso_dist = int(iso_dist / px_size)

    # Get list of image stacks and make Path objects
    stack_files, _ = get_files(pattern=stack_pattern, subdir=fn.parent)
    stack_files = [Path(fn.parent, f) for f in stack_files]
    # Check if any stacks were found
    if not stack_files:
        raise FileNotFoundError(f'No files with cropped features found in {fn.parent} matching pattern {stack_pattern}')

    # Limit to stacks of masks and stacks of images with the specified image mode(s)
    if not isinstance(image_modes, list):
        image_modes = [image_modes]
    stack_files = [f for f in stack_files if any([mode in f.stem for mode in image_modes]) or '_mask' in f.stem]

    # Check if any stacks are left
    if not stack_files or all(['_mask' in f.stem for f in stack_files]):
        raise FileNotFoundError(f'No files with cropped features found in {fn.parent} matching pattern {stack_pattern} '
                                f'and given image mode {image_modes}')

    # Initialize list to store results
    stats = []
    # Identify different feature classes
    keyword_pattern = r'^(.+)_.*_stack.*\.tif$'
    feature_classes = list(set([re.match(keyword_pattern, f.name).group(1) for f in stack_files]))

    # Loop over feature classes
    for feat_class in feature_classes:
        # Get all stacks for this feature class
        feat_stacks = [f for f in stack_files if feat_class in f.name]
        # Load mask stack
        mask_stack_file = next((f for f in feat_stacks if '_mask' in f.stem), None)
        if mask_stack_file is None:
            print('No mask stack found for feat_class {feat_class} in {fn.parent}.\n'
                  'Skipping to next feature class.')
            continue
        else:
            mask_stack = np.squeeze(AICSImage(mask_stack_file).data)
        # Load image stacks
        im_stack_files = [f for f in feat_stacks if '_mask' not in f.stem]
        im_stacks = [np.squeeze(AICSImage(f).data) for f in im_stack_files]
        im_modes = [re.search(f'{feat_class}_(.*)_stack', f.stem)[1] for f in im_stack_files]

        # Expand mask stack to 4D if necessary
        if mask_stack.ndim == 3:
            mask_stack = np.expand_dims(mask_stack, axis=0)
            im_stacks = [np.expand_dims(im_stack, axis=0) for im_stack in im_stacks]
        # Create new empty numpy stack to store masks used for extracting brighntess statistics
        if save_bkg_mask:
            bkg_stack = np.zeros(mask_stack.shape[:-1], dtype=bool)
        # Loop over images in stack
        for frame in range(mask_stack.shape[0]):
            # Create new dictionary to store statistics for this frame
            stats_frame = {'class': feat_class, 'frame': frame, 'px_size': px_size}
            # Get mask for this frame
            mask = mask_stack[frame, ...]

            # Check if the central feature in this frame is isolated
            labeled_mask, num_feat = label(np.any(mask != 0, axis=2), connectivity=1, return_num=True)
            if num_feat == 1:
                central_label = 1
            else:
                central_label = labeled_mask[int(labeled_mask.shape[0]//2), int(labeled_mask.shape[1]//2)]
            stats_frame['isolated'] = is_isolated(label_tile=labeled_mask, central_feat_nr=central_label,
                                                  min_dist=iso_dist)
            stats_frame['iso_dist'] = iso_dist

            # Transform to binary mask of central feature
            mask = labeled_mask == central_label
            # Add size of central feature to dictionary
            stats_frame['area'] = np.sum(mask)

            # Create dilated mask to extract brightness statistics
            mask_signal = polygon_dilation(binary_mask=mask, dist=psf_dist)
            mask_bkg = polygon_dilation(binary_mask=mask, dist=1.25*psf_dist) & ~mask_signal
            # Loop over image stacks
            for ind, im in enumerate([im_stack[frame] for im_stack in im_stacks]):
                # Get mean signal and background intensity
                stats_frame[f'{im_modes[ind]}_signal'] = np.mean(im[mask_signal])
                stats_frame[f'{im_modes[ind]}_bkg'] = np.mean(im[mask_bkg])

            # Transform dictionary to pandas data frame and append to list
            stats.append(pd.DataFrame(stats_frame, index=[0]))
            # Store used mask_bkg in bkg_stack (for later inspection)
            if save_bkg_mask:
                bkg_stack[frame, ...] = mask_bkg
        # Save bkg_stack as tif-stack
        if save_bkg_mask:
            tifffile.imwrite(mask_stack_file.with_name(mask_stack_file.name.replace('_mask', '_bkg-mask')),
                             (bkg_stack * 255).astype('uint8'), imagej=True, compression='zlib')

    # Concatenate list of data frames to one data frame
    stats = pd.concat(stats, ignore_index=True)
    return stats


def crop_to_features(mask: np.ndarray, images: List[np.ndarray],
                     target_dir: Union[str, Path] = Path(),
                     image_mode: List[str] = None,
                     feature_class: List[str] = None,
                     crop_shape: Tuple[int, int] = (800, 800)) -> None:
    """
    Automatically locate features in a mask and crop corresponding images.

    Parameters:
        mask (np.ndarray): RGB mask used to identify and crop around individual features.
        images (List[np.ndarray]): List of additional images to crop.
        target_dir (Union[str, Path], optional): Target directory to save the results. If not provided, a file dialog
            will be displayed to select the directory.
        image_mode (List[str], optional): List of image modalities corresponding to the images for cropping.
        feature_class (List[str], optional): List of names for the feature classes.
        crop_shape (Tuple[int, int], optional): Shape of the crop region (height, width).

    Returns:
        None: The function saves the cropped masks and images in the specified target directory.

    Notes:
        - This function processes a mask and crops corresponding images based on detected features in the mask.
        - Cropped masks and images are saved in the target directory.

    Example:
        # Crop images based on a mask
        crop_to_features(mask_array, image_list)
    """

    # Check if suffixes were specified
    if image_mode is None:
        raise ValueError('Suffix has to be defined.')
    # Check if there is a suffix for every image
    if images is not None and len(images) != len(image_mode):
        raise ValueError('Numbers of provided images and suffixes do not match.')
    # Check if there is are specified names for feature classes.
    if feature_class is None:
        # If not use standard nomenclature
        feature_class = ['dome_', 'flat_', 'sphere_']

    # Make sure target_dir is a Path object
    target_dir = Path(target_dir)
    # Prepare directory to save results
    target_dir.mkdir(parents=True, exist_ok=True)

    # Check if mask is uint8 or bool (to later exclude tiles with features too close to the cell edge)
    if mask.dtype == np.uint8:
        cell_mask = (255, 255, 255)
    elif mask.dtype == np.bool:
        cell_mask = [True, True, True]
    else:
        raise NotImplementedError('Only uint8 and bool masks are supported.')
    # Label all clathrin features in the mask (without cell mask)
    feat_mask = mask.copy()
    feat_mask[np.all(feat_mask == cell_mask, axis=2)] = 0
    label_mask = label(np.any(feat_mask != 0, axis=2), connectivity=1)
    del feat_mask

    # Loop over feature classes
    for d in range(3):
        # Get statistics about all features of this class
        stats = regionprops(label_mask * (mask[:, :, d] != 0))
        # Get size of whole mask
        imsize_y, imsize_x, _ = mask.shape
        # Loop over all features
        for feature_nr, feature in enumerate(stats):
            # In first iteration, create stacks for storing cropped masks and images
            if feature_nr == 0:
                mask_stack = []
                if images is not None:
                    image_stacks = [[] for _ in range(len(images))]
            # Get center of this feature
            center = feature.centroid
            # Get box around feature
            x_start = np.floor(max(0, center[1] - crop_shape[1] / 2)).astype(int)
            x_end = np.floor(min(imsize_x, center[1] + crop_shape[1] / 2)).astype(int)
            y_start = np.floor(max(0, center[0] - crop_shape[0] / 2)).astype(int)
            y_end = np.floor(min(imsize_y, center[0] + crop_shape[0] / 2)).astype(int)
            # Crop mask
            crop_mask = mask[y_start:y_end, x_start:x_end, :]

            # Skip if this tile is too close to image edge
            if np.any((y_end-y_start, x_end-x_start) != crop_shape):
                continue
            # Skip if this feature is too close to cell edge
            if np.any(np.all(crop_mask == cell_mask, axis=2)):  # Todo: Change criterion to distance of central feature?
                continue

            # Add current cropped mask to list
            # noinspection PyUnboundLocalVariable
            mask_stack.append(crop_mask)
            # Crop images and add to lists
            if images is not None:
                for ind, im in enumerate(images):
                    # Crop image
                    crop_im = im[y_start:y_end, x_start:x_end]
                    # Add current cropped image to stack
                    # noinspection PyUnboundLocalVariable
                    image_stacks[ind].append(crop_im)

        # Save resulting mask and image stacks, if any features were found
        if not mask_stack:
            print(f'No {feature_class[d][:-1]}s found, fullfilling all selection criteria in {target_dir} !')
        else:
            # Translate lists of cropped mask and images to numpy stacks
            mask_stack = np.stack(mask_stack, axis=0)  # keeping data type the same as when loaded
            for i in range(len(image_stacks)):
                image_stacks[i] = np.stack(image_stacks[i], axis=0)  # keeping data type the same as when loaded

            # Save mask stack as tif-stack
            tifffile.imwrite(
                Path(target_dir, feature_class[d] + 'mask' + '_stack' + '.tif'),
                mask_stack, imagej=True, compression='zlib')
            # Save image stacks as tif-stacks
            for ind, im_stack in enumerate(image_stacks):
                if im_stack.any():
                    tifffile.imwrite(
                        Path(target_dir, feature_class[d] + image_mode[ind] + '_stack' + '.tif'),
                        im_stack, imagej=True, compression='zlib')


def is_isolated(label_tile: np.ndarray[int, 2], central_feat_nr: int, min_dist: Union[float, int]) -> bool:
    """
    Check if a mask tile contains an isolated feature at the center.

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


if __name__ == '__main__':
    pass