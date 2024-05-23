import re
from math import ceil, floor
from pathlib import Path
from typing import Union, Tuple, List

import numpy as np
import pandas as pd
from aicsimageio import AICSImage
from skimage.measure import label, regionprops
from tifffile import tifffile

from pyclem.file_io import get_files
from pyclem.utils import divide, intersect2d, polygon_dilation, is_isolated


def summarize_stats(analysis_dir: Union[str, Path],
                    sample_pattern: str = r'\d{8}.*cell\d{3}_inv\.tif',
                    do_mrcnn_stats: bool = True,
                    do_feature_stats: bool = True,
                    do_brightness_stats: bool = True) -> None:
    """
    Summarizes accuracy statistics for all files in a given directory.
    """
    # Make sure the directory name is a Path object
    analysis_dir = Path(analysis_dir)
    if not analysis_dir.is_dir():
        raise ValueError('The specified path is not a directory!')

    # Summarize mrcnn stats
    if do_mrcnn_stats:
        print('Summarizing mrcnn stats ...')
        # Get existing stats files in the directory
        mrcnn_pattern = sample_pattern.replace(r'\.tif', r'_mrcnn-stats\.csv')
        stats_files, _ = get_files(pattern=mrcnn_pattern, subdir=analysis_dir)
        if not stats_files:
            print(f'No mrcnn-stats files found in the directory {analysis_dir}!')
        else:
            # Create summary dataframe
            summary = pd.DataFrame({
                'p': [0, 0, 0],
                'tp': [0, 0, 0],
                'fp': [0, 0, 0],
                'fn': [0, 0, 0]})
            # Create list to store names of the files included in summary
            included_files = []

            # Loop over stats files and add to summary
            for stats_file in stats_files:
                # Load stats file
                stats = pd.read_csv(Path(analysis_dir, stats_file))
                # Store sample name
                included_files.append(stats_file)
                # Add to summary
                summary['p'] = summary['p'] + stats['p']
                summary['tp'] = summary['tp'] + stats['tp']
                summary['fp'] = summary['fp'] + stats['fp']
                summary['fn'] = summary['fn'] + stats['fn']
            # Calculate TPR and FDR for summary statistics
            summary['TPR'] = summary.apply(lambda row: divide(a=row['tp'], b=row['p']), axis=1)
            summary['FDR'] = summary.apply(lambda row: divide(a=row['fp'], b=row['tp'] + row['fp']), axis=1)

            # Save dictionary as csv-file
            save_name = Path(analysis_dir, 'mrcnn-stats_summary.csv')
            summary.to_csv(save_name, index=False)
            # Save list of included files
            with open(save_name.with_name(save_name.stem + '_included_files.txt'), 'w') as f:
                for file in included_files:
                    f.write(file + '\n')

    # Summarize feature stats
    if do_feature_stats:
        print('Summarizing feature stats ...')
        # Get existing stats files in the directory
        feature_pattern = sample_pattern.replace(r'\.tif', r'_feature-stats\.csv')
        stats_files, _ = get_files(pattern=feature_pattern, subdir=analysis_dir)
        if not stats_files:
            print(f'No feature-stats files found in the directory {analysis_dir}!')
        else:
            # Create list to summarize feature stats
            summary = []
            # Loop over stats files and add to summary
            for stats_file in stats_files:
                # Load stats file
                stats = pd.read_csv(Path(analysis_dir, stats_file), index_col=0)
                # Add to summary
                summary.append(stats)
            # Concatenate all stats files
            summary = pd.concat(summary, ignore_index=False)
            # Save as csv-file
            save_name = Path(analysis_dir, 'feature-stats_summary.csv')
            summary.to_csv(save_name, index=True)

    # Summarize brightness stats
    if do_brightness_stats:
        print('Summarizing brightness stats ...')
        # Get existing stats files in the directory
        brightness_pattern = sample_pattern.replace(r'\.tif', r'_brightness-stats\.csv')
        stats_files, _ = get_files(pattern=brightness_pattern, subdir=analysis_dir)
        if not stats_files:
            print(f'No brightness-stats files found in the directory {analysis_dir}!')
        else:
            # Create list to summarize brightness stats
            summary = []
            # Loop over stats files and add to summary
            for stats_file in stats_files:
                # Load stats file
                stats = pd.read_csv(Path(analysis_dir, stats_file), index_col=0)
                # Set index name to sample name
                stats.index = [Path(stats_file).stem.replace('_brightness-stats', '.tif')] * len(stats)
                # Add to summary
                summary.append(stats)
            # Concatenate all stats files
            summary = pd.concat(summary, ignore_index=False)
            # Save as csv-file
            save_name = Path(analysis_dir, 'brightness-stats_summary.csv')
            summary.to_csv(save_name, index=True)


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
                stats_frame['isolated'] = True
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


if __name__ == '__main__':
    pass