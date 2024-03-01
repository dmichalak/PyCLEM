from pathlib import Path
from typing import List, Union, Tuple

import numpy as np
from skimage.measure import label, regionprops
from tifffile import tifffile


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
                                                                #  Probably better to do this in stats calculation.
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


if __name__ == '__main__':
    pass
