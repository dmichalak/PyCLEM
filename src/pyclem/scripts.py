import re
import shutil
from pathlib import Path
from typing import Union, List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from aicsimageio import AICSImage
from aicsimageio.writers import OmeTiffWriter
from skimage.feature import match_template
from skimage.io import imread, imsave
from skimage.util import invert
from tifffile import tifffile

from pyclem.analyse import crop_to_features
from pyclem.io import get_files, get_subdir
from pyclem.stats import get_mrcnn_stats, get_feature_stats, get_brightness_stats, summarize_stats
from pyclem.utils import auto_contrast_3d, apply_cellmask


def batch_crop_to_features(analysis_dir: Union[str, Path],
                           pattern: str = r'\d{8}.*cell\d{3}_inv\.tif',
                           mask_suffix: str = '_segmask_check.tif',
                           crop_size: Tuple[float, float] = (1.5, 1.5)) -> None:
    """
    Find samples that have both a (per default manually checked) segmentation mask and correlated fluorescent images.
    Then crop stacks of individual features from mask and images based on segmentation mask.

    :param analysis_dir: directory to search for samples
    :param pattern: regular expression pattern to identify samples
    :param mask_suffix: suffix of the segmentation mask file
    :param crop_size: size of the crop region in µm (height, width)

    :return: Files are directly saved to the analysis directory.
    """
    # Make sure the directory name is a Path object
    analysis_dir = Path(analysis_dir)
    if not analysis_dir.is_dir():
        raise ValueError('The specified path is not a directory!')
    # Get all samples in the directory
    samples, _ = get_files(pattern=pattern, subdir=analysis_dir)

    # Loop over samples
    for sample in samples:
        # Get full path for this sample
        fn = Path(analysis_dir, sample)
        # Try to load segmentation mask
        try:
            segmask = AICSImage(fn.with_name(fn.stem + mask_suffix))
        except FileNotFoundError:
            print(f'Desired segmentation mask (suffix: {mask_suffix}) missing for {sample} !\n'
                  'Skipping to next sample ...')
            continue

        # Search for correlated images (marked by '_TT' in the filename) for this sample
        corr_pattern = pattern.replace(r'_inv\.tif', r'.*_TT\.tif')
        corr_files, _ = get_files(pattern=corr_pattern, subdir=fn.parent)
        # If there are correlated images, load them and extract the image modalities from suffixes
        if len(corr_files) > 0:
            corr_images = [AICSImage(Path(fn.parent, corr_file)) for corr_file in corr_files]
            corr_modes = [re.search(r'cell\d+_(.*?)_TT.tif', corr_file)[1] for corr_file in corr_files]
        else:
            print(f'No correlated images found for {sample} !\n'
                  f'Skipping to next sample ...')
            continue

        # Load EM image for this sample
        em = AICSImage(fn)
        # Calculate crop size in pixels based on the pixel size of the EM image
        crop_shape = (np.array(crop_size) / em.physical_pixel_sizes.X).astype(int)
        # Add EM image to list of images to crop
        corr_images.append(em)
        corr_modes.append('em')

        # Crop images based on segmentation mask
        print(f'Cropping images for {sample} ...')
        crop_to_features(mask=np.squeeze(segmask.data),
                         images=[np.squeeze(corr_image.data) for corr_image in corr_images],
                         target_dir=fn.parent, image_mode=corr_modes, crop_shape=crop_shape)


def batch_apply_cellmask(analysis_dir: Union[str, Path],
                         pattern: str = r'\d{8}.*cell\d{3}_inv\.tif',
                         remove_edge_features: bool = False) -> None:
    """
    Applies cell mask (if available) to all segmentation masks (if available) in a given directory.
    """
    # Make sure the directory name is a Path object
    analysis_dir = Path(analysis_dir)
    if not analysis_dir.is_dir():
        raise ValueError('The specified path is not a directory!')

    # Get all samples in the directory
    samples, _ = get_files(pattern=pattern, subdir=analysis_dir)
    # Loop over samples
    for sample in samples:
        # Load cell mask
        try:
            fn_cellmask = Path(analysis_dir, sample).with_name(Path(sample).stem + '_cellmask.tif')
            cellmask = AICSImage(fn_cellmask)
        except FileNotFoundError:
            print(f'Cellmask file missing for {sample} !\n'
                  f'Skipping to next sample ...')
            continue
        # Load ai-generated segmentation mask
        try:
            fn_segmask = Path(analysis_dir, sample).with_name(Path(sample).stem + '_segmask.tif')
            segmask = AICSImage(fn_segmask)
        except FileNotFoundError:
            print(f'AI-generated segmentation mask file missing for {sample} !')
            segmask = None
        # Load manually checked segmentation mask
        try:
            fn_segmask_check = Path(analysis_dir, sample).with_name(Path(sample).stem + '_segmask_check.tif')
            segmask_check = AICSImage(fn_segmask_check)
        except FileNotFoundError:
            print(f'Manually checked segmentation mask file missing for {sample} !\n')
            segmask_check = None
        # Apply cellmask to segmentation masks
        print(f'Applying cellmask for {sample} ...')
        for mask, maskname in zip([segmask, segmask_check], [fn_segmask, fn_segmask_check]):
            if mask is not None:
                # Apply cellmask
                mask = apply_cellmask(cellmask=np.squeeze(cellmask.data).astype(bool),
                                      segmask=np.squeeze(mask.data),
                                      remove_edge_features=remove_edge_features)
                # Save mask
                tifffile.imwrite(maskname, mask, imagej=True, compression='zlib')


def batch_get_stats(analysis_dir: Union[str, Path],
                    pattern: str = r'\d{8}.*cell\d{3}_inv\.tif',
                    do_mrcnn_stats: bool = True,
                    do_feature_stats: bool = True,
                    do_brightness_stats: bool = True,
                    do_summary: bool = True) -> None:
    """
    Get statistics of accuracy metrics for mrcnn-segmentation for all samples in a given directory.
    Results are saved as csv-files in the respective folders.
    """
    # Make sure the directory name is a Path object
    analysis_dir = Path(analysis_dir)
    if not analysis_dir.is_dir():
        raise ValueError('The specified path is not a directory!')
    # Get all samples in the directory
    samples, _ = get_files(pattern=pattern, subdir=analysis_dir)

    # Loop over samples
    for sample in samples:
        # Get full path for this sample
        fn = Path(analysis_dir, sample)
        # Get stats for this sample
        try:
            if do_mrcnn_stats:
                print(f'Get segmentation accuracy stats for {sample} ...')
                stats = get_mrcnn_stats(file=fn)
                # Save stats as csv-file
                stats.to_csv(fn.with_name(fn.stem + '_mrcnn-stats.csv'))
            if do_feature_stats:
                print(f'Get feature stats for {sample} ...')
                stats, stats_detail = get_feature_stats(file=fn)
                # Save stats as csv-file
                stats.to_csv(fn.with_name(fn.stem + '_feature-stats.csv'))
                stats_detail.to_csv(fn.with_name(fn.stem + '_feature-stats-detail.csv'))
            if do_brightness_stats:
                print(f'Get brightness stats for {sample} ...')
                stats = get_brightness_stats(fn=fn, image_modes=['sola488', 'epi488'], save_bkg_mask=True)
                # Save stats as csv-file
                stats.to_csv(fn.with_name(fn.stem + '_brightness-stats.csv'))
        except FileNotFoundError:
            print(f'Necessary files missing for {sample} !')
            continue

    # Summarize statistics for all samples
    if do_summary:
        summarize_stats(analysis_dir=analysis_dir, sample_pattern=pattern,
                        do_mrcnn_stats=do_mrcnn_stats, do_feature_stats=do_feature_stats,
                        do_brightness_stats=do_brightness_stats)


def invert_and_save(fn: Union[Path, str, List[Path], List[str]]) -> None:
    """
    Loads the given image file(s) using AICSImageIO, inverts the pixel values using Sci-kit Image,
    and saves the inverted image(s) as OME TIFF.

    Parameters:
        fn (Union[Path, str, List[Path], List[str]]): The filename or list of filenames of the image(s) to be inverted.
                                                     It can be a single filename (str or Path), or a list of filenames.
                                                     Supported file formats are those that can be read by AICSImageIO.

    Returns:
        None: The function does not return any values. The inverted image is saved as an OME TIFF file.
    """
    # Ensure fn is a list of Path objects
    if isinstance(fn, (str, Path)):
        fn = [Path(fn)]
    elif isinstance(fn, (list, tuple)):
        fn = [Path(file) for file in fn]

    # Loop over fn
    for file in fn:
        # Load file as AICSImage object
        image = AICSImage(file)
        # Save inverted image (including pixel sizes)
        new_fn = file.with_name(file.stem + '_inv' + file.suffix)
        OmeTiffWriter.save(data=invert(image.data), uri=new_fn,
                           physical_pixel_sizes=image.physical_pixel_sizes)


def find_and_label_cells(overview_files: Union[str, Path, List[Union[str, Path]]] = None,
                         line_width: int = 15,
                         color: Union[str, int] = 'black',
                         font_size: int = 50,
                         contrast_limits: Tuple[float, float] = (0, 0.98)) -> None:
    """
       Automatically locate and annotate cells in overview images.

       Parameters:
           overview_files (Union[str, Path, List[Union[str, Path]], optional):
               A single overview image file or a list of overview image files (as strings or Path objects).
           line_width (int, optional):
               Width of the line used for annotating cell borders. Default is 15.
           color (Union[str, int], optional):
               Color for annotating cell borders. 'black' or 0 for black, 'white' or 1 for white.
               Default is 'black'.
           font_size (int, optional):
               Font size for cell labels. Default is 50.
           contrast_limits (Tuple[float, float], optional):
               Contrast limits for auto-scaling overview images before annotation. Should be a tuple of
               (low limit, high limit). Default is (0.05, 0.99).

       Notes:
           - This function processes overview images and their corresponding cell images, locating cells in
             the overview and annotating them with borders and labels.
           - Annotated overview images are saved as TIFF files with '_annotated' appended to the original filename.
           - Each cell is located using template matching.
           - Cell borders are annotated in the specified color with the specified line width.
           - Cell labels are added with the specified font size.

       Example:
           # Annotate cells in a single overview image
           find_and_label_cells("overview_image.tif")

           # Annotate cells in a list of overview images
           find_and_label_cells(["overview1.tif", "overview2.tif"])
       """

    # Make sure we are handling a list of Path objects
    if not isinstance(overview_files, list):
        overview_files = [Path(overview_files)]
    # Make sure list entries are path objects
    overview_files = [Path(file) if not isinstance(file, Path) else file for file in overview_files]

    # Loop over selected files (samples)
    for overview_file in overview_files:
        print(f'Annotating cells in {overview_file} ...')
        # Get file names for the corresponding cell images
        cell_name_pattern = re.sub(r'overview', r'cell[0-9][0-9][0-9]', overview_file.name)
        cell_files, _ = get_files(pattern=cell_name_pattern, subdir=overview_file.parent)
        # Load overview image
        overview = imread(overview_file)
        # Autoscale in order to save as jpeg later
        overview = auto_contrast_3d(input_im=overview, low=contrast_limits[0], high=contrast_limits[1])
        # Loop over cells for this file (sample)
        for cell_file in cell_files:
            # Load cell image
            cell = imread(Path(overview_file.parent, cell_file))
            y_width, x_width = cell.shape

            # Find cell in overview image
            result = match_template(image=overview, template=cell)
            # Get coordinates of upper left corner
            xy1 = np.unravel_index(np.argmax(result), result.shape)
            # Calculate the lower right corner of the square
            xy2 = [xy1[0] + x_width, xy1[1] + y_width]

            # Prepare color for drawing
            if color == 'white' or color == 1:
                color = np.iinfo(overview.dtype).max
            elif color == 'black' or color == 0:
                color = 0

            # Draw the border of the cell image with the specified line width
            overview[xy1[0]:xy1[0] + line_width, xy1[1]:xy2[1]] = color
            overview[xy2[0] - line_width:xy2[0], xy1[1]:xy2[1]] = color
            overview[xy1[0]:xy2[0], xy1[1]:xy1[1] + line_width] = color
            overview[xy1[0]:xy2[0], xy2[1] - line_width:xy2[1]] = color

            # Prepare label for this cell
            cell_name = re.search(r'cell[0-9][0-9][0-9]', cell_file)
            cell_name = cell_name[0]
            # Create a blank image
            label_im = Image.fromarray(cell * 0 + (1-color))
            # Get a drawing context
            draw = ImageDraw.Draw(label_im)
            # Prepare font
            font = ImageFont.truetype("arial.ttf", font_size)
            # Calculate the position to center the text
            xy_label = (line_width + 5, line_width + 5)
            # Draw the text on the image
            draw.text(xy_label, cell_name, font=font, fill=0)
            # Convert the image back to a NumPy array
            label_im = np.array(label_im)
            # Add to overview image
            overview[xy1[0]:xy2[0], xy1[1]:xy2[1]] = overview[xy1[0]:xy2[0], xy1[1]:xy2[1]] * label_im

        # Convert to uint8 before saving
        overview = (overview * 255).astype(np.uint8)
        # After finding and marking all cells, save annotated overview file as jpg
        new_name = overview_file.with_name(overview_file.stem + r'_annotated.jpg')
        imsave(new_name, overview, check_contrast=False)


def add_matching_files(target_dir: Union[str, Path],
                       source_dir: Union[str, Path],
                       extension: Union[List[str], str] = '.tif') -> None:
    """
    Copy matching files from the source directory to the target directory.

    Parameters:
        target_dir (Union[str, Path]):
            The target directory where missing files will be copied.
        source_dir (Union[str, Path]):
            The source directory containing the files to be copied.
        extension (Union[List[str], str], optional):
            A string or list of strings representing the file extensions to consider.
            Default is '.tif'.

    Notes:
        - This function looks for subdirectories in the target directory and copies files from the
          source directory that match the subdirectory name and the specified file extensions.
        - Files are copied only if they do not already exist in the target directory.
    """
    # make sure extensions is a list
    if not isinstance(extension, list):
        extension = [extension]
    # Get list of subdirectories in target_dir
    subdirs = get_subdir(root_dir=target_dir)
    # Loop over subdirectories
    for subdir in subdirs:
        source_files = []
        for ext in extension:
            # Get list of files in source_dir matching the subdirectory name and the given extensions
            pattern = subdir[:re.search(r'.*cell[0-9]*', subdir).end()] + r'.*' + ext
            tmp, _ = get_files(pattern=pattern, subdir=source_dir, id_dict=False)
            source_files.extend(tmp)
        # Copy files to target_dir
        for file in source_files:
            # prepare absolute names for copying file
            old_name = Path(source_dir, file)
            new_name = Path(target_dir, subdir, old_name.name)
            # copy file
            if not new_name.exists():
                shutil.copy(old_name, new_name)


def move_files_to_dest(pattern: str, source: Union[str, Path] = Path(), destination: Union[str, Path] = Path(),
                       id_dict: bool = False, copy: bool = True, string_to_exclude_from_subdir: str = '_inv',
                       make_subdir: bool = False):
    """
    Finds all files matching the specified regular expression file "pattern" (folder structure does not matter) in the
    "source" folder and moves them into the "destination" folder. New sub-folders in the "destination" folder are
    created based on the base filenames of the files.

    Parameters:
        pattern (str): Regular expression file pattern to match (e.g., "20230531_.*cell[0-9]{3}\.tif")
        source (str): Source folder to search for files (default: current directory)
        destination (str): Destination folder to move the files (default: current directory)
        id_dict (bool): If `True`, return IDs as a dict. Only works for named groups in pattern.
                        --> see get_files() (default: False)
        copy (bool): Flag indicating whether to copy the files instead of moving them (default: True)
        string_to_exclude_from_subdir (str): Substring to exclude from the base filenames when creating subdirectories
                                             (default: '_inv')
        make_subdir (bool): Flag indicating whether to create subdirectories in the destination folder (default: False)

    Returns:
        None. Files are moved/copied to the destination folder.
    """
    files, _ = get_files(pattern=pattern, subdir=source, id_dict=id_dict)
    for file in files:
        # Make sure file is a Path() object
        file = Path(file)
        # Prepare absolute path for file
        old_name = Path(source, file)
        # Prepare absolute path for new file
        if make_subdir:
            # Get base filename for creating subdirectory
            subdir_name = file.stem.replace(string_to_exclude_from_subdir, '')
            # If necessary, create new subdirectory
            Path(destination, subdir_name).mkdir(parents=True, exist_ok=True)
            new_name = Path(destination, subdir_name, file.name)
        else:
            new_name = Path(destination, file.name)
        # Move file
        if not new_name.exists():
            if not copy:
                shutil.move(old_name, new_name)
            else:
                shutil.copy(old_name, new_name)


def split_nd2_files(fn: Union[List[Path], Path, List[str], str],
                    channel: int = 0,
                    create_analysis: bool = True) -> None:
    """
    Splits a list of nd2 files into single tif images and saves them in a new directory.

    Parameters:
        fn (Union[List[Path], Path, List[str], str]):
            A list of nd2 file paths, a single nd2 file path, a list of nd2 file names, or a single nd2 file name.
        channel (int):
            The channel index (0-based) to extract from the nd2 files. Default is 0.
        create_analysis (bool):
            Logical to decide whether to create a new "analysis" directory in the same directory as the nd2 files.
            Default is False.

    Returns:
        None: The function does not return any value, but saves the split tif images in the (new) "analysis" directory.
    """
    # Ensure nd2_files is/are a list of Path objects
    if isinstance(fn, (str, Path)):
        fn = [Path(fn)]
    elif isinstance(fn, (list, tuple)):
        fn = [Path(file) for file in fn]

    for file in fn:
        # Get sample name from filename (everything before last underscore in filename)
        sample_name = file.stem.rsplit('_', 1)[0]
        # Get image modality for this file (str after last underscore in filename)
        image_modality = file.stem.split('_')[-1]
        # If necessary, prepare directory to save single tif images
        if file.parent.name == 'NSTORM':
            parent_dir = file.parent.parent
        else:
            parent_dir = file.parent
        if create_analysis:
            sample_dir = Path(parent_dir, 'analysis', sample_name)
        else:
            sample_dir = Path(parent_dir, sample_name)
        sample_dir.mkdir(parents=True, exist_ok=True)

        # Open the nd2 file using AICSImage
        nd2_file = AICSImage(file)
        # Loop over all cells
        for c, _ in enumerate(nd2_file.scenes):
            # Get image for this cell
            nd2_file.set_scene(c)
            cell_image = nd2_file.data[:, channel, ...]
            # Save the image in the new directory as OME TIFF
            tif_path = Path(sample_dir, f'{sample_name}_cell{c + 1:03d}_{image_modality}.tif')
            OmeTiffWriter.save(data=cell_image, uri=tif_path,
                               physical_pixel_sizes=nd2_file.physical_pixel_sizes)


if __name__ == '__main__':
    pass