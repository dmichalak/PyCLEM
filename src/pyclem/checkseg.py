import configparser
import warnings
from itertools import product
from pathlib import Path
from typing import Union, Tuple, List

import napari
import numpy as np
from aicsimageio import AICSImage
from magicgui import magicgui
from pyclem.utils import (handle_overlaps,
                          separate_mask_features,
                          shapes_to_mask,
                          apply_cellmask,
                          mask_to_shapes)
from skimage.morphology import remove_small_objects
from tifffile import tifffile

# Parameters:
MIN_AREA = 10e-03 ** 2 * np.pi  # Minimum area of a feature in µm^2 (default: circle with r = 8 nm)


def checkseg_main():
    viewer = napari.Viewer()
    viewer.window.add_dock_widget(checkseg_widget, area='right', name='Check Segmentation')
    napari.run()


@magicgui(
    viewer={'visible': False, 'label': 'Napari Viewer'},
    call_button='Start',
    tile_size={'widget_type': 'Slider', 'name': 'tile_size', 'min': 1, 'max': 10},
    remove_edge_feat={'widget_type': 'CheckBox', 'name': 'remove_edge_feat',
                      'label': 'Remove features that touch the edge of the cell mask'},
    redo={'widget_type': 'CheckBox', 'name': 'redo', 'label': 'Restart and ignore previous progress'},
)
def checkseg_widget(
        viewer: 'napari.viewer.Viewer',
        tile_size: int = 3,
        filename: Path = Path(r'some\EM-file_inv.tif'),
        remove_edge_feat: bool = False,
        redo: bool = False,
):
    """
    This widget allows you to perform segmentation checking.

    - Adjust the tile size using the slider.
    - Click 'Start' to begin the process.
    - Click 'Next' to proceed to the next step.
    etc.
    """
    # Initialize member variables, if they don't exist
    if 'ind' not in checkseg_widget.__annotations__:
        checkseg_widget.__annotations__['ind'] = None
    if 'tile_size_px' not in checkseg_widget.__annotations__:
        checkseg_widget.__annotations__['tile_size_px'] = None
    if 'centers' not in checkseg_widget.__annotations__:
        checkseg_widget.__annotations__['centers'] = None
    if 'px_size' not in checkseg_widget.__annotations__:
        checkseg_widget.__annotations__['px_size'] = None

    # Get current button mode
    mode = checkseg_widget._call_button.text

    if mode == 'Start':
        # Lock all widgets except the button
        checkseg_widget['tile_size'].enabled = False
        checkseg_widget['filename'].enabled = False
        checkseg_widget['remove_edge_feat'].enabled = False
        checkseg_widget['redo'].enabled = False

        # Prepare logical to organize workflow
        new_shapes = False
        progress_loaded = False
        # Initialize tile index
        checkseg_widget.ind = 0

        # Prepare shapes from scratch if shapes file for any class is missing
        shape_types = ['domes', 'flats', 'spheres']
        for shape_type in shape_types:
            fn_shapes = filename.with_name(filename.stem + f'_shapes_{shape_type}.csv')
            if not fn_shapes.exists():
                print('Prepare shapes for checking masks in CheckSegmentationGui.py')
                prep_for_gui(files=filename, parent_folder=filename.parent)
                # Create logical to mark that shapes were newly created
                new_shapes = True
                # Break loop to avoid multiple shape preparations
                break

        # Load progress from previous run
        if not new_shapes and not redo:
            fn_protocol = filename.with_name(filename.stem + '_segprotocol.txt')
            if fn_protocol.exists():
                config = configparser.ConfigParser()
                config.read(fn_protocol)
                if config.has_section('CHECKSEG'):
                    if ('current_ind' in config['CHECKSEG']) & ('tile_size' in config['CHECKSEG']):
                        checkseg_widget.ind = int(config['CHECKSEG']['current_ind'])
                        tile_size = int(config['CHECKSEG']['tile_size'])
                        checkseg_widget.tile_size.value = tile_size
                        progress_loaded = True

        # Load selected EM-image
        try:
            # Open EM image as AICSImage object and display in viewer
            em_file = AICSImage(filename)
            viewer.open(path=filename, plugin='napari-aicsimageio', name='EM image', interpolation2d='linear')
            # Get pixel size for EM image
            px_size = em_file.physical_pixel_sizes[-1]  # in µm
            checkseg_widget.px_size = px_size
        except FileNotFoundError:
            print(f'File not found: {filename}')
            return

        # Check for cellmask file and display in Viewer, if possible
        cellmask_fn = filename.with_name(filename.stem + '_cellmask.tif')
        if cellmask_fn.exists():
            cellmask = AICSImage(cellmask_fn)
            viewer.add_image(np.squeeze(cellmask.data), name='Cellmask', colormap='gray_r', blending='minimum',
                             scale=viewer.layers['EM image'].scale)
            # Remove features that touch the edge of the cellmask
            if remove_edge_feat:
                remove_edge_features(cellmask=np.squeeze(cellmask.data), feature_layer=viewer.layers['Domes'])
                remove_edge_features(cellmask=np.squeeze(cellmask.data), feature_layer=viewer.layers['Flats'])
                remove_edge_features(cellmask=np.squeeze(cellmask.data), feature_layer=viewer.layers['Spheres'])
        else:
            cellmask = None

        # Prepare grid and display in viewer
        grid, checkseg_widget.centers, checkseg_widget.tile_size_px = prepare_grid(
            im_shape=(em_file.dims.Y, em_file.dims.X),
            tile_size=tile_size,
            px_size=px_size)
        viewer.add_vectors(grid,
                           edge_color='yellow', edge_width=30,
                           name='Grid', vector_style='line',
                           scale=viewer.layers['EM image'].scale)

        # Prepare progress map and display in viewer
        progress_map = np.zeros(shape=(em_file.dims.Y, em_file.dims.X), dtype=np.uint8)
        viewer.add_image(progress_map,
                         name='Progress map', colormap='gray',
                         blending='additive', scale=viewer.layers['EM image'].scale)

        # Load shapes and display in viewer
        display_feature_shapes(viewer=viewer, file=filename)

        # skip to first tile that is not excluded by cellmask
        if not progress_loaded and cellmask is not None:
            checkseg_widget.ind = skip_empty_tiles(cellmask=np.squeeze(cellmask.data),
                                                   layer=viewer.layers['Progress map'],
                                                   centers=checkseg_widget.centers,
                                                   size=checkseg_widget.tile_size_px,
                                                   ind=checkseg_widget.ind,
                                                   last_ind=len(checkseg_widget.centers) - 1)

        # Mark skipped or previously checked tiles
        for i in range(checkseg_widget.ind):
            mark_tile(layer=viewer.layers['Progress map'],
                      center=checkseg_widget.centers[i],
                      size=checkseg_widget.tile_size_px)
        if checkseg_widget.ind < len(checkseg_widget.centers):
            # Move Napari viewer to current tile
            reset_view(viewer=viewer,
                       center=checkseg_widget.centers[checkseg_widget.ind],
                       size=checkseg_widget.tile_size_px,
                       scale=viewer.layers['EM image'].scale)
            # Change the button/mode for next run
            checkseg_widget._call_button.text = 'Next'
        else:
            # Center view on whole image
            reset_view(viewer=viewer,
                       center=tuple(value/2 for value in viewer.layers['EM image'].data.shape[-2:]),
                       size=np.max(viewer.layers['EM image'].data.shape[-2:]),
                       scale=viewer.layers['EM image'].scale)
            # Make grid and progress map invisible
            viewer.layers['Grid'].visible = False
            viewer.layers['Progress map'].visible = False
            # Change the button/mode for next run
            checkseg_widget._call_button.text = 'Save Mask and Finish'

    elif mode == 'Next':
        # Mark last tile as done in progress map
        mark_tile(layer=viewer.layers['Progress map'],
                  center=checkseg_widget.centers[checkseg_widget.ind],
                  size=checkseg_widget.tile_size_px)
        # Save feature shapes
        fn = Path(viewer.layers['EM image'].source.path)
        save_shapes(em_fn=fn,
                    domes_layer=viewer.layers['Domes'],
                    flats_layer=viewer.layers['Flats'],
                    spheres_layer=viewer.layers['Spheres'])
        # Save progress
        save_progress(fn_protocol=fn.with_name(fn.stem + '_segprotocol.txt'),
                      current_ind=checkseg_widget.ind+1,
                      tile_size=tile_size)

        # If we are not done yet, move to next tile
        if checkseg_widget.ind < len(checkseg_widget.centers)-1:
            # Check for existance of cellmask
            try:
                cellmask = viewer.layers['Cellmask'].data
            except KeyError:
                cellmask = None
            if cellmask is not None:
                # Increment the value of ind and skip empty tiles
                checkseg_widget.ind = skip_empty_tiles(cellmask=np.squeeze(viewer.layers['Cellmask'].data),
                                                       layer=viewer.layers['Progress map'],
                                                       centers=checkseg_widget.centers,
                                                       size=checkseg_widget.tile_size_px,
                                                       ind=checkseg_widget.ind + 1,
                                                       last_ind=len(checkseg_widget.centers) - 1)
            else:
                # Without cellmask, just increment the value of ind
                checkseg_widget.ind += 1
            reset_view(viewer=viewer, center=checkseg_widget.centers[checkseg_widget.ind],
                       size=checkseg_widget.tile_size_px, scale=viewer.layers['EM image'].scale)
        # After last tile, prepare for finishing
        else:
            # Make grid and progress map invisible
            viewer.layers['Grid'].visible = False
            viewer.layers['Progress map'].visible = False
            # Center view on whole image
            reset_view(viewer=viewer,
                       center=tuple(value/2 for value in viewer.layers['EM image'].data.shape[-2:]),
                       size=np.max(viewer.layers['EM image'].data.shape[-2:]),
                       scale=viewer.layers['EM image'].scale)
            # change the button/mode for next run
            checkseg_widget._call_button.text = 'Save Mask and Finish'

    elif mode == 'Save Mask and Finish':
        # Translate minimal feature size to pixels
        min_area_px = int(MIN_AREA / (checkseg_widget.px_size ** 2))

        # Check for existance of cellmask and remove edge features, if selected
        try:
            cellmask = viewer.layers['Cellmask'].data
        except KeyError:
            cellmask = None
        if remove_edge_feat and cellmask is not None:
            print('Checking for cellmask and removing edge features ...')
            # Remove edge features that overlap with cellmask
            remove_edge_features(cellmask=cellmask, feature_layer=viewer.layers['Domes'])
            remove_edge_features(cellmask=cellmask, feature_layer=viewer.layers['Flats'])
            remove_edge_features(cellmask=cellmask, feature_layer=viewer.layers['Spheres'])

        # Save feature shapes
        print('Saving segmentation shapes ...')
        fn = Path(viewer.layers['EM image'].source.path)
        save_shapes(em_fn=fn,
                    domes_layer=viewer.layers['Domes'],
                    flats_layer=viewer.layers['Flats'],
                    spheres_layer=viewer.layers['Spheres'])

        print('Creating segmentation mask from shapes ...')
        # Get mask shape before removing layers
        im_shape = viewer.layers['EM image'].data.shape
        # Remove unnecessary layers to save on memory
        for layer in ['Grid', 'Progress map', 'EM image']:
            try:
                viewer.layers.remove(layer)
            except ValueError:
                pass
        # Create segmentation mask from shapes
        mask = create_mask_from_shapes(mask_shape=im_shape,
                                       domes_layer=viewer.layers['Domes'],
                                       flats_layer=viewer.layers['Flats'],
                                       spheres_layer=viewer.layers['Spheres'],
                                       min_size=min_area_px)

        # Add cellmask to segmentation mask
        if cellmask is not None:
            print('Applying cellmask to new segmentation mask ...')
            mask = apply_cellmask(segmask=mask, cellmask=cellmask)

        # Save mask
        fn_mask = fn.with_name(fn.stem + '_segmask_check.tif')
        print(f'Saving new segmentation mask as {fn_mask} ...')
        save_mask(save_fn=fn_mask, mask=mask, exclude_alpha=True)

        # Reset viewer for next run
        print('Resetting viewer...')
        # Delete all existing layers
        viewer.layers.select_all()
        viewer.layers.remove_selected()
        # Set created attributes back to None
        checkseg_widget.ind = 0
        checkseg_widget.tile_size_px = None
        checkseg_widget.centers = None
        checkseg_widget.px_size = None
        # Unlock all widgets
        checkseg_widget['tile_size'].enabled = True
        checkseg_widget['filename'].enabled = True
        checkseg_widget['remove_edge_feat'].enabled = True
        checkseg_widget['redo'].enabled = True

        # Reset button/mode to 'Start'
        checkseg_widget._call_button.text = 'Start'
        print('Done!')


def skip_empty_tiles(cellmask: np.ndarray, layer: 'napari.Layer', centers: np.ndarray,
                     size: Union[float, int], ind: int, last_ind: int) -> int:
    """
    Checks if a tile is fully outside the cellmask.

    Parameters:
    -----------
    cellmask : np.ndarray
        The cellmask to check against.
    center : Tuple[Union[float, int], Union[float, int]]
        The center coordinates (row, column) of the square area to check.
    size : Union[float, int]
        The size of the square area to check.
    ind : int
        The index of the current tile.
    last_ind : int
        The index of the last tile.

    Returns:
    --------
    is_fully_outside : bool
        True if the tile is fully outside the cellmask, False otherwise.
    """
    # Make sure cellmask is a boolean array
    if cellmask.dtype != bool:
        cellmask = cellmask.astype(bool)
    # Initialize boolean variable
    exit_loop = False
    while not exit_loop:
        # Check if tile at current index is fully outside the cellmask
        x_min = max(int(centers[ind, 1] - size / 2), 0)
        x_max = min(int(centers[ind, 1] + size / 2), cellmask.shape[-1])
        y_min = max(int(centers[ind, 0] - size / 2), 0)
        y_max = min(int(centers[ind, 0] + size / 2), cellmask.shape[-2])
        # Check if all pixels in the tile are outside the cellmask
        is_fully_outside = np.all(cellmask[y_min:y_max, x_min:x_max])
        # If the tile is fully outside, mark it and increment index
        if is_fully_outside and ind < last_ind:
            mark_tile(layer=layer, center=centers[ind], size=size)
            ind += 1
        else:
            exit_loop = True
    return ind


def mark_tile(layer: 'napari.Layer', center: Tuple[Union[float, int], Union[float, int]],
              size: Union[float, int], value: int = 32) -> None:
    """
    Adds a constant value to a square area of a Napari image layer, effectively marking a processed tile.

    Parameters:
    -----------
    layer : napari.Layer
        The Napari image layer to which the constant value will be added.
    center : Tuple[Union[float, int], Union[float, int]]
        The center coordinates (row, column) of the square area to mark.
    size : Union[float, int]
        The size of the square area to mark.
    value : int, optional
        The constant value to add to the marked area (default: 32).

    Returns:
    --------
    None

    Notes:
    ------
    - This function modifies the provided Napari image layer by setting a square region,
     centered at the given coordinates, to the specified constant value.
    - The size parameter defines the side length of the square area to mark.
    - The value parameter determines the value of marked pixels.
    - After calling this function, the image layer will be updated with the marked tile.
    """
    # Get image data
    data = layer.data
    # Get indices of pixels to be marked
    x_min = max(int(center[1] - size / 2), 0)
    x_max = min(int(center[1] + size / 2), data.shape[-1])
    y_min = max(int(center[0] - size / 2), 0)
    y_max = min(int(center[0] + size / 2), data.shape[-2])
    # Mark area
    data[..., y_min:y_max, x_min:x_max] = value
    # Update image layer
    layer.data = data


def reset_view(viewer: 'napari.Viewer', center: Tuple[Union[float, int], Union[float, int]],
               size: Union[float, int], scale: Tuple[Union[float, int], Union[float, int]] = None) -> None:
    """
    Reset the view of a Napari viewer to a specified center and size.

    Parameters:
        viewer (napari.Viewer):
            The Napari viewer instance for which the view will be reset.

        center (Tuple[Union[float, int], Union[float, int]]):
            The new center coordinates (x, y) to center the view on.

        size (Union[float, int]):
            The new size of the view. The zoom level will be adjusted to include an area of 1.2 * size.

        scale (Tuple[Union[float, int], Union[float, int]]):
            The scale of the viewer layers. Defaults to None.

    Returns:
        None

    This function was copied and adapted from the Napari plugin "affinder" by Juan Nunez-Iglesias.
    Copied on Aug 7, 2023.
    URL of the original GitHub project is https://github.com/jni/affinder.
    """
    if scale is None:
        scale = (1, 1)
    if viewer.dims.ndisplay != 2:
        return  # Todo: Make this work for n-dimensional images? (e.g. 3D)
        #          For insights, see https://github.com/jni/affinder
    # Center viewer on new position
    viewer.camera.center = center * scale
    # Set zoom to include area of 1.2 * size
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        canvas_size = viewer._canvas_size
        zoom_scaling = np.mean(scale)  # Todo: Check if there is a better way to implement scaling
    viewer.camera.zoom = np.min(canvas_size) / np.max(1.2 * size * zoom_scaling)


def create_mask_from_shapes(mask_shape: Tuple[int, int], domes_layer: 'napari.Layer',
                            flats_layer: 'napari.Layer', spheres_layer: 'napari.Layer',
                            min_size: int = 100) -> np.ndarray:
    """
       Create a composite mask from shape layers representing different classes.

       Parameters:
              mask_shape (Tuple[int, int]): The 2D shape of the mask to be created.
           domes_layer (napari.Layer):
               Napari shape layer containing dome shapes.
           flats_layer (napari.Layer):
               Napari shape layer containing flat shapes.
           spheres_layer (napari.Layer):
               Napari shape layer containing sphere shapes.
           min_size (int, optional):
               Minimum area threshold for keeping mask features. Defaults to MIN_AREA.

       Returns:
           np.ndarray:
               A composite mask combining shapes from different layers. Features are assigned different color channels
               (dome: red, flat: green, sphere: blue). Overlapping features are resolved, and the mask is separated
               into distinct classes. An alpha channel is added to indicate valid structure regions. Features below
               the size threshold are removed, and the final mask is returned with the alpha channel.

       Note:
           This function combines shapes from different classes into a composite mask and processes the mask to ensure
           distinct classes, proper separation, and removal of small structures. The final mask includes an alpha
           channel to indicate valid structure regions.
       """
    # Create new mask from shapes layers
    dome_mask = shapes_to_mask(mask_shape=mask_shape, shapes_layer=domes_layer)
    flat_mask = shapes_to_mask(mask_shape=mask_shape, shapes_layer=flats_layer)
    sphere_mask = shapes_to_mask(mask_shape=mask_shape, shapes_layer=spheres_layer)
    # Assemble to 3-layer boolean mask
    mask = np.stack((dome_mask, flat_mask, sphere_mask), axis=2)
    # Handle potentially overlapping features of different class
    mask[:, :, 0], mask[:, :, 1], mask[:, :, 2] = handle_overlaps(domes=mask[:, :, 0],
                                                                  flats=mask[:, :, 1],
                                                                  spheres=mask[:, :, 2])
    # Make sure features of different class (e.g. domes & spheres) do not touch directly
    mask = separate_mask_features(mask)
    # Create boolean alpha-channel (set to 1 everywhere)
    alpha = np.full((mask.shape[0], mask.shape[1]), True)
    # Set alpha-channel to 0 at places without structures
    alpha[mask.sum(axis=-1) == 0] = False
    # Delete features below size-threshold (using alpha-channel)
    alpha = remove_small_objects(alpha, min_size=min_size, connectivity=1)
    for d in range(3):
        mask[:, :, d] = mask[:, :, d] * alpha  # apply deletions to structure classes
    # Make mask a uint8 RGB array and add alpha channel to mask
    mask = np.uint8(mask * 255)
    mask = np.concatenate((mask, alpha[:, :, np.newaxis]), axis=-1)
    # Return mask
    return mask


def save_mask(save_fn: Union[Path, str], mask: np.ndarray,
              exclude_alpha: bool = False) -> None:
    """
    Save a mask array as an image file.

    Parameters:
        save_fn (Union[Path, str]):
            The original image file name or path.
        mask (np.ndarray):
            The mask array to be saved as an image.
        exclude_alpha (bool, optional):
            Whether to exclude the alpha channel from the saved image. Defaults to False.

    Returns:
        None:
            The function does not return any value.

    Notes:
        - If `exclude_alpha` is True and the mask has an alpha channel, it will be removed before saving.
        - The mask is saved as a uint8.
    """
    # Optionally, remove alpha-channel before saving
    if exclude_alpha and mask.shape[2] > 3:
        mask = mask[:, :, :-1]
    # Make sure mask is uint8
    if mask.dtype != np.uint8:
        mask = np.uint8(mask)
    # Save mask using tifffile
    tifffile.imwrite(save_fn, mask, imagej=True, compression='zlib')


def save_shapes(em_fn: Union[str, Path],  domes_layer: 'napari.Layer',
                flats_layer: 'napari.Layer', spheres_layer: 'napari.Layer') -> None:
    """
    Use Napari builtins to save shapes layers in csv files.

    Parameters:
        em_fn (Union[str, Path]): Source path of the EM image.
        domes_layer ('napari.Layer'): The layer containing dome shapes.
        flats_layer ('napari.Layer'): The layer containing flat shapes.
        spheres_layer ('napari.Layer'): The layer containing sphere shapes.

    Returns:
        None: The function does not return any value.
    """
    # Make sure em_fn is a Path object
    em_fn = Path(em_fn)
    # Save Napari shapes layers
    for layer in [domes_layer, flats_layer, spheres_layer]:
        if layer.data:
            layer.save(em_fn.with_name(em_fn.stem + f'_shapes_{layer.name.lower()}.csv').as_posix())


def save_progress(fn_protocol: Union[Path, str], current_ind: int, tile_size: int) -> None:
    """
    Save the current index and tile size to segprotocol file.

    Parameters:
        fn_protocol(Union[Path, str]): The filename or path to save the progress to.
        current_ind (int): The index of the current tile.
        tile_size (int): The size of the tiles in µm units.

    Returns:
        None: The function does not return any value.
    """
    # Prepare config parser
    config = configparser.ConfigParser()
    if fn_protocol.exists():
        config.read(fn_protocol)
    if not config.has_section('CHECKSEG'):
        config.add_section('CHECKSEG')
    config.set('CHECKSEG', 'current_ind', str(current_ind))
    config.set('CHECKSEG', 'tile_size', str(tile_size))
    with open(fn_protocol, 'w') as configfile:
        config.write(configfile)


def display_feature_shapes(viewer: 'napari.Viewer',
                           file: Union[Path, str] = None,
                           shape_types: List[str] = ['flats', 'domes', 'spheres']) -> None:
    """
    Add shapes to a Napari viewer.

    If the shapes dictionary is empty, empty shapes with appropriate names are added to the viewer.
    Otherwise, shapes from the dictionary are added with appropriate colors and names.

    Parameters:
        viewer (napari.Viewer):
            The Napari viewer instance to which the shapes will be added.

        file (Union[Path, str]):
            File path to the EM image. Default is None.

        shape_types (List[str]):
            List of shape types to display. Default is ['flats', 'domes', 'spheres'].

    Returns:
        None
    """
    # Prepare shape files
    shape_files = []
    if file is not None:
        for shape_type in shape_types:
            if file.with_name(file.stem + f'_shapes_{shape_type}.csv').exists():
                shape_files.append(file.with_name(file.stem + f'_shapes_{shape_type}.csv'))
            else:
                shape_files.append(None)
    else:
        shape_files = [None] * len(shape_types)

    # Prepare display parameters
    par_mapping = {
        'domes': ('darkred', 'red', 3, 0.4),
        'flats': ('darkgreen', 'green', 3, 0.4),
        'spheres': ('darkblue', 'blue', 3, 0.4),
    }

    # Loop over shape files
    for shape_type, shape_file in zip(shape_types, shape_files):
        if shape_file is not None:
            # Prepare layer name
            layer_name = shape_type.capitalize()
            # Prepare color and other parameters
            edge_c, face_c, edge_w, opacity = par_mapping.get(layer_name.lower(), ('darkgrey', 'grey', 3))
            # Add shapes to viewer
            try:
                viewer.open(path=shape_file, name=layer_name, layer_type='shapes', opacity=opacity,
                            edge_width=edge_w, edge_color=edge_c, face_color=face_c,
                            scale=viewer.layers['EM image'].scale)
            except IndexError:
                # Add empty layer instead
                print(f'Could not load shapes from {shape_file}.\n'
                      f'This file likely contains no shapes to display.\n'
                      f'Adding empty shapes layer instead.')
                # Add empty shapes to viewer
                viewer.add_shapes(shape_type='polygon', name=shape_type.capitalize(), opacity=opacity,
                                  edge_width=edge_w, edge_color=edge_c, face_color=face_c,
                                  scale=viewer.layers['EM image'].scale)
        else:
            # Prepare color and other parameters
            edge_c, face_c, edge_w, opacity = par_mapping.get(shape_type, ('darkgrey', 'grey', 3))
            # Add empty shapes to viewer
            viewer.add_shapes(shape_type='polygon', name=shape_type.capitalize(), opacity=opacity,
                              edge_width=edge_w, edge_color=edge_c, face_color=face_c,
                              scale=viewer.layers['EM image'].scale)


def remove_edge_features(cellmask: np.ndarray, feature_layer: 'napari.Layer'):
    """
    Removes features in a napari shapes layer that contain any vertices outside the image or that overlap with the
    area outside the cell as marked by the cellmask.
    """
    # Make sure cellmask is binary
    cellmask = cellmask.astype(bool)
    y, x = np.squeeze(cellmask.shape)

    # Prepare logical list for overlaps (rows: features)
    is_edge_feature = []
    # Loop over features
    for f, feature in enumerate(feature_layer.data):
        # Get pixel coordinates for this feature
        feature_px = feature.astype(int)
        # Check if any of the feature pixels are outside the image or inside the cellmask
        if (np.any(feature_px[:, 0] < 0) or np.any(feature_px[:, 0] > y) or
                np.any(feature_px[:, 1] < 0) or np.any(feature_px[:, 1] > x) or
                np.any(cellmask[feature_px[:, 0], feature_px[:, 1]])):
            # Select feature for removal
            feature_layer.selected_data.add(f)
    # Remove edge features
    feature_layer.remove_selected()


def prep_for_gui(files: Union[List[Union[str, Path]], str, Path], parent_folder: Union[str, Path],
                 line_density: float = 1 / 10):
    """
    Preprocess feature contours from mask files and save them as numpy arrays in NPZ format.

    Parameters:
        files (Union[List[Union[str, Path]], str, Path]):
            A list of filenames or a single filename (str or Path object) containing mask files.
        parent_folder (Union[str, Path]):
            The parent folder path where the mask files are located.
        line_density (float, optional):
            The line density parameter for the mask_to_shapes function, controlling the number of vertices
            for each shape. Default is 1/10.

    Notes:
        - This function takes mask files as input, processes the feature contours from the masks using the
          mask_to_shapes function, and saves them as numpy arrays in NPZ format.
        - If multiple files are provided in the 'files' parameter, the function processes each one in the list.
    """
    # Ensure that files is a list
    if isinstance(files, (str, Path)):
        files = [files]

    for file in files:
        # Ensure that file is a Path object (absolute path)
        file = Path(parent_folder, file).resolve()
        print(f'Preparing feature contours for file: {file}')

        # Open corresponding mask as numpy array
        if file.with_name(file.stem + '_segmask_check.tif').exists():
            mask_file = file.with_name(file.stem + '_segmask_check.tif')
        elif file.with_name(file.stem + '_segmask.tif').exists():
            mask_file = file.with_name(file.stem + '_segmask.tif')
        else:
            raise FileNotFoundError(f'Could not find a mask for {file}')
        mask = np.squeeze(AICSImage(mask_file).data)

        # If necessary, check for existence of cellmask and apply it to the segmask
        if not np.any(np.all(mask == [255, 255, 255], axis=2)):
            cellmask_file = file.with_name(file.stem + '_cellmask.tif')
            if cellmask_file.exists():
                cellmask = np.squeeze(AICSImage(cellmask_file).data)
                mask = apply_cellmask(cellmask=cellmask, segmask=mask)
                # Make sure mask is uint8
                if mask.dtype != np.uint8:
                    mask = np.uint8(mask)
                # Save mask using tifffile
                tifffile.imwrite(mask_file, mask, imagej=True, compression='zlib')

        # Set "outside" pixels (i.e., pixels where all rgb values are 255) to 0
        # (this is to avoid issues with the mask_to_shapes function)
        indices = np.where(np.all(mask == (255, 255, 255), axis=2))
        mask[indices[0], indices[1], :] = 0

        # Translate mask to shapes and save in Napari-readable format
        shape_classes = ['domes', 'flats', 'spheres']
        for i in range(mask.shape[2]):
            shape_data = mask_to_shapes(mask[:, :, i], rho=line_density)
            # Save shapes as csv file
            shape_data.to_csv(file.with_name(file.stem + f'_shapes_{shape_classes[i]}.csv'), index=False)


def prepare_grid(im_shape: Tuple[int, int], tile_size: float, px_size: float) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Creates a grid of vectors that can be overlaid onto an image to split it into tiles in Napari.

    Parameters:
        im_shape (tuple[int, int]): The shape of the input image (height, width).
        tile_size (float): The desired size of each tile in physical units (e.g., micrometers).
        px_size (float): The pixel size in physical units (e.g., micrometers) for converting tile_size to pixels.

    Returns:
        Tuple[np.ndarray, np.ndarray, int]: A tuple containing three NumPy arrays.
                                            The first array, grid_vectors, is a 3-dimensional array of shape
                                            (num_tiles_x + num_tiles_y + 2, 2, 2) representing the grid vectors
                                            for splitting the image into tiles.
                                            The second array, centers, is a 2-dimensional array of shape
                                            (num_tiles_x * num_tiles_y, 2) representing the coordinates of the centers
                                            of each tile in the image.
                                            The third element of the tuple is an integer representing the size of each
                                            tile in pixels.
    """
    # Translate tile_size from SI units to pixels
    tile_size = np.round(tile_size / px_size).astype(int)
    # Get number of tiles in each dimension
    num_tiles = np.floor(np.array(im_shape) / tile_size).astype(int)
    # Create coordinate "tics"
    tics_x = np.linspace(0, num_tiles[1] * tile_size, num_tiles[1]+1)
    tics_x = np.append(tics_x, im_shape[1])
    tics_y = np.linspace(0, num_tiles[0] * tile_size, num_tiles[0]+1)
    tics_y = np.append(tics_y, im_shape[0])
    # Create grid vectors
    grid_vectors = np.zeros((num_tiles[1] + num_tiles[0] + 4, 2, 2))
    # Add start points for vectors
    grid_vectors[:len(tics_x), 0, 1] = tics_x
    grid_vectors[:len(tics_x) + 1, 0, 0] = 0
    grid_vectors[len(tics_x):, 0, 1] = 0
    grid_vectors[len(tics_x):, 0, 0] = tics_y
    # Add actual vectors
    grid_vectors[:len(tics_x), 1, 1] = 0
    grid_vectors[:len(tics_x) + 1, 1, 0] = im_shape[0]
    grid_vectors[len(tics_x):, 1, 1] = im_shape[1]
    grid_vectors[len(tics_x):, 1, 0] = 0
    # Create an array containing the centers of each box
    centers = []
    for a, b in product(range(len(tics_x)-1), range(len(tics_y)-1)):
        center_x = (tics_x[a] + tics_x[a + 1]) / 2
        center_y = (tics_y[b] + tics_y[b + 1]) / 2
        centers.append((center_y, center_x))
    centers = np.array(centers)
    # Return grid vectors, centers, and tile size in px
    return grid_vectors, centers, tile_size


if __name__ == '__main__':
    checkseg_main()
