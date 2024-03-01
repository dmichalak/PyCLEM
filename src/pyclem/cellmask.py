from pathlib import Path

import napari
import numpy as np
from aicsimageio import AICSImage
from aicsimageio.writers import OmeTiffWriter
from magicgui import magicgui
from skimage import transform

from pyclem.utils import shapes_to_mask


def cellmask_main():
    """
    Main function to run the cellmask widget in Napari.
    """
    # Create napari viewer
    viewer = napari.Viewer()
    # Add cell outline widget to napari viewer
    viewer.window.add_dock_widget(cellmask_widget, area='right', name='Cell Outline')
    napari.run()  # Continue after closing Napari (manually for now)


@magicgui(viewer={'visible': False, 'label': 'Napari Viewer'},
          rescale_to={'widget_type': 'Slider', 'name': 'rescale_px_size_to', 'min': 5, 'max': 20},
          call_button='Start')
def cellmask_widget(viewer: 'napari.viewer.Viewer',
                    rescale_to: int = 10,
                    filename: Path = Path(r'some\EM-file_inv.tif')):
    """
    A Napari widget for outlining cells and saving cell masks.

    This widget provides an interactive way to outline cells on an image.
    It will load a selected image and use the "Start" button to begin the cell outlining process.
    Once the outlining is done, clicking the "Finish & Save" button will save the results as a binary mask,
    before resetting the widget for the next use.

    Parameters
    ----------
    viewer : napari.viewer.Viewer
        The Napari viewer instance to which images and shapes will be added.
    rescale_to : int, optional
        Pixel size to which the image data should be rescaled (default: 10).
    filename : Path, optional
        The path to the EM image file to be loaded (default: Path('some\\EM-file_inv.tif')).

    Returns
    -------
    None
        This function does not return any value but handles the interactive cell outlining and saving process.

    Notes
    -----
    - Ensure the EM image file is compatible with the AICSImage class.
    - After clicking "Save Mask and Finish," the widget clears all layers and resets to "Start" mode.
    """

    # Initialize member variable for image files
    if 'im_file' not in cellmask_widget.__annotations__:
        cellmask_widget.__annotations__['im_file'] = None
    if 'cellmask_file' not in cellmask_widget.__annotations__:
        cellmask_widget.__annotations__['cellmask_file'] = None

    # Get current button mode
    mode = cellmask_widget._call_button.text

    if mode == 'Start':
        # Load image as AICSImage
        cellmask_widget.im_file = AICSImage(filename)

        # Check for existance of cellmask file
        fn_cellmask = filename.with_name(filename.stem + '_cellmask.tif')
        if fn_cellmask.exists():
            # Load cellmask as AICSImage
            cellmask_widget.cellmask_file = AICSImage(fn_cellmask)
        else:
            cellmask_widget.cellmask_file = None

        # Rescale image data to a pixel size of 10nm (to avoid extremely large images)
        im = transform.rescale(image=np.squeeze(cellmask_widget.im_file.data),
                               scale=cellmask_widget.im_file.physical_pixel_sizes[-1]/(rescale_to*1e-03),
                               order=1, preserve_range=True)
        # Add image to napari viewer
        viewer.add_image(im, name='EM Image', colormap='gray', interpolation2d='nearest')
        # Rescale and add cellmask, if available
        if cellmask_widget.cellmask_file is not None:
            cellmask = transform.rescale(image=np.squeeze(cellmask_widget.cellmask_file.data),
                                         scale=cellmask_widget.cellmask_file.physical_pixel_sizes[-1]/1e-02,
                                         order=0, preserve_range=True)
            # Add cellmask to napari viewer
            viewer.add_image(cellmask, name='Cell Mask Old', colormap='gray', interpolation2d='nearest', opacity=0.2)

        # Add shapes layer to napari viewer
        viewer.add_shapes(name='Cell Mask', shape_type='polygon', edge_width=10, edge_color='white',
                          face_color='white')
        # Change the button/mode for next run
        cellmask_widget._call_button.text = 'Finish and Save'

    elif mode == 'Finish and Save':
        # # Save cell outline as polygon with numpy
        # mask_shapes = {'cellmask': list_to_array(viewer.layers['Cell Mask'].data)}
        # fn_save = filename.with_name(filename.stem + '_cellmask.npz')
        # np.savez(str(fn_save), **mask_shapes)

        # Save binary cell mask using AICSImage
        mask = shapes_to_mask(mask_shape=viewer.layers['EM Image'].data.shape, shapes_layer=viewer.layers['Cell Mask'])
        # Resize mask to fit original image
        mask = transform.resize(image=mask, output_shape=cellmask_widget.im_file.shape[-2:],
                                order=0, preserve_range=True)
        # If available, combine with existing cellmask
        if cellmask_widget.cellmask_file is not None:
            mask = np.logical_or(mask, np.squeeze(cellmask_widget.cellmask_file.data).astype(bool))
        # Save mask as OME-TIFF
        fn_save = filename.with_name(filename.stem + '_cellmask.tif')
        OmeTiffWriter.save(data=np.uint8(mask*255), uri=fn_save,
                           physical_pixel_sizes=cellmask_widget.im_file.physical_pixel_sizes, overwrite_file=True)

        # Delete all layers
        viewer.layers.select_all()
        viewer.layers.remove_selected()
        # Reset image and cellmask to None
        cellmask_widget.im_file = None
        cellmask_widget.cellmask_file = None
        # change the button/mode for next run
        cellmask_widget._call_button.text = 'Start'


if __name__ == '__main__':
    cellmask_main()