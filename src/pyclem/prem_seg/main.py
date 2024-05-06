import os
import shutil
import tkinter.filedialog as fd
from argparse import Namespace
from pathlib import Path
from tkinter import Tk

from prem_seg.utils import (get_files,
                            mrcnn_preprocess,
                            mrcnn_postprocess,
                            mrcnn_prep_trainingdata)
from prem_seg.worm.PR_segmentation import run_segmentation

# TODO: implement a proper parameter input. This could eventually be handled in a GUI or something like that.
# Parameters for PR_segmentation.py:
#################################################
# Initial directory for file dialog (your general data directory e.g. L-drive)
startdir = r'Z:\AA_AMA\data\Quantitative_CLEM\20240318_QCLEM_MDA_CLCa-GFP_sola\analysis'  # e.g. r'L:/AA_AMA/data'
# Directory containing Mask R-CNN code. This should be on the same internal hard-drive as the used CUDA library.
# Otherwise, the segmentation software does not find the CUDA library for some reason.
root_dir = os.path.abspath("E:/prem-seg")  # e.g. "E:/prem-seg"
# Regular expression pattern to pick file used for evaluation or training
EM_file_pattern = r'^.*cell\d{3}.*_inv\.tif$'  # e.g. r'^.*cell\d{3}.*_inv\.tif$' corresponds to '***cell012***_inv.tif'

# Decide if you want to delete individual image tiles after processing
delete_tiles = True
# Path to trained weights file
model_path = r'./Models/mask_rcnn_pr_0600.h5'
# Tell Mask R-CNN what to do
command = 'evaluate'  # 'train' or 'evaluate'
# Regular expression pattern to load tiled images
FILE_EXT = 'x[0-9][0-9][0-9]y[0-9][0-9][0-9].tif'
# Directory to save logs and model checkpoints
default_dataset_year = "2014"
# Default limit for number of images to analyze with Mask R-CNN (???)
im_limit = 500
# Number of GPUs to be used
gpu_number = 1

# Parameters for pre-processing:
###################################################
# Define size and overlap of individual tiles
tile_size = 600
tile_overlap = 200
# Define target pixelsize for Mask R-CNN segmentation. This is based on the 15k magnification used on the "old"
# EM-camera. The factor 2 is necessary to ensure that enough image tiles contain more than one CCP structure.
pixelsize_mrcnn = 2*1.23e-09  # in m
# Define c_limit (in fractions of max) for auto-contrast as tuple (lower_limit, upper_limit)
c_limit = (0.05, 0.998)
# Minimal size of features to keep in post-processing (in px)
min_feature_size = 100
# Sub-folder created for image tiles
tile_dir_name = 'Maskrcnn_evaluation'
# Sub-folder created for tile segmentation masks
result_dir_name = 'Maskrcnn_results'

# Parameters for generating training-data:
#####################################################
# Define how file-extensions are changed in training data
# first list entry: image file-ext being replaced
# second list entry: file-ext of masks used for training
train_ext = ('inv.tif', 'goodRGBmask.tif')
# Sub-folder created for cropped image/mask pairs used for training
train_dir_name = 'Maskrcnn_training'
# Offset from feature edge used for cropping
offset = 200
# Step-size to increase offset if necessary
offset_step = 10
# Maximum size for features to be removed from borders of cropped images
max_size_remove = 1000  # This corresponds to ~ 77x77 nm at a mrcnn_px_size of 2.46e-09 m


#############################################################################################################
# Main body of AI-based PR-Segmentation:
#############################################################################################################
def main():
    # Select directory containing EM-files to segment (will walk through sub-folders and search for EM_file_pattern)
    root = Tk()
    root.withdraw()
    parent_dir = fd.askdirectory(title='Select Parent Folder Containing EM-Images to Process',
                                 initialdir=startdir)
    # make sure the path matches the used OS
    parent_dir = Path(parent_dir)
    # Change working directory to folder containing the EM-images
    # (to prevent potential problems when saving filenames)
    os.chdir(parent_dir)
    # walk through parent_dir to find all files matching EM_file_pattern
    filenames, _ = get_files(pattern=EM_file_pattern, subdir=parent_dir)

    # Prepare files for segmentation or training, respectively
    if command == 'evaluate':
        print('Run Mask R-CNN pre-processing!')
        mrcnn_preprocess(files=filenames, tile_size=tile_size, tile_overlap=tile_overlap, px_size_mrcnn=pixelsize_mrcnn,
                         contrast_limits=c_limit, parent_folder=parent_dir, tile_dir=tile_dir_name)
    elif command == 'train':
        # Todo: Adapt mrcnn_prep_trainingdata to work with the new folder structure!
        print('Run Mask R-CNN preparation of training data!')
        # mrcnn_prep_trainingdata(files=filenames, parent_dir=parent_dir, train_dir='Maskrcnn_training',
        #                         contrast_limits=c_limit, train_ext=train_ext, px_size_mrcnn=pixelsize_mrcnn,
        #                         offset=offset, offset_step=offset_step, max_tile_size=tile_size,
        #                         max_size_border_feature=max_size_remove)
        print('This part of the code is not yet adapted to the new folder structure!')
    else:
        exit('Command not recognized! Use "train" or "evaluate".')

    # Prepare arguments for Mask R-CNN segmentation
    print('Run Mask R-CNN segmentation!')
    # Prepare path to save Mask R-CNN logs
    logs_path = Path(parent_dir, 'Mask-RCNN_logs')
    if command == 'evaluate':
        # Prepare path to image tiles
        tile_dir = Path(parent_dir, tile_dir_name)
        # Prepare directory to save results
        result_dir = Path(parent_dir, result_dir_name)
        # Create argument input for PR_segmentation.py
        args = Namespace(command=command,
                         dataset=str(tile_dir),
                         download=False,
                         filter=FILE_EXT,
                         gpu=gpu_number,
                         limit=im_limit,
                         logs=str(logs_path),
                         model=str(model_path),
                         resultFolder=str(result_dir),
                         year=default_dataset_year)
    elif command == 'train':
        # Prepare path to text-file containing training pairs
        list_path = Path(parent_dir, train_dir_name, 'training_list.txt')
        # Create argument input for PR_segmentation.py
        args = Namespace(command=command,
                         dataset=str(list_path),
                         download=False,
                         filter=None,
                         gpu=gpu_number,
                         limit=im_limit,
                         logs=str(logs_path),
                         model=str(model_path),
                         resultFolder=None,
                         year=default_dataset_year)
    else:
        exit('Command not recognized! Use "train" or "evaluate".')
    # Change working directory to drive with CUDA library
    # (to avoid issues with automatically finding the CUDA library)
    os.chdir(root_dir)
    # Run Mask R-CNN segmentation
    run_segmentation(args)

    # If used for evaluation, perform post-processing to reassemble whole segmentation mask
    if command == 'evaluate':
        print('Run Mask R-CNN post-processing!')
        mrcnn_postprocess(files=filenames, px_size_mrcnn=pixelsize_mrcnn, tile_size=tile_size,
                          tile_overlap=tile_overlap, min_feature_size=min_feature_size, parent_folder=parent_dir,
                          result_folder=result_dir_name)

    # Delete temporary tile folders
    if delete_tiles:
        for folder in [tile_dir, result_dir]:
            shutil.rmtree(folder)


if __name__ == '__main__':
    # import cProfile
    # import pstats
    # profiler = cProfile.Profile()
    # profiler.enable()
    main()
    # profiler.disable()
    # stats = pstats.Stats(profiler).sort_stats('cumtime')
    # stats.print_stats()
