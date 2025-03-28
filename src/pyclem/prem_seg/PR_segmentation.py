"""
Mask R-CNN
Configurations and data loading code for MS COCO.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 coco.py train --dataset=/path/to/coco/ --model=coco

    # Train a new model starting from ImageNet weights. Also auto download COCO dataset
    python3 coco.py train --dataset=/path/to/coco/ --model=imagenet --download=True

    # Continue training a model that you had trained earlier
    python3 coco.py train --dataset=/path/to/coco/ --model=/path/to/weights.h5

    # Continue training the last model you trained
    python3 coco.py train --dataset=/path/to/coco/ --model=last

    # Run COCO evaluation on the last model you trained
    python3 coco.py evaluate --dataset=/path/to/coco/ --model=last
"""

import os
import sys
import time
import numpy as np
import imgaug  # https://github.com/aleju/imgaug (pip3 install imgaug)
import skimage
from mrcnn import visualize
import fnmatch
from scipy import ndimage
import math
import matplotlib
import matplotlib.pyplot as plt
import cv2

# Download and install the Python COCO tools from https://github.com/waleedka/coco
# That's a fork from the original https://github.com/pdollar/coco with a bug
# fix for Python 3.
# I submitted a pull request https://github.com/cocodataset/cocoapi/pull/50
# If the PR is merged then use the original repo.
# Note: Edit PythonAPI/Makefile and replace "python" with "python3".
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils

import zipfile
import urllib.request
import shutil

# Root directory of the project
ROOT_DIR = os.path.abspath("../Maskrcnn/")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")
DEFAULT_DATASET_YEAR = "2014"


############################################################
#  Configurations
############################################################
class WormConfig(Config):
    """Configuration for training on MS COCO.
    Derives from the base Config class and overrides values specific
    to the COCO dataset.
    """
    # Give the configuration a recognizable name
    NAME = "worm"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1
    DETECTION_MIN_CONFIDENCE = 0.8  # 0.7

    # Number of classes (including background)
    NUM_CLASSES = 3+1  #3 + 1

    STEPS_PER_EPOCH = 300

    # Number of validation steps to run at the end of every training epoch.
    # A bigger number improves accuracy of validation stats, but slows
    # down the training.
    VALIDATION_STEPS = 100  # 50

    BACKBONE = "resnet101"

    # Uncomment to train on 8 GPUs (default is 1)
    GPU_COUNT = 1

    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)

    # Number of classes (including background)
    NUM_CLASSES = 3 + 1  # 3 + 1  # worm has 22 classes
    LEARNING_RATE = 0.02  # 0.001

    LOSS_WEIGHTS = {
        "rpn_class_loss": 1.,
        "rpn_bbox_loss": 1.,
        "mrcnn_class_loss": 10.,
        "mrcnn_bbox_loss": 1.,
        "mrcnn_mask_loss": 1.
    }

    IMAGE_RESIZE_MODE = "square"
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512


############################################################
#  Dataset
############################################################
class WormDataset(utils.Dataset):
    def load_pairs(self, dataset_file, subset=1.0, class_ids=None, class_map=None):
        """Load a data set
        dataset_file: The filename of the paired list (image, mask).
        class_ids: If provided, only loads images that have the given classes.
        class_map: TODO: Not implemented yet. Supports mapping classes from
                     different datasets to the same class ID.
        return_coco: If True, returns the COCO object.
        """

        # Add classes
        # self.add_class("worm", 1, "Cell")
        self.add_class("worm", 1, "Dome")
        self.add_class("worm", 2, "Flat")
        self.add_class("worm", 3, "Sphere")


        with open(dataset_file, 'r') as f:
            medical_images = f.readlines()

        # define the subset to load
        totalData = len(medical_images)
        beginData = 0
        endData = totalData - 1
        if subset > 0:
            endData = int(float(totalData)*subset)-1
            print("load %d images.", endData+1)
        else:
            beginData = int(float(totalData)*(1+subset))
            print("load %d images.", len(medical_images)-beginData)


        # Add images
        for idx, image_name in enumerate(medical_images[beginData:endData]):
            image_name = image_name[:-1]  # remove eno of line
            (src_name, gt_name) = image_name.split(',')
            self.add_image(
                "worm", image_id=idx,
                path=src_name,
                maskFn=gt_name)

    def load_mask(self, image_id):
        """Load instance masks for the given image.

        Different datasets use different ways to store masks. This
        function converts the different mask format to one format
        in the form of a bitmap [height, width, instances].

        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a COCO image, delegate to parent class.
        image_info = self.image_info[image_id]

        instance_masks = []
        class_ids = []

        maskIm = skimage.io.imread(self.image_info[image_id]['maskFn'])
        # maskIm=maskIm/255
        height = maskIm.shape[0]
        width = maskIm.shape[1]

        NUM_CLASSES = 3+1  # 3+1
        # Build mask of shape [height, width, instance_count] and list
        # of class IDs that correspond to each channel of the mask.
        for i in range(1, NUM_CLASSES):
            # m = np.zeros([height, width], dtype=bool)
            m = np.isin(maskIm, [i])
            m = m * 1
            m = np.uint8(m)
            if np.count_nonzero(m) > 0:
                labeled, nr_objects = ndimage.label(m)

                if nr_objects >= 1 and np.count_nonzero(labeled == 1) > 5:
                    instance_masks.append(labeled == 1)
                    class_ids.append(i)

                if nr_objects > 1:
                     for j in range(1, nr_objects + 1):
                         merged_m = (labeled == j);
                         object1x = ndimage.measurements.center_of_mass(labeled == j)[0]
                         object1y = ndimage.measurements.center_of_mass(labeled == j)[1]
                         for k in range(1, nr_objects + 1):
                             #plt.figure()
                             #plt.imshow(m)
                             #plt.show()
                             #plt.close()
                             m = (labeled == k)
                             lbl = ndimage.measurements.center_of_mass(m);
                             dist = math.sqrt((lbl[0] - object1x) ** 2 + (lbl[1] - object1y) ** 2)
                             if dist < 0:
                                 merged_m = merged_m + m

                         instance_masks.append(merged_m)
                         class_ids.append(i)

        # Pack instance masks into an array
        if class_ids:
            mask = np.stack(instance_masks, axis=2)
            class_ids = np.array(class_ids, dtype=np.int32)
            return mask, class_ids
        else:
            # Call super class to return an empty mask
            return super(WormDataset, self).load_file(image_id, )


############################################################
#  COCO Evaluation
############################################################
def build_coco_results(dataset, image_ids, rois, class_ids, scores, masks):
    """
    Arrange results to match COCO specs in http://cocodataset.org/#format
    """
    # If no results, return an empty list
    if rois is None:
        return []

    results = []
    for image_id in image_ids:
        # Loop through detections
        for i in range(rois.shape[0]):
            class_id = class_ids[i]
            score = scores[i]
            bbox = np.around(rois[i], 1)
            mask = masks[:, :, i]

            result = {
                "image_id": image_id,
                "category_id": dataset.get_source_class_id(class_id, "coco"),
                "bbox": [bbox[1], bbox[0], bbox[3] - bbox[1], bbox[2] - bbox[0]],
                "score": score,
                "segmentation": maskUtils.encode(np.asfortranarray(mask))
            }
            results.append(result)
    return results


def evaluate_coco(model, dataset, coco, eval_type="bbox", limit=0, image_ids=None):
    """Runs official COCO evaluation.
    dataset: A Dataset object with valiadtion data
    eval_type: "bbox" or "segm" for bounding box or segmentation evaluation
    limit: if not 0, it's the number of images to use for evaluation
    """
    # Pick COCO images from the dataset
    image_ids = image_ids or dataset.image_ids

    # Limit to a subset
    if limit:
        image_ids = image_ids[:limit]

    # Get corresponding COCO image IDs.
    coco_image_ids = [dataset.image_info[id]["id"] for id in image_ids]

    t_prediction = 0
    t_start = time.time()

    results = []
    for i, image_id in enumerate(image_ids):
        # Load image
        image = dataset.load_image(image_id)

        # Run detection
        t = time.time()
        r = model.detect([image], verbose=0)[0]
        t_prediction += (time.time() - t)

        # Convert results to COCO format
        # Cast masks to uint8 because COCO tools errors out on bool
        image_results = build_coco_results(dataset, coco_image_ids[i:i + 1],
                                           r["rois"], r["class_ids"],
                                           r["scores"],
                                           r["masks"].astype(np.uint8))
        results.extend(image_results)

    # Load results. This modifies results with additional attributes.
    coco_results = coco.loadRes(results)

    # Evaluate
    cocoEval = COCOeval(coco, coco_results, eval_type)
    cocoEval.params.imgIds = coco_image_ids
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    print("Prediction time: {}. Average {}/image".format(
        t_prediction, t_prediction / len(image_ids)))
    print("Total time: ", time.time() - t_start)


############################################################
#  Training
############################################################
def run_segmentation(args):
    print("Command: ", args.command)
    print("Model: ", args.model)
    print("Dataset: ", args.dataset)
    print("Year: ", args.year)
    print("Logs: ", args.logs)
    print("GPU: ", args.gpu)

    matplotlib.get_backend()
    matplotlib.use("TkAgg")

    if args.gpu == '0':
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # Configurations
    if args.command == "train":
        config = WormConfig()
    else:
        class InferenceConfig(WormConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
            DETECTION_MIN_CONFIDENCE = 0
            DETECTION_NMS_THRESHOLD = 0.1
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.model.lower() == "coco":
        model_path = COCO_MODEL_PATH
    elif args.model.lower() == "last":
        # Find last trained weights
        model_path = model.find_last()
    elif args.model.lower() == "imagenet":
        # Start from ImageNet trained weights
        model_path = model.get_imagenet_weights()
    else:
        model_path = args.model

    # Load weights
    print("Loading weights ", model_path)
    if args.model.lower() == "coco":
        model.load_weights(model_path, by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(model_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        # Training dataset. Use the training set and 35K from the
        # validation set, as in the Mask RCNN paper.
        dataset_train = WormDataset()
        dataset_train.load_pairs(args.dataset, subset=0.99)
        dataset_train.prepare()

        # Validation dataset
        dataset_val = WormDataset()
        dataset_val.load_pairs(args.dataset, subset=-0.01)
        dataset_val.prepare()

        augmentation = imgaug.augmenters.OneOf([
            imgaug.augmenters.Fliplr(0.5),
            imgaug.augmenters.Flipud(0.5),
            imgaug.augmenters.OneOf([imgaug.augmenters.Affine(rotate=-10),
                                      imgaug.augmenters.Affine(rotate=10),
                                      imgaug.augmenters.Affine(rotate=15)]),
            imgaug.augmenters.Multiply((0.9, 1.1))
            # imgaug.augmenters.GaussianBlur(sigma=(0.0, 1.5))
            # iaa.GaussianBlur(sigma=(0,3.0))
        ])
        # *** This training schedule is an example. Update to your needs ***

        # Training - Stage 1
        print("Training network heads")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=200,
                    layers='heads',
                    augmentation=augmentation)

        # Training - Stage 2
        # Finetune layers from ResNet stage 4 and up
        print("Fine tune Resnet stage 4 and up")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE/2,
                    epochs=400,
                    layers='4+',
                    augmentation=augmentation)

        # Training - Stage 3
        # Fine tune all layers
        print("Fine tune all layers")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE/2,
                    epochs=600,
                    layers='all',
                    augmentation=augmentation)

    elif args.command == "evaluate":
        # Validation dataset

        color = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
        class_names = ['BG', 'Dome', 'Flat', 'Sphere']
        for root, dirs, files in os.walk(args.dataset):
            for d1 in os.listdir(root):
                inputDir=os.path.join(root,d1)
                for root1, dirs2, files in os.walk(inputDir):
                    for f1 in fnmatch.filter(files, args.filter):
                        inputFn = os.path.join(inputDir, f1)
                        print("Running evaluation on ", inputFn)
                        image = skimage.io.imread(inputFn)
                        # image = np.expand_dims(image, axis=2)
                        # image = np.concatenate((image,image,image), axis=2)
                        results = model.detect([image], verbose=0)
                        r = results[0]

                        resultFn = inputFn.replace(args.dataset, args.resultFolder)


                        index = np.where(r['scores'] >= 0.98)[0]
                        r1 = {}
                        r1['rois'] = r['rois'][index, :]
                        r1['masks'] = r['masks'][:, :, index]
                        r1['class_ids'] = r['class_ids'][index]
                        r1['scores'] = r['scores'][index]

                        index = np.argsort(r1['scores'])[::-1]
                        r2 = {}
                        r2['rois'] = r1['rois'][index, :]
                        r2['masks'] = r1['masks'][:, :, index]
                        r2['class_ids'] = r1['class_ids'][index]
                        r2['scores'] = r1['scores'][index]



                        # if len(r2['scores'])>0:
                        #     visualize.display_instances(image, r2['rois'], r2['masks'], r2['class_ids'], class_names, r2['scores'],"",(16, 16),None,True,False,color)



                        bbx = r2['rois']
                        scores = r2['scores']
                        classid = r2['class_ids']


                        N = r2["rois"].shape[0]
                        masks = r2["masks"]
                        mask_f=0*image[:,:,0]
                        mask_s=0*image[:,:,0]
                        for i in range(N):
                            if scores[i]>=0.01 :
                                mask = masks[:, :, i]
                                test=mask_f*mask
                                if np.count_nonzero(test) == 0:
                                    mask_f=mask_f+classid[i]*mask
                                else:
                                    mask_f=mask_f+mask-test
                                mask_s=mask_s+scores[i]*mask
                                ccolor = color[classid[i]]


                        resultFn = inputFn.replace(args.dataset, args.resultFolder)
                        curr_out_dir = os.path.dirname(resultFn)
                        if not os.path.exists(curr_out_dir):
                            os.makedirs(curr_out_dir)
                        cv2.imwrite(resultFn.replace(".tif", "_mrcnn_seg.jpg"), 50*mask_f)




    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'evaluate'".format(args.command))


if __name__ == '__main__':
    import argparse
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train Mask R-CNN on MS COCO.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'evaluate' on MS COCO")
    parser.add_argument('--dataset', required=True,
                        metavar="/path/to/paired_file",
                        help='paired file')
    parser.add_argument('--resultFolder', required=False,
                        metavar="/path/to/result",
                        help='folder for result')
    parser.add_argument('--filter', required=False,
                        metavar="*.png",
                        help='filter to get files')
    parser.add_argument('--year', required=False,
                        default=DEFAULT_DATASET_YEAR,
                        metavar="<year>",
                        help='Year of the MS-COCO dataset (2014 or 2017) (default=2014)')
    parser.add_argument('--model', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--limit', required=False,
                        default=500,
                        metavar="<image count>",
                        help='Images to use for evaluation (default=500)')
    parser.add_argument('--download', required=False,
                        default=False,
                        metavar="<True|False>",
                        help='Automatically download and unzip MS-COCO files (default=False)',
                        type=bool)
    parser.add_argument('--gpu', required=False,
                        default=1,
                        metavar="number of gpu, 0 means only cpu",
                        help='Number of gpu to use (default=1)')
    # pack arguments into input_args
    input_args = parser.parse_args()

    # Run segmentation program
    run_segmentation(input_args)
