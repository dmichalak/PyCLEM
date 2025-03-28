
# coding: utf-8

# # Mask R-CNN - Inspect Training Data
# 
# Inspect and visualize data loading and pre-processing code.

# In[1]:


import os
import matplotlib


import matplotlib.pyplot as plt
import cv2

from mrcnn import utils
from mrcnn import visualize
#from visualize import display_images
#from model import log

# get_ipython().magic('matplotlib inline')

ROOT_DIR = os.getcwd()


# ## Configurations
# 
# Run one of the code blocks below to import and load the configurations to use.

# In[2]:


# Run one of the code blocks

# Shapes toy dataset
# import shapes
# config = shapes.ShapesConfig()

# MS COCO Dataset
# import coco
# config = coco.CocoConfig()
# COCO_DIR = "C:\\Users\\liujiamin\\Softwares\\Mask_RCNN-master\\dataset\\coco2014"  # TODO: enter value here


from prem_segmentation.src import worm

matplotlib.get_backend()
matplotlib.use("TkAgg")

config = worm.WormConfig()
# plaque_DIR = "C:\\Users\\liujiamin\\Softwares\\Mask_RCNN-master\\dataset\\plaque2014"  # TODO: enter value here
# dataset_file="C:\\Users\\liujiamin\\Softwares\\Mask_RCNN-master\\dataset\\plaque2014\\plaque_pairlist_11classes_point5_new.txt"
dataset_file="F:\\projects\\MG\\list_stephen.txt"

# ## Dataset

# In[3]:


# Load dataset
if config.NAME == 'worm':
    dataset = worm.WormDataset()
    dataset.load_pairs(dataset_file, subset=0.66)

# Must call before using the dataset
dataset.prepare()

print("Image Count: {}".format(len(dataset.image_ids)))
print("Class Count: {}".format(dataset.num_classes))
for i, info in enumerate(dataset.class_info):
    print("{:3}. {:50}".format(i, info['name']))


# ## Display Samples
# 
# Load and display images and masks.

# In[4]:


# # Load and display random samples
# # image_ids = np.random.choice(dataset.image_ids, 4)
# # for image_id in image_ids:
# for image_id in range(0, len(dataset.image_ids)):
#     image = dataset.load_image(image_id)
#     mask, class_ids = dataset.load_mask(image_id)
#     visualize.display_top_masks(image, mask, class_ids, dataset.class_names)
#     plt.close()


# ## Bounding Boxes
# 
# Rather than using bounding box coordinates provided by the source datasets, we compute the bounding boxes from masks instead. This allows us to handle bounding boxes consistently regardless of the source dataset, and it also makes it easier to resize, rotate, or crop images because we simply generate the bounding boxes from the updates masks rather than computing bounding box transformation for each type of image transformation.

# In[5]:


# Load random image and mask.
# image_id = random.choice(dataset.image_ids)
num_class1=0
num_class2=0
num_class3=0
num_class4=0
num_class5=0
num_class6=0
num_class7=0
num_class8=0
num_class9=0
num_class10=0
num_class11=0
num_class12=0
num_class13=0
num_class14=0
num_class15=0
num_class16=0
num_class17=0
num_class18=0
num_class19=0
num_class20=0
num_class21=0
num_class22=0


for image_id in range(0, len(dataset.image_ids),1):
#for image_id in range(0, 307, 1):
    print("image_id ", dataset.image_info[image_id]['path'])
    image = dataset.load_image(image_id)
    mask, class_ids = dataset.load_file(image_id, )

    if class_ids.any():
        num_class1=num_class1+sum(class_ids==1)
        num_class2 = num_class2 + sum(class_ids == 2)
        num_class3 = num_class3 + sum(class_ids == 3)
        num_class4 = num_class4 + sum(class_ids == 4)
        num_class5 = num_class5 + sum(class_ids == 5)
        num_class6 = num_class6 + sum(class_ids == 6)
        num_class7 = num_class7 + sum(class_ids == 7)
        num_class8 = num_class8 + sum(class_ids == 8)
        num_class9 = num_class9 + sum(class_ids == 9)
        num_class10 = num_class10 + sum(class_ids == 10)
        num_class11 = num_class11 + sum(class_ids == 11)
        num_class12 = num_class12 + sum(class_ids == 12)
        num_class13 = num_class13 + sum(class_ids == 13)
        num_class14 = num_class14 + sum(class_ids == 14)
        num_class15 = num_class15 + sum(class_ids == 15)
        num_class16 = num_class16 + sum(class_ids == 16)
        num_class17 = num_class17 + sum(class_ids == 17)
        num_class18 = num_class18 + sum(class_ids == 18)
        num_class19 = num_class19 + sum(class_ids == 19)
        num_class20 = num_class20 + sum(class_ids == 20)
        num_class21 = num_class21 + sum(class_ids == 21)
        num_class22 = num_class22 + sum(class_ids == 22)
    else:
        temp=1


    # Compute Bounding box
    bbox = utils.extract_bboxes(mask)
    #


    # Display image and additional stats
    # print("image_id ", image_id, dataset.image_reference(image_id))
    # log("image", image)
    # log("mask", mask)
    # log("class_ids", class_ids)
    # log("bbox", bbox)

    if len(bbox)>0:
    #     # if max(class_ids)==1:
    #         # Display image and instances
        img_scaled = cv2.normalize(image, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        visualize.display_instances(img_scaled, bbox, mask, class_ids, dataset.class_names)
        plt.close()



# ## Resize Images
# 6
# To support multiple images per batch, images are resized to one size (1024x1024). Aspect ratio is preserved, though. If an image is not square, then zero padding is added at the top/bottom or right/left.

# In[6]:


# # Load random image and mask.
# image_id = np.random.choice(dataset.image_ids, 1)[0]
# image = dataset.load_image(image_id)
# mask, class_ids = dataset.load_mask(image_id)
# original_shape = image.shape
# # Resize
# image, window, scale, padding = utils.resize_image(
#     image,
#     min_dim=config.IMAGE_MIN_DIM,
#     max_dim=config.IMAGE_MAX_DIM,
#     padding=config.IMAGE_PADDING)
# mask = utils.resize_mask(mask, scale, padding)
# # Compute Bounding box
# bbox = utils.extract_bboxes(mask)
#
# # Display image and additional stats
# print("image_id: ", image_id, dataset.image_reference(image_id))
# print("Original shape: ", original_shape)
# log("image", image)
# log("mask", mask)
# log("class_ids", class_ids)
# log("bbox", bbox)
# # Display image and instances
# visualize.display_instances(image, bbox, mask, class_ids, dataset.class_names)
#
#
# # ## Mini Masks
# #
# # Instance binary masks can get large when training with high resolution images. For example, if training with 1024x1024 image then the mask of a single instance requires 1MB of memory (Numpy uses bytes for boolean values). If an image has 100 instances then that's 100MB for the masks alone.
# #
# # To improve training speed, we optimize masks by:
# # * We store mask pixels that are inside the object bounding box, rather than a mask of the full image. Most objects are small compared to the image size, so we save space by not storing a lot of zeros around the object.
# # * We resize the mask to a smaller size (e.g. 56x56). For objects that are larger than the selected size we lose a bit of accuracy. But most object annotations are not very accuracy to begin with, so this loss is negligable for most practical purposes. Thie size of the mini_mask can be set in the config class.
# #
# # To visualize the effect of mask resizing, and to verify the code correctness, we visualize some examples.
#
# # In[7]:
#
#
# image_id = np.random.choice(dataset.image_ids, 1)[0]
# image, image_meta, class_ids, bbox, mask = modellib.load_image_gt(
#     dataset, config, image_id, use_mini_mask=False)
#
# log("image", image)
# log("image_meta", image_meta)
# log("class_ids", class_ids)
# log("bbox", bbox)
# log("mask", mask)
#
# display_images([image]+[mask[:,:,i] for i in range(min(mask.shape[-1], 7))])
#
#
# # In[8]:
#
#
# visualize.display_instances(image, bbox, mask, class_ids, dataset.class_names)
#
#
# # In[9]:
#
#
# # Add augmentation and mask resizing.
# image, image_meta, class_ids, bbox, mask = modellib.load_image_gt(
#     dataset, config, image_id, augment=True, use_mini_mask=True)
# log("mask", mask)
# display_images([image]+[mask[:,:,i] for i in range(min(mask.shape[-1], 7))])
#
#
# # In[10]:
#
#
# mask = utils.expand_mask(bbox, mask, image.shape)
# visualize.display_instances(image, bbox, mask, class_ids, dataset.class_names)
#
#
# # ## Anchors
# #
# # The order of anchors is important. Use the same order in training and prediction phases. And it must match the order of the convolution execution.
# #
# # For an FPN network, the anchors must be ordered in a way that makes it easy to match anchors to the output of the convolution layers that predict anchor scores and shifts.
# # * Sort by pyramid level first. All anchors of the first level, then all of the second and so on. This makes it easier to separate anchors by level.
# # * Within each level, sort anchors by feature map processing sequence. Typically, a convolution layer processes a feature map starting from top-left and moving right row by row.
# # * For each feature map cell, pick any sorting order for the anchors of different ratios. Here we match the order of ratios passed to the function.
# #
# # **Anchor Stride:**
# # In the FPN architecture, feature maps at the first few layers are high resolution. For example, if the input image is 1024x1024 then the feature meap of the first layer is 256x256, which generates about 200K anchors (256*256*3). These anchors are 32x32 pixels and their stride relative to image pixels is 4 pixels, so there is a lot of overlap. We can reduce the load significantly if we generate anchors for every other cell in the feature map. A stride of 2 will cut the number of anchors by 4, for example.
# #
# # In this implementation we use an anchor stride of 2, which is different from the paper.
#
# # In[11]:
#
#
# # Generate Anchors
# anchors = utils.generate_pyramid_anchors(config.RPN_ANCHOR_SCALES,
#                                           config.RPN_ANCHOR_RATIOS,
#                                           config.BACKBONE_SHAPES,
#                                           config.BACKBONE_STRIDES,
#                                           config.RPN_ANCHOR_STRIDE)
#
# # Print summary of anchors
# num_levels = len(config.BACKBONE_SHAPES)
# anchors_per_cell = len(config.RPN_ANCHOR_RATIOS)
# print("Count: ", anchors.shape[0])
# print("Scales: ", config.RPN_ANCHOR_SCALES)
# print("ratios: ", config.RPN_ANCHOR_RATIOS)
# print("Anchors per Cell: ", anchors_per_cell)
# print("Levels: ", num_levels)
# anchors_per_level = []
# for l in range(num_levels):
#     num_cells = config.BACKBONE_SHAPES[l][0] * config.BACKBONE_SHAPES[l][1]
#     anchors_per_level.append(anchors_per_cell * num_cells // config.RPN_ANCHOR_STRIDE**2)
#     print("Anchors in Level {}: {}".format(l, anchors_per_level[l]))
#
#
# # Visualize anchors of one cell at the center of the feature map of a specific level.
#
# # In[12]:
#
#
# ## Visualize anchors of one cell at the center of the feature map of a specific level
#
# # Load and draw random image
# image_id = np.random.choice(dataset.image_ids, 1)[0]
# image, image_meta, _, _, _ = modellib.load_image_gt(dataset, config, image_id)
# fig, ax = plt.subplots(1, figsize=(10, 10))
# ax.imshow(image)
# levels = len(config.BACKBONE_SHAPES)
#
# for level in range(levels):
#     colors = visualize.random_colors(levels)
#     # Compute the index of the anchors at the center of the image
#     level_start = sum(anchors_per_level[:level]) # sum of anchors of previous levels
#     level_anchors = anchors[level_start:level_start+anchors_per_level[level]]
#     print("Level {}. Anchors: {:6}  Feature map Shape: {}".format(level, level_anchors.shape[0],
#                                                                 config.BACKBONE_SHAPES[level]))
#     center_cell = config.BACKBONE_SHAPES[level] // 2
#     center_cell_index = (center_cell[0] * config.BACKBONE_SHAPES[level][1] + center_cell[1])
#     level_center = center_cell_index * anchors_per_cell
#     center_anchor = anchors_per_cell * (
#         (center_cell[0] * config.BACKBONE_SHAPES[level][1] / config.RPN_ANCHOR_STRIDE**2) \
#         + center_cell[1] / config.RPN_ANCHOR_STRIDE)
#     level_center = int(center_anchor)
#
#     # Draw anchors. Brightness show the order in the array, dark to bright.
#     for i, rect in enumerate(level_anchors[level_center:level_center+anchors_per_cell]):
#         y1, x1, y2, x2 = rect
#         p = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, facecolor='none',
#                               edgecolor=(i+1)*np.array(colors[level]) / anchors_per_cell)
#         ax.add_patch(p)
#
#
# # ## Data Generator
# #
#
# # In[13]:
#
#
# # Create data generator
# random_rois = 2000
# g = modellib.data_generator(
#     dataset, config, shuffle=True, random_rois=random_rois,
#     batch_size=4,
#     detection_targets=True)
#
#
# # In[14]:
#
#
# # Uncomment to run the generator through a lot of images
# # to catch rare errors
# # for i in range(1000):
# #     print(i)
# #     _, _ = next(g)
#
#
# # In[15]:
#
#
# # Get Next Image
# if random_rois:
#     [normalized_images, image_meta, rpn_match, rpn_bbox, gt_class_ids, gt_boxes, gt_masks, rpn_rois, rois],     [mrcnn_class_ids, mrcnn_bbox, mrcnn_mask] = next(g)
#
#     log("rois", rois)
#     log("mrcnn_class_ids", mrcnn_class_ids)
#     log("mrcnn_bbox", mrcnn_bbox)
#     log("mrcnn_mask", mrcnn_mask)
# else:
#     [normalized_images, image_meta, rpn_match, rpn_bbox, gt_boxes, gt_masks], _ = next(g)
#
# log("gt_class_ids", gt_class_ids)
# log("gt_boxes", gt_boxes)
# log("gt_masks", gt_masks)
# log("rpn_match", rpn_match, )
# log("rpn_bbox", rpn_bbox)
# image_id = image_meta[0][0]
# print("image_id: ", image_id, dataset.image_reference(image_id))
#
# # Remove the last dim in mrcnn_class_ids. It's only added
# # to satisfy Keras restriction on target shape.
# mrcnn_class_ids = mrcnn_class_ids[:,:,0]
#
#
# # In[16]:
#
#
# b = 0
#
# # Restore original image (reverse normalization)
# sample_image = modellib.unmold_image(normalized_images[b], config)
#
# # Compute anchor shifts.
# indices = np.where(rpn_match[b] == 1)[0]
# refined_anchors = utils.apply_box_deltas(anchors[indices], rpn_bbox[b, :len(indices)] * config.RPN_BBOX_STD_DEV)
# log("anchors", anchors)
# log("refined_anchors", refined_anchors)
#
# # Get list of positive anchors
# positive_anchor_ids = np.where(rpn_match[b] == 1)[0]
# print("Positive anchors: {}".format(len(positive_anchor_ids)))
# negative_anchor_ids = np.where(rpn_match[b] == -1)[0]
# print("Negative anchors: {}".format(len(negative_anchor_ids)))
# neutral_anchor_ids = np.where(rpn_match[b] == 0)[0]
# print("Neutral anchors: {}".format(len(neutral_anchor_ids)))
#
# # ROI breakdown by class
# for c, n in zip(dataset.class_names, np.bincount(mrcnn_class_ids[b].flatten())):
#     if n:
#         print("{:23}: {}".format(c[:20], n))
#
# # Show positive anchors
# visualize.draw_boxes(sample_image, boxes=anchors[positive_anchor_ids],
#                      refined_boxes=refined_anchors)
#
#
# # In[17]:
#
#
# # Show negative anchors
# visualize.draw_boxes(sample_image, boxes=anchors[negative_anchor_ids])
#
#
# # In[18]:
#
#
# # Show neutral anchors. They don't contribute to training.
# visualize.draw_boxes(sample_image, boxes=anchors[np.random.choice(neutral_anchor_ids, 100)])
#
#
# # ## ROIs
#
# # In[19]:
#
#
# if random_rois:
#     # Class aware bboxes
#     bbox_specific = mrcnn_bbox[b, np.arange(mrcnn_bbox.shape[1]), mrcnn_class_ids[b], :]
#
#     # Refined ROIs
#     refined_rois = utils.apply_box_deltas(rois[b].astype(np.float32), bbox_specific[:,:4] * config.BBOX_STD_DEV)
#
#     # Class aware masks
#     mask_specific = mrcnn_mask[b, np.arange(mrcnn_mask.shape[1]), :, :, mrcnn_class_ids[b]]
#
#     visualize.draw_rois(sample_image, rois[b], refined_rois, mask_specific, mrcnn_class_ids[b], dataset.class_names)
#
#     # Any repeated ROIs?
#     rows = np.ascontiguousarray(rois[b]).view(np.dtype((np.void, rois.dtype.itemsize * rois.shape[-1])))
#     _, idx = np.unique(rows, return_index=True)
#     print("Unique ROIs: {} out of {}".format(len(idx), rois.shape[1]))
#
#
# # In[20]:
#
#
# if random_rois:
#     # Dispalay ROIs and corresponding masks and bounding boxes
#     ids = random.sample(range(rois.shape[1]), 8)
#
#     images = []
#     titles = []
#     for i in ids:
#         image = visualize.draw_box(sample_image.copy(), rois[b,i,:4].astype(np.int32), [255, 0, 0])
#         image = visualize.draw_box(image, refined_rois[i].astype(np.int64), [0, 255, 0])
#         images.append(image)
#         titles.append("ROI {}".format(i))
#         images.append(mask_specific[i] * 255)
#         titles.append(dataset.class_names[mrcnn_class_ids[b,i]][:20])
#
#     display_images(images, titles, cols=4, cmap="Blues", interpolation="none")
#
#
# # In[21]:
#
#
# # Check ratio of positive ROIs in a set of images.
# if random_rois:
#     limit = 10
#     temp_g = modellib.data_generator(
#         dataset, config, shuffle=True, random_rois=10000,
#         batch_size=1, detection_targets=True)
#     total = 0
#     for i in range(limit):
#         _, [ids, _, _] = next(temp_g)
#         positive_rois = np.sum(ids[0] > 0)
#         total += positive_rois
#         print("{:5} {:5.2f}".format(positive_rois, positive_rois/ids.shape[1]))
#     print("Average percent: {:.2f}".format(total/(limit*ids.shape[1])))
#
