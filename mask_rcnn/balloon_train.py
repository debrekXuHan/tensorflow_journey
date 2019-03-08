#!/usr/bin/python
# -*- coding: utf-8 -*-
import sys
import os
import json
import numpy as np
import datetime
import skimage

from mrcnn.config import Config
from mrcnn import model as modellib, utils

ROOT_DIR = sys.path[0]
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, 'logs')
BALLOON_DATA = os.path.join(ROOT_DIR, 'balloon')
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

class BalloonConfig(Config):
    # give the configuration a recognized name
    NAME = "balloon"

    # use GTX1080Ti, which can fit two images
    IMAGES_PER_GPU = 2

    # number of classes (including background)
    NUM_CLASSES = 1 + 1

    # number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # skip detection with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9

##################################
#  Dataset Class
##################################
class BalloonDataset(utils.Dataset):
    def load_balloon(self, dataset_dir, subset):
        # add class (we have only one class to add)
        self.add_class("balloon", 1, "balloon")

        # train or validation dataset
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)

        # load annotations
        annotations = json.load(open(os.path.join(dataset_dir, "via_region_data.json")))
        annotations = list(annotations.values())  # don't need the keys
        annotations = [a for a in annotations if a['regions']]  # eliminate un-annotated images

        # add images
        for a in annotations:
            if type(a['regions']) is dict:
                polygons = [r['shape_attributes'] for r in a['regions'].values()]
            else:
                polygons = [r['shape_attributes'] for r in a['regions']]

            image_path = os.path.join(dataset_dir, a['filename'])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            self.add_image(
                source="balloon",
                image_id=a['filename'],
                path=image_path,
                width=width, height=height,
                polygons=polygons
            )

    def load_mask(self, image_id):
        """Generate instance masks for an image.
        Returns:
        masks: a bool array of shape [height, width, instance count] with
               one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        image_info = self.image_info[image_id]

        # if not a balloon dataset image, delegate it
        if image_info['source'] != "balloon":
            return super(self.__class__, self).load_mask(image_id)

        # convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        mask = np.zeros([image_info['height'], image_info['width'], len(image_info['polygons'])],
                        dtype=np.uint8)
        for i, p in enumerate(image_info['polygons']):
            # get position of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = 1

        # Return mask, and array of class IDs of each instance.
        # Since we have one class ID only, we return an array of 1s.
        return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        """ return the path of the image """
        image_info = self.image_info[image_id]
        if image_info['source'] == "balloon":
            return image_info['path']
        else:
            return super(self.__class__, self).image_reference(image_id)


def train(model):
    # training dataset
    dataset_train = BalloonDataset()
    dataset_train.load_balloon(args.dataset, "train")
    dataset_train.prepare()

    # validation dataset
    dataset_val = BalloonDataset()
    dataset_val.load_balloon(args.dataset, "val")
    dataset_val.prepare()

    # no need to train all layers, just the train the heads
    print("Training network heads...")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=30,
                layers='heads')

def color_splash(image, mask):
    """ color splash effect
    [input]
    image: RGB image on [height, width, 3]
    mask: instance segmentation mask [height, width, instance_cnt]

    [return]
    result image
    """
    # make a grayscale copy of the image
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255

    # copy color pixels from original color image at mask area
    if mask.shape[-1] > 0:
        # collapse the mask into one layer
        mask = (np.sum(mask, -1, keepdims=True) >= 1)
        splash = np.where(mask, image, gray).astype(np.uint8)
    else:
        splash = gray.astype(np.uint8)
    return splash

def detect_and_color_splash(model, image_path=None, video_path=None):
    assert image_path or video_path

    # image or video?
    if image_path:
        print("Running on '{}'".format(image_path))
        # read image
        image = skimage.io.imread(image_path)
        # detect balloons
        result = model.detect([image], verbose=1)[0]
        # color splash
        splash = color_splash(image, result['masks'])
        # save output
        file_name = "splash_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
        print("Generating ", file_name)
        skimage.io.imsave(file_name, splash)
    elif video_path:
        print("Running on '{}'".format(video_path))

if __name__ == "__main__":
    import argparse

    # parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect balloons.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'splash'")
    parser.add_argument("--dataset", required=False,
                        default=BALLOON_DATA,
                        metavar="/path/to/balloon/dataset",
                        help="directory of the Balloon dataset")
    parser.add_argument("--weights", required=False,
                        default="coco",
                        metavar="/path/to/weights.h5",
                        help="path to weights .h5 file or 'coco'")
    parser.add_argument("--logs", required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument("--image", required=False,
                        metavar="path or URL to image",
                        help="Image to apply the color splash effect on")
    parser.add_argument("--video", required=False,
                        metavar="path or URL to video",
                        help="Video to apply the color splash effect on")
    args = parser.parse_args()

    # validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "splash":
        assert args.image or args.video,\
               "Provide --image or --video to apply color splash"
    else:
        raise ValueError("Unsupported command")

    print("weights: ", args.weights)
    print("dataset: ", args.dataset)
    print("logs: ", args.logs)

    if args.command == "train":
        config = BalloonConfig()
    else:
        class InferenceConfig(BalloonConfig):
            # set batch size to 1 since we'll be running inference on
            # one image at a time. batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()
    config.display()

    # create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_MODEL_PATH
        # download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # exclude the last layers because they require a matching number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # train or evaluate
    if args.command == "train":
        train(model)
    elif args.command == "splash":
        detect_and_color_splash(model,
                                image_path=args.image,
                                video_path=args.video)
    else:
        print("'{}' is not supported. "
              "Please use 'train' or 'splash'.".format(args.command))
