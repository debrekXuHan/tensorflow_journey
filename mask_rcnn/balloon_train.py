#!/usr/bin/python
# -*- coding: utf-8 -*-
import sys
import os
import json

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


def train(model):
    # training dataset
    dataset_train = BalloonDataset()
    dataset_train.load_balloon(args.dataset, "train")

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
