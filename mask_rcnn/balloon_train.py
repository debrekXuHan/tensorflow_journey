#!/usr/bin/python
# -*- coding: utf-8 -*-
import sys
import os

ROOT_DIR = sys.path[0]
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, 'logs')
BALLOON_DATA = os.path.join(ROOT_DIR, 'balloon')

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