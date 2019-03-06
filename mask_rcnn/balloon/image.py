#!/usr/bin/python
# -*- coding: utf-8 -*-
import sys
import os
import json
import random
import cv2
import numpy as np

ROOT_DIR = sys.path[0]
BALLOON_DATA = os.path.join(ROOT_DIR, 'train')

annotations = json.load(open(os.path.join(BALLOON_DATA, "via_region_data.json")))
annotations = list(annotations.values())
annotation = random.sample(annotations, 1)
info_dict= annotation[0]

imageFilename = os.path.join(BALLOON_DATA, info_dict['filename'])
img = cv2.imread(imageFilename)

contours = []
for i in info_dict['regions'].keys():
    all_x = info_dict['regions'][i]['shape_attributes']['all_points_x']
    all_y = info_dict['regions'][i]['shape_attributes']['all_points_y']
    all_x = np.reshape(np.array(all_x), (-1, 1))
    all_y = np.reshape(np.array(all_y), (-1, 1))
    x_y = np.reshape(np.hstack((all_x, all_y)), (-1, 1, 2))
    contours.append(x_y)

cv2.drawContours(img, contours, -1, (0, 0, 255), 3)
cv2.imshow("image with contour", img)
cv2.waitKey(0)
