#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np

path = '.\\origin'
save = '.\\training_data'

num_class = 4
result_type = [0,128,191,255]

# print(path)
# print(save)
folders = os.walk(path)
for path_model, dirs, filelist in folders:
    for filename in filelist:
        full_dirs = os.path.join(path_model, filename)
        img1 = cv2.imread(full_dirs, 0)
        img = np.hstack((img1,img1))

        if ('label' in path_model):
            new_img = np.zeros(img.shape)
            for i in range(num_class):
                new_img[img == result_type[i]] = i
            img = new_img

        save_dirs = full_dirs.replace(path, save)
        print(save_dirs, img.shape)
        cv2.imwrite(save_dirs, img)
