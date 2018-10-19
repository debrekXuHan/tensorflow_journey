#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np

result_type = [0,128,191,255]

def get_img(image_path, mask_path):
    image_path = os.walk(image_path)
    image_path_list = []
    mask_path_list = []
    for path_model, d, filelist in image_path:
        for filename in filelist:
            image_path_list.append(os.path.join(path_model, filename))
            # mask_path_this = filename[0:filename.rindex('_')] + '_labelMark' + filename[filename.rindex('_'):len(filename)]
            mask_path_list.append(os.path.join(mask_path, filename))

    return image_path_list, mask_path_list

def adjustData(img, mask, flag_multi_class, num_class):
    if (flag_multi_class):
        img = img / 255
        # print('mask: ', np.unique(mask))
        # new_mask = np.zeros(mask.shape)
        # for i in range(num_class):
        #     new_mask[mask == result_type[i]] = i
        mask = np.expand_dims(mask, -1)
    elif (np.max(img) > 1):
        img = img / 255
        mask = mask / 255
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0
    return (img, mask)

def trainGenerator(batch_size, steps_per_epoch, train_path, image_folder, mask_folder,
                   img_height, img_width, flag_multi_class, num_class, target_size = (256, 256)):
    image_path_list, mask_path_list = get_img(train_path + image_folder, train_path + mask_folder)
    image_len = len(image_path_list)

    # train_list = []
    # label_list = []
    # batch = 0
    # for i in range(0, steps_per_epoch):
    #     index = i % image_len
    #     batch += 1
    #     img = cv2.imread(image_path_list[index], 0)
    #     mask = cv2.imread(mask_path_list[index], 0)
    #     print('mask path',mask_path_list[index])
    #     print('image path',image_path_list[index])
    #     img = cv2.resize(img, target_size, interpolation = cv2.INTER_NEAREST)
    #     mask = cv2.resize(mask, target_size, interpolation = cv2.INTER_NEAREST)

    #     img = np.reshape(img, target_size + (1,))
    #     img, mask = adjustData(img, mask, flag_multi_class, num_class)
    #     train_list.append(img)
    #     label_list.append(mask)
    #     if (0 == batch % batch_size) | (steps_per_epoch == batch):
    #         img_arr = np.array(train_list)
    #         mask_arr = np.array(label_list)
    #         print(img_arr.shape, mask_arr.shape)
    #         train_list = []
    #         label_list = []


    while True:
        train_list = []
        label_list = []
        batch = 0
        for i in range(0, steps_per_epoch):
            try:
                index = i % image_len
                batch += 1
                img = cv2.imread(image_path_list[index], 0)
                mask = cv2.imread(mask_path_list[index], 0)
                print('mask path',mask_path_list[index])
                print('image path',image_path_list[index])
                img = cv2.resize(img, target_size, interpolation = cv2.INTER_NEAREST)
                mask = cv2.resize(mask, target_size, interpolation = cv2.INTER_NEAREST)

                img = np.reshape(img, target_size + (1,))
                img, mask = adjustData(img, mask, flag_multi_class, num_class)
                train_list.append(img)
                label_list.append(mask)
                if (0 == batch % batch_size) | (steps_per_epoch == batch):
                    img_arr = np.array(train_list)
                    mask_arr = np.array(label_list)
                    yield(img_arr, mask_arr)
                    train_list = []
                    label_list = []
            except:
                print('img read error')
                pass

def testGenerator(test_path, num_image = 128, target_size = (256, 256)):
    while True:
        for i in range(1, 1 + num_image):
            try:
                img = cv2.imread(os.path.join(test_path, "%d.bmp" %i), 0)
                print('img shape',img.shape)
                img = cv2.resize(img, target_size, interpolation = cv2.INTER_NEAREST)

                img = img / 255.
                img = np.reshape(img, target_size + (1,))
                img = np.reshape(img, (1, ) + img.shape)
                print(img.shape)
                yield img
            except:
                pass

def labelVisualize(img, num_class):
    img_out = np.zeros(img.shape)
    for i in range(num_class):
        img_out[i == img] = result_type[i]
    return img_out

def saveResult(save_path, npyfile, flag_multi_class = True, num_class = 4):
    for i, img in enumerate(npyfile):
        img = np.argmax(img, axis = -1)
        img = labelVisualize(img, num_class)
        print(np.unique(img))
        cv2.imwrite(os.path.join(save_path, '%d_predict.bmp' %i), img)
