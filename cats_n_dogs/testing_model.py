#!/usr/bin/python
#-*- coding: utf-8 -*-
from keras.models import *
from keras.layers import *
from keras.preprocessing.image import *
from keras.applications import *

import cv2
import numpy as np

def calc_para(MODEL, img, img_size, lambda_func=None):
    width = img_size[0]
    height = img_size[1]
    input_tensor = Input((height, width, 3))
    x = input_tensor

    if lambda_func:
        x = Lambda(lambda_func)(x)

    base_model = MODEL(input_tensor=x, weights='imagenet', include_top=False)
    model = Model(base_model.input, GlobalAveragePooling2D()(base_model.output))

    img = cv2.resize(img, img_size)
    img = np.expand_dims(img, axis=0)
    para = model.predict(img)
    return para

def pre_trained_para(img):
    data_input = []

    para = calc_para(ResNet50, img, (224, 224))
    data_input.append(para)

    para = calc_para(InceptionV3, img, (299, 299), inception_v3.preprocess_input)
    data_input.append(para)

    para = calc_para(Xception, img, (299, 299), xception.preprocess_input)
    data_input.append(para)

    data_input = np.concatenate(data_input, axis=1)
    return data_input


if __name__=="__main__":
    img = cv2.imread('test.jpg')
    test_input = pre_trained_para(img)

    model = load_model('dogcat_model.h5')
    test_pred = model.predict(test_input)[0, 0]

    classes = ''
    if test_pred > 0.7:
        classes = 'dog'
        print("It is a dog!")
    elif test_pred < 0.3:
        classes = 'cat'
        print("It is a cat!")
    else:
        classes = 'not sure'
        print("I am not sure about it.")

    cv2.putText(img, classes, (10, 20), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 3)
    cv2.imwrite('pred.jpg', img)