#!/usr/bin/python
#-*- coding: utf-8 -*-
from keras.models import *
from keras.layers import *
from keras.applications import *
from keras.preprocessing.image import *
import h5py

def write_para(MODEL, model_name, image_size, lambda_func=None):
    width = image_size[0]
    height = image_size[1]
    input_tensor = Input((height, width, 3))
    x = input_tensor
    if lambda_func:
        x = Lambda(lambda_func)(x)

    base_model = MODEL(input_tensor=x, weights='imagenet', include_top=False)
    model = Model(base_model.input, GlobalAveragePooling2D()(base_model.output))

    gen = ImageDataGenerator()
    train_generator = gen.flow_from_directory('dataset/train', image_size, shuffle=False, batch_size=16)
    test_generator = gen.flow_from_directory('dataset/test', image_size, shuffle=False, batch_size=16, class_mode=None)

    train = model.predict_generator(train_generator, len(train_generator))
    test = model.predict_generator(test_generator, len(test_generator))

    with h5py.File("para_%s.h5" %model_name) as h:
        h.create_dataset("train", data=train)
        h.create_dataset("test", data=test)
        h.create_dataset("label", data=train_generator.classes)

write_para(ResNet50, 'ResNet50', (224, 224))
write_para(InceptionV3, 'InceptionV3', (299, 299), inception_v3.preprocess_input)
write_para(Xception, 'Xception', (299, 299), xception.preprocess_input)