#!/usr/bin/env python
# -*- coding: utf-8 -*-

from data import *
from model import *
import tensorflow as tf

## Training parameters
BS = 2
STEP_PER_EPOCH = 2000
EPOCH = 1

myGene = trainGenerator(BS, STEP_PER_EPOCH, '/data_ext/debrek/unet/training_data/',
                        'image', 'label', 1024, 1024, True, 4)

## Model definition
model = unet()

# Training process
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = 0.4)
sess = tf.Session(config = tf.ConfigProto(gpu_options = gpu_options))

model_checkpoint = ModelCheckpoint('unet_oct.hdf5', monitor = 'loss', verbose = 1, save_best_only = True)
model.fit_generator(myGene, steps_per_epoch = STEP_PER_EPOCH, epochs = EPOCH,
                    callbacks = [model_checkpoint])
