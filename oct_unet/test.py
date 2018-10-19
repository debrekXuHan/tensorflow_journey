#!/usr/bin/env python
# -*- coding: utf-8 -*-
from model import *
from data import *
import keras

#--loading model--#
model = unet()
# model.summary()
model.load_weights('unet_oct.hdf5')
# model.load_weights('unet_membrane.hdf5')

#--testing process--#
testGene = testGenerator('/data_ext/debrek/unet/training_data/image/')
results = model.predict_generator(testGene, 128, verbose = 1)
print('results shape: ', results.shape)
saveResult('/data_ext/debrek/unet/result/', results)
