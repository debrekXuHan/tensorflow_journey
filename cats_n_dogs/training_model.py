#!/usr/bin/python
#-*- coding: utf-8 -*-
import h5py
import numpy as np
from sklearn.utils import shuffle
from keras.models import *
from keras.layers import *
from keras.preprocessing.image import *
import pandas as pd

np.random.seed(2017)

x_train = []
x_test = []

for filename in ['para_ResNet50.h5', 'para_InceptionV3.h5', 'para_Xception.h5']:
    with h5py.File(filename, 'r') as h:
        x_train.append(np.array(h['train']))
        x_test.append(np.array(h['test']))
        y_train = np.array(h['label'])

x_train = np.concatenate(x_train, axis=1)
x_test = np.concatenate(x_test, axis=1)
print("shape of training data: ", x_train.shape)

# shuffling data from dataset before training
x_train, y_train = shuffle(x_train, y_train)

input_tensor = Input(x_train.shape[1:])
x = Dropout(0.5)(input_tensor)
x = Dense(1, activation='sigmoid')(x)

model = Model(input_tensor, x)
model.compile(optimizer='adadelta',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=128,
          epochs=8,
          validation_split=0.2)

model.save('dogcat_model.h5')

y_pred = model.predict(x_test, verbose=1)
y_pred = y_pred.clip(min=0.005, max=0.995)

df = pd.read_csv('dataset/sample_submission.csv')
gen = ImageDataGenerator()
test_generator = gen.flow_from_directory('dataset/test', (224, 224), shuffle=False, batch_size=16, class_mode=None)

for i, fname in enumerate(test_generator.filenames):
    # pay attention to the order of generator sequence here
    index = int(fname[fname.rfind('/')+1:fname.rfind('.')])
    df.at[index-1, 'label'] = y_pred[i]

df.to_csv('dataset/pred.csv', index=None)
df.head(10)
