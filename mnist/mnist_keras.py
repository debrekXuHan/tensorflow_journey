#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
np.random.seed(1337)  # for reproducibility
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten
from keras.optimizers import Adam

import input_data

# Load the MNIST Dataset
mnist = input_data.read_data_sets("MNIST_data/", one_hot = True, val_size=0)
x_train = mnist.train._images.reshape(-1, 1, 28, 28)
x_test  = mnist.test._images.reshape(-1, 1, 28, 28)
y_train = mnist.train._labels
y_test  = mnist.test._labels

# Build the CNN model using Keras
model = Sequential()

# Conv layer 1 output shape (32, 28, 28)
model.add(Convolution2D(
    batch_input_shape=(None, 1, 28, 28),
    filters=32,
    kernel_size=5,
    strides=1,
    padding="same",
    data_format="channels_first",
    name="conv1"
))
model.add(Activation("relu"))

# Pooling layer 1 (max pooling) output shape (32, 14, 14)
model.add(MaxPooling2D(
    pool_size=2,
    strides=2,
    padding="same",
    data_format="channels_first"
))

# Conv layer 2 output shape (64, 14, 14)
model.add(Convolution2D(64, 5, strides=1, padding="same", data_format="channels_first"))
model.add(Activation("relu"))

# Pooling layer 2 (max pooling) output shape (64, 7, 7)
model.add(MaxPooling2D(2, 2, "same", data_format="channels_first"))

# Fully Connected layer 1 input shape (64 * 7 * 7) = (3136), output shape (1024)
model.add(Flatten())
model.add(Dense(1024))
model.add(Activation("relu"))

# Fully Connected layer to shape (10) for 10 classes
model.add(Dense(10))
model.add(Activation("softmax"))

# Adam optimizer
adam = Adam(lr=1e-4)

model.compile(
    optimizer=adam,
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

print("Training ----------")
model.fit(x_train, y_train, epochs=1, batch_size=64)

print("Testing ----------")
loss, accuracy = model.evaluate(x_test, y_test)
print("\ntest loss: ", loss)
print("\ntest accuracy: ", accuracy)