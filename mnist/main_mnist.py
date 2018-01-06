import input_data
import numpy as np
import matplotlib.pyplot as plt

mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)

index = 0
fig = plt.figure()
plt.imshow(np.reshape(mnist.train.images[index, :], [28, 28]), cmap = 'binary') # display in black and white   
plt.show()