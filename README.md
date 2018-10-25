# README
This records my learning proces of tensorflow. I upload all the codes I wrote during this study process.<br>
**Enjoy it!**

## test
After TensorFlow_GPU was installed, I wrote the code to test if it works well.
* tensorflow_test1.py<br>
Import TensorFlow module to print "Hello World!"<br>
* tensorflow_test2.py<br>
Use TensorFlow to fit a linear plane.

## mnist
Use the MNIST Dataset to build tensorflow model.
* test_mnist.py<br>
In this demo, I input the image data and the label data. Then try to print out the label and show the image by a certain index.<br>
* main.py<br>
Main function file in MNIST folder.<br>
* input_data.py<br>
By using the method 'read_data_sets', we can divide the dataset into: `mnist.train.images`, `mnist.train.labels`, `mnist.test.images`, `mnist.test.labels`.<br>
* mnist_softmax.py<br>
Use TensorFlow to train a Softmax Model for MNIST dataset and check the model accuracy using the testing data.<br>
* mnist_conv_nn.py<br>
Use TensorFlow to train a neural network model and optimize with Adam Optimizer.<br>

## oct_unet
Use the dataset from AI Challenge 2018: https://challenger.ai/competition/fl2018
Use U-Net to train the a medical image segmentation neural network.