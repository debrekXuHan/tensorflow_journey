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
Use the dataset from AI Challenge 2018: https://challenger.ai/competition/fl2018 <br>
Use U-Net to train the a medical image segmentation neural network.

## data_prediction
We have some optical data from kindergarten and use the Lightgbm to train a model to give out optical diagnosis.

## cats_n_dogs
I used the fine-tuning method with pre-trained ResNet50, InceptionV3 and Xception. Neural Network weights are trained from Imagenet.<br>
The dataset is from Kaggle: https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/data 

## mask_rcnn
pre-trained model download:  [mask_rcnn_coco.h5](https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5)
balloon dataset download: [balloon_dataset.zip](https://github.com/matterport/Mask_RCNN/releases/download/v2.1/balloon_dataset.zip)
