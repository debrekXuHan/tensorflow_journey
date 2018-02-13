#-----Using Adam Optimizer to train this dataset-----#
import input_data
import tensorflow as tf

mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)
sess = tf.Session()

x = tf.placeholder("float", shape = [None, 784])
y_ = tf.placeholder("float", shape = [None, 10])

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

sess.run(tf.global_variables_initializer())
