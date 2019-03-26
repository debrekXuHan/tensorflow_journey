#-----Using Neural Network to train this dataset-----#
import input_data
import tensorflow as tf
import os
import cv2
import numpy as np

model_saving_path = "./model"
model_name = "mnist_cnn_model"

def image_trans(image_path):
    im = cv2.imread(image_path)
    im = cv2.resize(im, (28, 28))
    im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)

    im = cv2.adaptiveThreshold(im, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 10)
    im = 1.0 - (im / 255.0)

    im = im.reshape(1, 28*28)
    im = im.astype("float32")
    return im

def mnist_conv_nn_train():
    mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)

    x = tf.placeholder("float", [None, 784], name="x")
    y_ = tf.placeholder("float", [None, 10], name="y_")

    # --1. Initialize weights and bias as small positive numbers
    # Add some normal noise on weights to avoid symmetry and 0-gradient
    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev = 0.1)
        return tf.Variable(initial)

    def bias_variable(shape):
        initial = tf.constant(0.1, shape = shape)
        return tf.Variable(initial)


    # --2. Convolution and Pooling
    def conv2d(x, W):
        return tf.nn.conv2d(x, W, strides = [1, 1, 1, 1], padding = 'SAME')

    def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize = [1, 2, 2, 1],
                              strides = [1, 2, 2, 1], padding = 'SAME')

    # --3. 1st-layer convolution
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    x_image = tf.reshape(x, [-1, 28, 28, 1])

    #ReLU Neurons
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    # --4. 2nd-layer convolution
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    # --5. Fully connected layer
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # --6. Dropout to reduce over-fitting
    keep_prob = tf.placeholder("float", name="keep_prob")
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # --7. Output layer
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])

    y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2, name="softmax")

    # --8. Train the model and evaluate the accuracy
    cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = 0.8)
    config = tf.ConfigProto(gpu_options = gpu_options)

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()

        # --9. Train the model
        iter_num = 5000
        batch_num = 50
        for i in range(iter_num):
            batch = mnist.train.next_batch(batch_num)
            if 0 == (i % 1000):
                train_accuracy = sess.run(accuracy, feed_dict = {
                    x: batch[0], y_: batch[1], keep_prob: 1.0})
                print("step %d, training accuracy %g" %(i, train_accuracy))
            sess.run(train_step, feed_dict = {x: batch[0], y_: batch[1], keep_prob: 0.5})
        saver.save(sess, os.path.join(model_saving_path, model_name), global_step=iter_num)

        # --10. Accuracy of the model for testing data
        test_accuracy = sess.run(accuracy, feed_dict = {
            x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})
        print("testing accuracy %g" %(test_accuracy))

def mnist_conv_nn_test(image):
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(os.path.join(model_saving_path, model_name+"-5000.meta"))
        saver.restore(sess, tf.train.latest_checkpoint(model_saving_path))

        graph = tf.get_default_graph()
        x = graph.get_tensor_by_name('x:0')
        y_ = graph.get_tensor_by_name('y_:0')
        keep_prob = graph.get_tensor_by_name('keep_prob:0')

        y = np.zeros((1, 10), dtype="float32")
        feed_dict = {x:image, y_:y, keep_prob:1.0}

        result = graph.get_tensor_by_name('softmax:0')
        prob = sess.run(result, feed_dict)
        print(prob)

        max_prob = np.max(prob)
        index = np.argmax(prob, axis=1)
        print("This number is {:d}.".format(index[0]))
