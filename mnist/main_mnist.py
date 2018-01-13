import input_data
import tensorflow as tf

mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)

#Softmax Model: y = softmax(Wx + b)

# We need to input any number of images whose dimension is 28*28=764
# for each. "None" means the first dimension can be in any length.
x = tf.placeholder("float", [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x, W) + b)

y_ = tf.placeholder("float", [None, 10])
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# Ready to run the model
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

iter_num = 1000
batch_num = 100
for i in range(iter_num):
    batch_xs, batch_ys = mnist.train.next_batch(batch_num)
    sess.run(train_step, feed_dict = {x: batch_xs, y_: batch_ys})

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

print("Softmax Model accuracy: ", sess.run(accuracy, feed_dict = {x: mnist.test.images, y_: mnist.test.labels}))