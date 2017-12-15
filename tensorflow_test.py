# My first tensorflow code
import tensorflow as tf
string = 'Hello, TensorFlow!'
hello = tf.constant(string)
sess = tf.Session()
print(sess.run(hello).decode('utf-8'))  # print as unicode

a = tf.constant(10)
b = tf.constant(32)
print(sess.run(a + b))