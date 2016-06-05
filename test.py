import tensorflow as tf
import numpy as np



m = tf.ones((10,2))
n = tf.constant([-1, -1], shape = [1,2], dtype = tf.float32)
test= m*n

sess =tf.Session()

print(sess.run(test))



