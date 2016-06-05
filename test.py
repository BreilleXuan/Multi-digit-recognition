import tensorflow as tf
import numpy as np

a = tf.constant([[1.,3.,4.,10.,10.],[1.,3.,4.,10.,10.]], shape=(2,5))
b = tf.constant([[1,1,3,3,4],[5,10,10,10,10]])

# correct_y = tf.cast(labels, tf.int64)
# R = tf.cast(tf.equal(pred, correct_y), tf.float32) # reward per example
# reward = tf.reduce_mean(R)


sess= tf.Session()
print(sess.run(a))
