import tensorflow as tf
import numpy as np

pred_tensor1 = tf.ones((5, 11))
pred_tensor2 = tf.nn.softmax(pred_tensor1) # batch_size * 11
pred = tf.arg_max(pred_tensor2, 1) # batch_size * 1

labels = tf.ones((5, 5))
label = labels[:,0]

correct_y = tf.cast(label, tf.int64)
R = tf.cast(tf.equal(pred, correct_y), tf.float32) # reward per example
reward = tf.reduce_mean(R) # overall reward

sess= tf.Session()
# print(sess.run(correct_y))
# print(sess.run(R))
a = sess.run(pred_tensor1)
b = sess.run(pred_tensor2)
c = sess.run(pred)
l = sess.run(label)
re = sess.run(reward)

r = sess.run(R)


print(np.shape(b))
print(np.shape(c))

print(c)
print(l)
print(re)
