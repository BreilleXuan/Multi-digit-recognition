import tensorflow as tf
from RAM_model import *
from utils import *
import time
import numpy as np

optimizer = tf.train.AdamOptimizer(lr)
train_op = optimizer.minimize(cost)

sess = tf.InteractiveSession()
saver = tf.train.Saver()

if load_path is not None:
    try:
        saver.restore(sess, load_path)
        print("LOADED SAVED MODEL")
    except:
        print("FAILED TO LOAD SAVED MODEL")
        exit()
else:
    init = tf.initialize_all_variables()
    sess.run(init)

img_tensor, mean, std = read_img("img_name_list.txt", "img/")
img_labels, img_tensor_labels = genr_label("labels.txt")
img_tensor = (img_tensor - mean) / std

for step in xrange(max_iters):
    start_time = time.time()
    
    ind = [int(np.random.rand() * 10), int(np.random.rand() * 10)]

    feed_dict = {img: img_tensor[ind], labels: img_labels[ind], tensor_labels: img_tensor_labels[ind]}
    fetches = [train_op, output_pred, cost, reward]
    
    results = sess.run(fetches, feed_dict=feed_dict)
    _, prediction, cost_fetched, reward_fetched = results
    
    duration = time.time() - start_time

    if (step + 1) % 20 == 0:
        print('Step %d: cost = %.5f reward = %.5f (%.3f sec)' % (step + 1, cost_fetched, reward_fetched, duration))
        print("labels:")
        print(img_labels[ind])
        print("prediction")
        print(prediction)
    # if (step + 1) % 2000 == 0:
    #     evaluate(dataset) 

    if (step + 1) % 4000 == 0:
        saver.save(sess, save_dir + save_prefix + ".ckpt")
        print("Model Saved")
    
     










