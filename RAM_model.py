import tensorflow as tf
import numpy as np
from tensorflow.models.rnn import seq2seq
from tensorflow.models.rnn import rnn_cell
from RAM_parameters import *

DO_SHARE=None # workaround for variable_scope(reuse=True)

img = tf.placeholder(tf.float32, shape=(batch_size, width * height * channels), name="images")
labels = tf.placeholder(tf.float32, shape=(batch_size, max_length), name="labels")
tensor_labels = tf.placeholder(tf.float32, shape=(batch_size, max_length, n_classes+1), name="tensor_labels")

lstm_cell = rnn_cell.LSTMCell(cell_size, g_size) # encoder Op
b = tf.get_variable("b", [1], initializer=tf.constant_initializer(0.0))

def linear(x,output_dim):
    """
    affine transformation Wx+b
    assumes x.shape = (batch_size, num_features)
    """
    w=tf.get_variable("w", [x.get_shape()[1], output_dim]) 
    b=tf.get_variable("b", [output_dim], initializer=tf.constant_initializer(0.0))
    return tf.matmul(x,w)+b

def glimpseSensor(img, normLoc):
    loc = ((normLoc + 1) / 2) * tf.constant([width, height], shape = [1,2], dtype = tf.float32) # normLoc coordinates are between -1 and 1
    loc = tf.cast(loc, tf.int32)

    img = tf.reshape(img, (batch_size, width, height, channels))
    zooms = []

    max_radius = minRadius * (2 ** (depth - 1)) #2 * (2 ** (3-1)) = 8
    
    # process each image individually
    for k in xrange(batch_size):
        imgZooms = []
        one_img = img[k,:,:,:]
   
        # pad image with zeros
        one_img = tf.image.pad_to_bounding_box(one_img, max_radius, max_radius, \
            max_radius * 4 + width, max_radius * 4 + height)
        
        for i in xrange(depth): # i = 0
            r = int(minRadius * (2 ** (i))) #r = 2

            d_raw = 2 * r   #4
            d = tf.constant(d_raw, shape=[1])
            d = tf.tile(d, [2])
            d = tf.concat(0, [d, tf.constant(channels, shape=[1])])

            loc_k = loc[k,:] 
            loc_start = max_radius * 2 + loc_k - r  #location of left vertix
            loc_start = tf.concat(0, [loc_start, tf.constant(0, shape=[1])])
            
            zoom = tf.slice(one_img, loc_start, d)
            zoom = tf.image.resize_images(zoom, sensorBandwidth, sensorBandwidth)
            
            imgZooms.append(zoom)
    
        zooms.append(tf.pack(imgZooms))
        
    zooms = tf.pack(zooms)
        
    return zooms

def get_glimpse(glimpse_input, loc):
    glimpse_input = tf.reshape(glimpse_input, (batch_size, totalSensorBandwidth))    
    with tf.variable_scope("loc",reuse=DO_SHARE):
        hl = tf.nn.relu(linear(loc, hl_size)) #glimpse vector
    with tf.variable_scope("glimpse",reuse=DO_SHARE):
        hg = tf.nn.relu(linear(glimpse_input, hg_size)) # loc vector
    # combine two feature vectors   
    # g = tf.concat(1, [hl,hg])   
    g = hl * hg   
    return g
    
def sample(output):
    with tf.variable_scope("sample",reuse=DO_SHARE):
        mean_loc = tf.tanh(linear(output, 2)) #batch_size * 2
    sample_loc = mean_loc + tf.random_normal(mean_loc.get_shape(), 0, loc_sd)
    return mean_loc, sample_loc

def RNN_LSTM(input, state):
    with tf.variable_scope("lstm", reuse = DO_SHARE):
        return lstm_cell(input, state)

def lable_pred(output):
    output = tf.reshape(output, (batch_size, cell_out_size))

    with tf.variable_scope("pred", reuse = DO_SHARE):
        pred_tensor = linear(output, n_classes + 1)

    pred_tensor = tf.nn.softmax(pred_tensor)
    pred = tf.arg_max(pred_tensor, 1)
    return pred_tensor, pred

def baselineFunc():
    with tf.variable_scope("baseline", reuse = DO_SHARE):
        return b + 1

# to use for maximum likelihood with glimpse location
def gaussian_pdf(mean, sample):
    Z = 1.0 / (loc_sd * tf.sqrt(2.0 * np.pi))
    a = -tf.square(sample - mean) / (2.0 * tf.square(loc_sd))
    return Z * tf.exp(a)

def calc_reward(pred_tensor, pred, labels, tensor_labels, p_loc, baseline):

    correct_y = tf.cast(labels, tf.int64)
    R = tf.cast(tf.equal(pred, correct_y), tf.float32) # reward per example
    reward = tf.reduce_mean(R) # overall reward

    R = tf.reshape(R, (batch_size, 1))
    J = tf.concat(1, [tf.log(pred_tensor + 1e-5) * tensor_labels, lmda * tf.log(p_loc + 1e-5) * (R-baseline)])
    J = tf.reduce_sum(J, 1)
    J = tf.reduce_mean(J, 0)
    cost = -J

    return cost, reward


mean_locs, sampled_locs = [0] * total_step, [0] * total_step
pred_tensors, preds = [0] * max_length, [0] * max_length
# initial state
lstm_state = lstm_cell.zero_state(batch_size, tf.float32)
# build the graph
for t in range(total_step):
    if t == 0:
        loc = tf.random_uniform((batch_size, 2), minval=-1, maxval=1)
    else: 
        loc = sampled_locs[t-1]

    glimpse = glimpseSensor(img, loc)
    glimpse_vector = get_glimpse(glimpse, loc)   
    output, lstm_state = RNN_LSTM(glimpse_vector, lstm_state)
    mean_locs[t], sampled_locs[t] = sample(output)
    if (t + 1) % glimpses == 0:
        if (t + 1) == glimpses:
            DO_SHARE = False
        pred_tensors[(t+1) / glimpses - 1], preds[(t+1) / glimpses - 1] = lable_pred(output)  # batch_size * 1
    DO_SHARE = True

baseline = baselineFunc()

sampled_locs = tf.concat(0, sampled_locs)
sampled_locs = tf.reshape(sampled_locs, (batch_size, total_step, 2))
mean_locs = tf.concat(0, mean_locs)
mean_locs = tf.reshape(mean_locs, (batch_size, total_step, 2))

p_loc = gaussian_pdf(mean_locs, sampled_locs) #batch_size * total_step * 2
p_loc = tf.reshape(p_loc, (batch_size, total_step * 2))

cost = 0.
reward = 0.
for i in range(max_length):
    cost_, reward_ = calc_reward(pred_tensors[i], preds[i], 
                                 labels[:,i], tensor_labels[:,i, :], \
                                 p_loc[:, i * glimpses : (i+1)*glimpses], \
                                 baseline)
    cost += cost_
    reward += reward_















