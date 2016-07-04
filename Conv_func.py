import tensorflow as tf


def conv_layer(self,idx,inputs,filters,size,stride):
	channels = inputs.get_shape()[3]
	weight = tf.Variable(tf.truncated_normal([size,size,int(channels),filters], stddev=0.1))
	biases = tf.Variable(tf.constant(0.1, shape=[filters]))

	pad_size = size//2
	pad_mat = np.array([[0,0],[pad_size,pad_size],[pad_size,pad_size],[0,0]])
	inputs_pad = tf.pad(inputs,pad_mat)

	conv = tf.nn.conv2d(inputs_pad, weight, strides=[1, stride, stride, 1], padding='VALID',name=str(idx)+'_conv')	
	conv_biased = tf.add(conv,biases,name=str(idx)+'_conv_biased')	
	if self.disp_console : print '    Layer  %d : Type = Conv, Size = %d * %d, Stride = %d, Filters = %d, Input channels = %d' % (idx,size,size,stride,filters,int(channels))
	return tf.maximum(self.alpha*conv_biased,conv_biased,name=str(idx)+'_leaky_relu')

def pooling_layer(self,idx,inputs,size,stride):
	if self.disp_console : print '    Layer  %d : Type = Pool, Size = %d * %d, Stride = %d' % (idx,size,size,stride)
	return tf.nn.max_pool(inputs, ksize=[1, size, size, 1],strides=[1, stride, stride, 1], padding='SAME',name=str(idx)+'_pool')


self.x = tf.placeholder('float32',[None,448,448,3])
self.conv_1 = self.conv_layer(1,self.x,16,3,1)
self.pool_2 = self.pooling_layer(2,self.conv_1,2,2)