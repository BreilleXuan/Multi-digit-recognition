import numpy as np
from RAM_parameters import *
import cv2

def dense_to_one_hot(labels):
    labels_one_hot = np.zeros((batch_size, n_classes))
    for i in range(batch_size):
        labels_one_hot[i][labels[i]] = 1
    return labels_one_hot

def read(filename):
	contents = []
	f = open(filename)
	for i in f:
		contents.append(i)
	return contents

def read_img(img_name, img_path):
	img_name = read(img_name)
	n = len(img_name)

	for i,name in enumerate(img_name):
		img = cv2.imread((img_path + name).strip())
		if i == 0:
			w,h,c = np.shape(img)
			img_tensor = np.zeros((n,w,h,c))

		img_tensor[i] = img

	mean = np.mean(img_tensor)
	std = np.std(img_tensor)

	return img_tensor.reshape(n, w*h*c), mean, std

def split_label(label_str):
	label = np.ones(max_length) * (11)
	for i in range(len(label_str)):
		label[i] = int(label_str[i])
	return label

def one_hot_label(label_str):
	one_hot_label_tensor = np.zeros((max_length, n_classes + 1))
	n = len(label_str)
	for i in range(n):
		one_hot_label_tensor[i, int(label_str[i])] = 1
	if n < max_length:
		for j in range(n, max_length):
			one_hot_label_tensor[j, -1] = 1
	return one_hot_label_tensor

def genr_label(label_file):
	rd = read(label_file)
	n = len(rd)

	labels = np.zeros((n, max_length)) * (-1)
	tensor_labels = np.zeros((n, max_length, n_classes + 1))

	for i,label in enumerate(rd):
		labels[i] = split_label(label.strip())
		tensor_labels[i] = one_hot_label(label.strip())

	return labels, tensor_labels

if __name__ == '__main__':
	img_tensor, mean, std = read_img("img_name_list.txt", "img/")
	labels, tensor_labels = genr_label("labels.txt")

	ind = [1,2]
	print(img_tensor[ind])




