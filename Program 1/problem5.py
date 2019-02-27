# utilizing the union-find algorithms defined by 
# Arup Guha at http://www.cs.ucf.edu/~dmarino/ucf/cop3503/sampleprogs/djset.java

import numpy as np
import tensorflow as tf
import math
import matplotlib.pyplot as plt
import random
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# load mnist data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# initialize visited array to show that no
# pixel has been visited
visited = [0] * 784

# each parent is assumed to be set to 0
parents = [0] * 784

# load mnist data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# number of images, rows and columns in each image
image_count, rows, cols = x_train.shape

# normalize the dataset so that pixel values are between
# 0 and 1
x_train = tf.keras.utils.normalize(x_train, axis = 1)
x_test = tf.keras.utils.normalize(x_test, axis = 1)

# make a new set that is manipulable
x_train_ = x_train
x_test_ = x_test

# if a pixel value is greater than zero, 
# set it to one for further computation
x_train_[x_train_ > 0] = 1
x_test_[x_test_ > 0] = 1

# find algorithm defined by Arup Guha
def find(v): 
	if(parents[v] == v):
		return v;
	res = find(parents[v])
	parents[v] = res
	return res;

# run dfs on the image
def dfs(x, i, j):
	global visited
	global parents

	# create two disparity vectors that correspond
	# to the differnece in position. Will correlate to 
	# moving left, right, up, and down
	dx = [0, 1, 0, -1]
	dy = [1, 0, -1, 0]

	# index of current pixel in the visited and 
	# parents array
	current_pixel = (i * rows) + j

	# you're at the image, so set it's visited 
	# boolean to true
	visited[current_pixel] = 1

	# if the pixel is black, then set the parent
	# value to -1 because it is not considered
	# a white region
	if x[i][j] == 1:
		parents[current_pixel] = -1
		return

	# else, you must go to the pixel to the left, right
	# up, and down from it
	for n in range(4):
		new_x = j + dx[n]
		new_y = i + dy[n]

		# get the next pixel that you're moving to
		next_pixel = (i * rows) + j + (cols * dy[n]) + dx[n]

		# if the new pixel is in bounds and it's black, 
		# set the parents cell to -1
		if 0 <= new_y < rows and 0 <= new_x < cols:
			if x[new_y][new_x] == 1:
				parents[next_pixel] = -1

		# else, if you haven't visited the pixel and it's a white 
		# pixel, run dfs on that
		if 0 <= new_y < rows and 0 <= new_x < cols:
			if x[new_y][new_x] == 0 and visited[next_pixel] == 0:
				parents[next_pixel] = find(current_pixel)
				dfs(x, new_y, new_x)

# run dfs on all pixels, so long as they haven't been visited
def djset(x):
	for i in range(28):
		for j in range(28):
			current_pixel = (i * rows) + j
			if visited[current_pixel] == 0:
				dfs(x, i, j)

	return len(np.unique(parents)) - 1

# append the calculated regions value to the image's 
# input vector
def add_features(train, test):
	for q in range(image_count):
		np.append(train[q], djset(x_train_[q]))

	for q in range(10000):
		np.append(x_test[q], djset(x_test_[q]))

# create the new dataset that includes the the region count
# to each image in the two sets
add_features(x_train_, x_test_)

# create the model the same way as problem4
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

model.compile(optimizer='sgd', 
			  loss='sparse_categorical_crossentropy', 
			  metrics=['accuracy'])
model.fit(x_train, y_train, epochs=10)

val_los, val_acc = model.evaluate(x_test, y_test)
print(val_los, val_acc)