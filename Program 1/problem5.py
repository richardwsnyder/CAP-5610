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

visited = [0] * 784

parents = list(range(0, 784))

(x_train, y_train), (x_test, y_test) = mnist.load_data()

image_count, rows, cols = x_train.shape

x_train = tf.keras.utils.normalize(x_train, axis = 1)
x_test = tf.keras.utils.normalize(x_test, axis = 1)

x_train_ = x_train
x_test_ = x_test

x_train_[x_train_ > 0] = 1
x_test_[x_test_ > 0] = 1

def add_features(train, test):
	for q in range(image_count):
		np.append(train[q], djset(x_train_[q]))

	for q in range(10000):
		np.append(x_test[q], djset(x_test_[q]))

def find(v):
	if(parents[v] == v):
		return v;
	res = find(parents[v])
	parents[v] = res
	return res;

def dfs(x, i, j):
	global visited
	global parents

	dx = [0, 1, 0, -1]
	dy = [1, 0, -1, 0]

	current_pixel = (i * rows) + j

	visited[current_pixel] = 1
	if x[i][j] == 1:
		parents[current_pixel] = -1
		return

	for n in range(4):
		new_x = j + dx[n]
		new_y = i + dy[n]
		next_pixel = (i * rows) + j + (cols * dy[n]) + dx[n]
		if 0 <= new_y < rows and 0 <= new_x < cols:
			if x[new_y][new_x] == 1:
				parents[next_pixel] = -1

		if 0 <= new_y < rows and 0 <= new_x < cols:
			if x[new_y][new_x] == 0 and visited[next_pixel] == 0:
				parents[next_pixel] = find(current_pixel)
				dfs(x, new_y, new_x)

def djset(x):
	for i in range(28):
		for j in range(28):
			current_pixel = (i * rows) + j
			if visited[current_pixel] == 0:
				dfs(x, i, j)

	# print(np.unique(parents))
	return len(np.unique(parents)) - 1

add_features(x_train_, x_test_)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))


sgd = tf.keras.optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)

model.compile(optimizer=sgd, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=10, batch_size=32)

test_loss, test_acc = model.evaluate(x_test, y_test)
print(test_loss, test_acc)