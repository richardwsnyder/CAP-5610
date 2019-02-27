import numpy as np
import tensorflow as tf
import math
import matplotlib.pyplot as plt
import random
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)

x_train.astype('float32') / 255
x_test.astype('float32') / 255

# change to categorical encoding for loss function
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

np.random.seed()

# weight matrix has to be of size (784, 10) because
# each image has 784 pixel that can have weighted inputs, 
# and you must apply those weights differently across 
# each z
weight = np.zeros((784, 10))
# make a manipulable version of weight
w = weight

# 10 different biases to correspond to the 10 different 
# z values you can calculate
bias = np.random.rand(10, 1)
# make a manipulable version of bias
b = bias

# calculate the value of z
def calculateZ(w, x, b):
	return w.T.dot(x) + b

# pass in the array z that contains the 10 calculated 
# values from calculateZ()
def softmax(z, k):
	# take the max of the array z
	z_max = z.max()
	z2 = z - z_max
	return np.exp(z[k]) / z.sum()

# cross entropy loss function
def cat_cross_ent(y, a):
	loss = 0.0
	for k in range(10):
		loss += y[k] * np.log(a[k])

	return -1 * loss

# partial derivative of L with respect to 
# z_j
def partialLwrtZ(y, a, j):
	sum = 0.0
	for k in range(10):
		kroneker = 0
		if j == k:
			kroneker = 1
		sum += y[k] * (a[j] - kroneker)

	return sum

# partial derivative of L with respect to 
# w_ji
def partialLwrtW(y, a, j, x):
	return partialLwrtZ(y, a, j) * x

# started working on gradient descent
def train(learning_rate):
	global w
	global b
	z = np.zeros((10, 1))
	a = np.zeros((10, 1))
	for i in range(60000):
		x_i = x_train[0]
		y_i = y_train[0]
		for j in range(10):
			z[j] = calculateZ(w.T[j], x_i, b[j])
		for k in range(10):
			a[k] = softmax(z, k)
		for j in range(10):
			w.T[j] -= learning_rate * partialLwrtW(y_i, a, j, x_i)
	
def test():
	global w
	global b

	count = 0
	z = np.zeros((10, 1))
	a = np.zeros((10, 1))
	for i in range(10000):
		x = x_test[i]
		y = y_test[i]
		for j in range(10):
			z[j] = calculateZ(w.T[j], x, b[j])
		for k in range(10):
			a[k] = softmax(z, k)
		if np.argmax(a) == np.argmax(y):
			count += 1
	print("Accuracy - %0.4f" % ((count / 10000) * 100))

train(0.5)
print("done training")
test()



