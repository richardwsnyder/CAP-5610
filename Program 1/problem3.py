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

def train(learning_rate):
	global w
	global b
	z = np.zeros((10, 1))
	x_i = x_train[0]
	y_i = y_train[0]
	for j in range(10):
		z[j] = calculateZ(w.T[j], x_i, b[j])
		print(z[j])


train(0.5)
