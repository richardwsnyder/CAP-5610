import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import random

# load the training and test data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# reshape
x_train = x_train.reshape(60000, 784)
x_train = x_train.astype('float32') / 255 
x_test = x_test.reshape(10000, 784)
x_test = x_test.astype('float32') / 255

# Convert the labels into vectors of size 10
# where the index of the label is 1
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# weight matrix of size 784 rows for each image 
# by 10 columns for each z node
weight = np.zeros((784, 10))
# manipulable weight matrix
w = weight

# bias vector or size 10 for each z node
bias = np.random.rand(10, 1)
b = bias

# seed random number generator
np.random.seed()

# Softmax takes in a vector z that contains the 
# calculated z values for all 10 z nodes
# This function will return a 10x1 vector, which will be stored
# as activation functions for each z node
def softmax(z):
    # find the maximum value of the z vector
    z1 = z.max()
    # subtract that maximum value from each entry in the 
    # vector and apply the exponential function to each entry
    z2 = np.exp(z - z1)
    return z2 / z2.sum()

# Categorical cross entropy loss function
# Take in a vector y (one hot encodings) and a vector z
# Apply exponential function to each item of z and take sum over 
# those values. Then take the log of that calculated value. Subtract that value
# from each entry of z as to not have underflow.
def catCrossEnt(y, z):
    return  -1 * np.sum(y.T.dot(z - np.log(np.exp(z).sum()))) 

# Categorical cross entropy derivative with respect to w_ji
# takes in an image x, a vector y (one hot encodings),
# and a vector a (activation values)
# Returns a 784 x 10 matrix of weighted values 
def partialLwrtW(x, y, a):
    dz = a - y.T 
    return x.T.dot(dz.T)

# Categorical cross entropy derivative with respect to b_j
# Returns a 10 x 1 vector of biases
def partialLwrtB(y, a):
    return np.sum(a - y.T)

# train the model based off of stochastic gradient descent
def train_model(epochs, learning_rate):
    global w
    global b

    for epoch in range(epochs):
        s = np.random.permutation(60000)
        x_shuffled = x_train[s]
        y_shuffled = y_train[s]
        for i in range(60000):
            x = x_shuffled[i].reshape(1, 784)
            y = y_shuffled[i].reshape(1, 10)
            # this will make z a 10x1 vector
            z = w.T.dot(x.T) + b 
            a = softmax(z)
            dw = partialLwrtW(x, y, a)
            db = partialLwrtB(y, a).T
            w = w - (dw * learning_rate) 
            b = b - (db * learning_rate) 
            
        print("Epoch %d completed" % epoch)

# use the newly created model to test accuracy on new
# test data
def test_model():
    global w
    global b
    loss = 0.0
    correct = 0
    for i in range(10000):
        x = x_test[i].reshape(784, 1)
        y = y_test[i].reshape(10, 1)
        z = w.T.dot(x) + b 
        a = softmax(z)
        loss += catCrossEnt(y, z) 
        if np.argmax(a) == np.argmax(y):
            correct += 1

    print("Accuracy - %0.4f percent" % ((correct / 10000) * 100))
        
train_model(10, 0.01)
test_model()