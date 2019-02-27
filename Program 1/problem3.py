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