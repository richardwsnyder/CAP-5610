import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist

# load mnist data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# normalize the grayscale values to between 0 and 1
x_train = tf.keras.utils.normalize(x_train, axis = 1)
x_test = tf.keras.utils.normalize(x_test, axis = 1)

# forward propogating model
model = tf.keras.models.Sequential()

# add a flattened (784 x 1) version of each image
model.add(tf.keras.layers.Flatten())

# create an output layer of 10 nodes that use the softmax activation function
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

# initialize the model to use categorical crossentropy as its loss function
model.compile(optimizer = 'sgd',
			  loss = 'sparse_categorical_crossentropy',
			  metrics = ['accuracy'])

# fit the model to the training data
model.fit(x_train, y_train, epochs=10)

# evaluate the model based on the test data
val_los, val_acc = model.evaluate(x_test, y_test)
print(val_los)
print(val_acc)