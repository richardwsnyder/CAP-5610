from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import cifar10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
input_shape = (32, 32, 3)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.astype('float32')
x_train /= 255

x_test = x_test.astype('float32')
x_test /= 255

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
import tensorflow.keras
from tensorflow.keras.backend import clear_session

length = 10000
accuracy_stats = []
for i in range(5):
  start_idx = i * length
  end_idx = (i + 1) * length
  val_set_x = x_train[start_idx:end_idx]
  val_set_y = y_train[start_idx:end_idx]
  
  train_x = np.concatenate((x_train[:start_idx], x_train[end_idx:]))
  train_y = np.concatenate((y_train[:start_idx], y_train[end_idx:]))
  
  cnn = Sequential()
  cnn.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
  cnn.add(BatchNormalization())

  cnn.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
  cnn.add(BatchNormalization())
  cnn.add(MaxPooling2D(pool_size=(2, 2)))
  cnn.add(Dropout(0.25))

  cnn.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
  cnn.add(BatchNormalization())
  cnn.add(Dropout(0.25))

  cnn.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
  cnn.add(BatchNormalization())
  cnn.add(MaxPooling2D(pool_size=(2, 2)))
  cnn.add(Dropout(0.25))

  cnn.add(Flatten())

  cnn.add(Dense(512, activation='relu'))
  cnn.add(BatchNormalization())
  cnn.add(Dropout(0.5))

  cnn.add(Dense(128, activation='relu'))
  cnn.add(BatchNormalization())
  cnn.add(Dropout(0.5))

  cnn.add(Dense(10, activation='softmax'))
  
  cnn.compile(loss=tensorflow.keras.losses.categorical_crossentropy,
              optimizer=tensorflow.keras.optimizers.Adam(),
              metrics=['accuracy'])

  cnn.fit(train_x, train_y,
            batch_size=256,
            epochs=50,
            verbose=1,
            validation_data=(val_set_x, val_set_y))
  score1 = cnn.evaluate(x_test, y_test, verbose=0)
  accuracy_stats.append(score1[1])
  clear_session()
sum = 0.0
for num in accuracy_stats:
  sum += num
avg_score = sum / 5
print(avg_score)