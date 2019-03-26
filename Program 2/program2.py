from tensorflow.keras.datasets import cifar10
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np

batch_size = 128
num_classes = 10
epochs = 1

img_x, img_y = 32, 32

# load the dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = x_train.reshape(x_train.shape[0], img_x, img_y, 3)
x_test = x_test.reshape(x_test.shape[0], img_x, img_y, 3)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

input_shape = (img_x, img_y, 3)

model = Sequential()
model.add(Conv2D(32, (3, 3), activation = 'relu', input_shape = input_shape))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation = 'relu'))
model.add(MaxPooling2D(2, 2))
model.add(Conv2D(64, (3, 3), activation = 'relu'))
model.add(Flatten())
model.add(Dense(64, activation = 'relu'))
model.add(Dense(10, activation = 'softmax'))
model.summary()

model.compile(optimizer = 'rmsprop',
			  loss='categorical_crossentropy',
			  metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=epochs, batch_size = batch_size, validation_data=(x_test, y_test))

history_dict = history.history
loss_values = history_dict['loss']
test_loss_values = history_dict['val_loss']
epochs_range = range(1, epochs + 1)

plt.plot(epochs_range, loss_values, 'bo', label='Training loss')
plt.plot(epochs_range, test_loss_values, 'ro', label='Test loss')
plt.title('Training and test loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

acc_values = history_dict['acc']
test_acc_values = history_dict['val_acc']

plt.plot(epochs_range, acc_values, 'bo', label='Training accuracy')
plt.plot(epochs_range, test_acc_values, 'ro', label='Test accuracy')
plt.title('Training and test accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
# # define the class names
# class_names = ['airplan', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# # create the number of folds for k-fold 
# # validation
# k_folds = 10

# # shuffle the training data
# p = np.random.permutation(50000)
# x_train_shuffled = x_train[p]
# y_train_shuffled = y_train[p]

# # size of validation block based off
# # of number of folds
# size_validation = x_train_shuffled.shape[0] / k_folds

# x_validation_list = []
# y_validation_list = []

# x_training_list = []
# y_training_list = []

# # append the different validation sets to their
# # respective validation lists
# for i in range(k_folds):
# 	x_validation_list.append(x_train_shuffled[(i * size_validation) : ((i + 1) * size_validation)])
# 	y_validation_list.append(y_train_shuffled[(i * size_validation) : ((i + 1) * size_validation)])

# print(x_validation_list[0].shape)