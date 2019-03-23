from tensorflow.keras.datasets import cifar10
import matplotlib.pyplot as plt
import numpy as np

# load the dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# define the class names
class_names = ['airplan', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# create the number of folds for k-fold 
# validation
k_folds = 10

# shuffle the training data
p = np.random.permutation(50000)
x_train_shuffled = x_train[p]
y_train_shuffled = y_train[p]

# size of validation block based off
# of number of folds
size_validation = x_train_shuffled.shape[0] / k_folds

x_validation_list = []
y_validation_list = []

x_training_list = []
y_training_list = []

# append the different validation sets to their
# respective validation lists
for i in range(k_folds):
	x_validation_list.append(x_train_shuffled[(i * size_validation) : ((i + 1) * size_validation)])
	y_validation_list.append(y_train_shuffled[(i * size_validation) : ((i + 1) * size_validation)])

# append the different validation sets to their
# respective validation lists
for i in range(k_folds):
	x_training_list.append(x_train_shuffled[((i - 1) * size_validation) : ])
	y_training_list.append(y_train_shuffled[(i * size_validation) : ((i + 1) * size_validation)])