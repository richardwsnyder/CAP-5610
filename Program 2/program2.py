from tensorflow.keras.datasets import cifar10
import matplotlib.pyplot as plt
import numpy as np

# load the dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# define the class names
class_names = ['airplan', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# create the number of folds for k-fold 
# validation
k_folds = 5

# shuffle the training data
p = np.random.permutation(50000)
x_train_shuffled = x_train[p]
y_train_shuffled = y_train[p]