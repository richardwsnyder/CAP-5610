from tensorflow.keras.datasets import mnist
import numpy as np
import tensorflow as tf
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = tf.keras.utils.normalize(x_train, axis = 1)
image = x_train[1]
# num rows
m = len(image)
print(m)
# num columns
n = len(image[0])
print(n)
#storage array
arr = [[-1] * n for i in range(m)]

label_counter = 1
left = 0
up = 0

for i in range(m):
	for j in range(n):
		if(image[i][j] != 0): 
			pass
		else:
			left = -1
			up = -1
			if(j > 0):
				if(image[i][j - 1] == 0):
					left = arr[i][j - 1]
			if(i > 0):
				if(image[i - 1][j] == 0):
					up = arr[i - 1][j]
			if(left == -1 and up == -1):
				arr[i][j] = label_counter
				label_counter += 1
			elif(left != -1 and up == -1):
				arr[i][j] = left
			elif(up != -1 and left == -1):
				arr[i][j] = up
			else:
				arr[i][j] = min(left, up)

for i in range(27, 0, -1):
	for j in range(27, 0, -1):
		if(arr[i][j] == -1):
			pass
		else:
			left = -1
			if(j > 0):
				left = arr[i][j - 1]

			if(arr[i][j] != -1 and left != -1 and arr[i][j - 1] > arr[i][j]):
				arr[i][j - 1] = arr[i][j]

for i in range(m):
	print(arr[i])
print()
print(y_train[1])

print("the number of white regions:", label_counter)