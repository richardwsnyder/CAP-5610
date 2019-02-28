import numpy as np
from tensorflow.keras.datasets import mnist

# activation function for logistic regression
def sigma(z):
		return 1.0 / (1.0 + np.exp(-z))

# derivative of activation function
def sigmaPrime(z):
	return sigma(z) * (1 - sigma(z))

# calculate the value of z
def calculateZ(w, x, b):
	return w.T.dot(x) + b

# binary cross entropy loss function
def bceLoss(a, y):
	return (-1.0 * (y * np.log(a))) - ((1 - y) * np.log(y - a))

# gradient of bce with respect to w_i
def gradient_w(a, y, x_i):
	return (a - y) * x_i

# gradient of bce with respect to b
def gradient_b(a, y):
	return a - y

# seed the random number generator
np.random.seed()

# load mnist data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape((60000, 28*28, 1))
x_train = x_train.astype('float32') / 255

x_test = x_test.reshape((10000, 28*28, 1))
x_test = x_test.astype('float32') / 255

# Classifier class
class Classifier:
	
	# initialize the class with a specific number that it will classify
	# and a randomized weight vector, with bias 0
	def __init__(self, num):
		self.number = num
		self.weight = np.random.randn(784, 1)
		self.b = 0

	# train the Classifier by manipulating the loss and bias
	def train(self, epochs):

		for epoch in range(epochs):
			for i in range(60000):
				# print(i)
				x_i = x_train[i]
				z = calculateZ(self.weight, x_i, self.b)
				a = sigma(z)
				aPrime = sigmaPrime(z)
				y = 0
				if self.number == y_train[i]:
					y = 1
				loss = bceLoss(a, y)
				lr = 0.5
				self.weight -= lr * gradient_w(a, y, x_i)
				self.b -= lr * gradient_b(a, y)
			print("finished epoch {} for classifier {}".format(epoch, self.number))
	# predict a test value
	def predict(self, x_i):
		z = calculateZ(self.weight, x_i, self.b)
		return sigma(z)

# create the 10 Classifiers that represent each digit
class_0 = Classifier(0)
class_1 = Classifier(1)
class_2 = Classifier(2)
class_3 = Classifier(3)
class_4 = Classifier(4)
class_5 = Classifier(5)
class_6 = Classifier(6)
class_7 = Classifier(7)
class_8 = Classifier(8)
class_9 = Classifier(9)

# train each Classifier
class_0.train(3)
class_1.train(3)
class_2.train(3)
class_3.train(3)
class_4.train(3)
class_5.train(3)
class_6.train(3)
class_7.train(3)
class_8.train(3)
class_9.train(3)

# prediction is used for each individual test image
prediction = []

# predicted value based off of argmax()
predictVal = 0

# count how many predictions you get correct
counter = 0

# loop through all 10,000 test images
for p in range(10000):
	# print("this is the expected  value: " + str(y_test[p]))
	prediction.append(class_0.predict(x_test[p]))
	prediction.append(class_1.predict(x_test[p]))
	prediction.append(class_2.predict(x_test[p]))
	prediction.append(class_3.predict(x_test[p]))
	prediction.append(class_4.predict(x_test[p]))
	prediction.append(class_5.predict(x_test[p]))
	prediction.append(class_6.predict(x_test[p]))
	prediction.append(class_7.predict(x_test[p]))
	prediction.append(class_8.predict(x_test[p]))
	prediction.append(class_9.predict(x_test[p]))
	# print("this is the predicted value: " + str(np.argmax(prediction)))
	# print()
	predictVal = np.argmax(prediction)
	prediction.clear()
	if(predictVal == y_test[p]):
		counter = counter + 1

print(counter / 10000)