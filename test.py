from src.network import Network
from keras.datasets import fashion_mnist
import numpy as np

print('Now starts the program...')




(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
ohl_train = np.zeros([y_train.shape[0], len(np.unique(y_train))], dtype=int)
for index, item in enumerate(y_train):
  ohl_train[index, item] = 1
print(ohl_train.shape)
# print('one_hot_label', one_hot_label)
# x_train = x_train.reshape(x_train.shape[0], x_train.shape[1]*x_train.shape[2])

print(x_train.shape, ohl_train.shape)
# nn = Network([x_train.shape[1], 5, ohl_train.shape[1]])
# training_data = list(zip(x_train[:10], ohl_train[:10]))
training_inputs = [np.reshape(x, (784, 1)) for x in x_train]
training_results = [np.reshape(x, (len(np.unique(y_train)), 1)) for x in ohl_train]

training_data = list(zip(training_inputs[:20], training_results[:20]))
testing_data = list(zip(training_inputs[-10:], training_results[-10:]))
print(training_data[0][0][1])
net = Network([784, 30, 10])
net.SGD(training_data, 10, 10, 3.0, test_data=testing_data)
print(net.feedforward(training_inputs[0]))

# net.SGD(training_data, epochs=10, mini_batch_size=10, eta=0.1, test_data=testing_data)
# print(nn.feed_forward([1, 2]))