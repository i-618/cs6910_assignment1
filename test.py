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
training_inputs = x_train.reshape(-1, 784, 1)
training_results = ohl_train.reshape(-1, 10, 1)

training_data = list(zip(training_inputs[:500], training_results[:500]))
testing_data = list(zip(training_inputs[-100:], training_results[-100:]))
print(training_data[0][0][1])
net = Network([784, 100, 30, 30, 50, 10])
net.SGD(training_data, 10000, 20, 0.5, test_data=testing_data)
print(net.feedforward(training_inputs[0]))

# net.SGD(training_data, epochs=10, mini_batch_size=10, eta=0.1, test_data=testing_data)
# print(nn.feed_forward([1, 2]))