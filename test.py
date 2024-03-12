from src.network import Network
from keras.datasets import fashion_mnist
import numpy as np

print('Now starts the program...')




(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
ohl_train = np.zeros([y_train.shape[0], len(np.unique(y_train))], dtype=int)
for index, item in enumerate(y_train):
  ohl_train[index, item] = 1
# print('one_hot_label', one_hot_label)
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1]*x_train.shape[2])

print(x_train.shape, ohl_train.shape)
nn = Network([x_train.shape[1], 5, ohl_train.shape[1]])

print(nn.feedforward(x_train[0]))

nn.SGD(list(zip(x_train[:10], ohl_train[:10])), epochs=10, mini_batch_size=10, eta=0.1, test_data=None)
# print(nn.feed_forward([1, 2]))