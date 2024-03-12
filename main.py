from src.NeuralNetwork import NeuralNetwork
from keras.datasets import fashion_mnist
import numpy as np

print('Now starts the program...')

layers = [{'num_neurons': 3, 'activation': 'sigmoid'},
          {'num_neurons': 3, 'activation': 'sigmoid'},
          {'num_neurons': 5, 'activation': 'sigmoid'},
          {'num_neurons': 10, 'activation': 'sigmoid'},
          {'num_neurons': 9, 'activation': 'sigmoid'},
          {'num_neurons': 3, 'activation': 'sigmoid'},]
nn = NeuralNetwork(input_dim=2, output_dim=1, nn_archtre=layers, last_layer_activation='sigmoid')

train_data={'inputs':[[0,0],[0,1],[1,0],[1,1]], 'labels':[[0],[1],[1],[0]]}

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
one_hot_label = np.zeros([y_train.shape[0], len(np.unique(y_train))], dtype=int)
for index, item in enumerate(y_train):
  one_hot_label[index, item] = 1
# print('one_hot_label', one_hot_label)

print(nn.feed_forward([1, 2], return_layer_outputs=False))
nn.train(train_data=train_data, epochs=5000, learning_rate=0.8)
for inp in train_data['inputs']:
    print(inp, nn.feed_forward(inp, return_layer_outputs=False))

# print(nn.feed_forward([1, 2]))