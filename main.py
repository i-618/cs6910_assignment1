from src.NeuralNetwork import NeuralNetwork
from keras.datasets import fashion_mnist
import numpy as np

print('Now starts the program...')


train_data={'inputs':[[0,0],[0,1],[1,0],[1,1]], 'labels':[[0],[1],[1],[0]]}

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
one_hot_label = np.zeros([y_train.shape[0], len(np.unique(y_train))], dtype=int)
for index, item in enumerate(y_train):
  one_hot_label[index, item] = 1

x_train_flattened = x_train.reshape(-1, 784)


layers = [{'num_neurons': 3, 'activation': 'sigmoid'},
          {'num_neurons': 3, 'activation': 'sigmoid'},
          {'num_neurons': 5, 'activation': 'sigmoid'},
          {'num_neurons': 10, 'activation': 'sigmoid'},
          {'num_neurons': 9, 'activation': 'sigmoid'},
          {'num_neurons': 3, 'activation': 'sigmoid'},]
nn = NeuralNetwork(input_dim=x_train_flattened.shape[1], output_dim=one_hot_label.shape[1], nn_archtre=layers, last_layer_activation='sigmoid')
resp = nn.feed_forward(x_train_flattened[0], return_layer_outputs=False)
print(sum(resp), resp)

train_data={'inputs':x_train_flattened[:20], 'labels':one_hot_label[:20]}

# nn.train(train_data=train_data, epochs=5000, learning_rate=0.8)

# for inp in train_data['inputs']:
#     print(inp, nn.feed_forward(inp, return_layer_outputs=False))

# print(nn.feed_forward([1, 2]))