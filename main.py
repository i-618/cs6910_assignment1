from src.NeuralNetwork import NeuralNetwork
from keras.datasets import fashion_mnist
import numpy as np

print('Now starts the program...')


train_data={'inputs':[[0,0],[0,1],[1,0],[1,1]], 'labels':[[0],[1],[1],[0]]}

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
one_hot_label = np.zeros([y_train.shape[0], len(np.unique(y_train)), 1], dtype=int)
for index, item in enumerate(y_train):
  one_hot_label[index, item] = [1]

x_train_flattened = x_train.reshape(-1, 784, 1)/np.max(x_train)


layers = [
          {'num_neurons': 50, 'activation': 'relu'},
          {'num_neurons': 50, 'activation': 'tanh'},
          {'num_neurons': 50, 'activation': 'sigmoid'},
          ]
nn = NeuralNetwork(input_dim=x_train_flattened.shape[1], output_dim=one_hot_label.shape[1], nn_archtre=layers, 
                   last_layer_activation='softmax', weight_initializer='xavier')
resp = nn.feed_forward(x_train_flattened[0], return_layer_outputs=False)
print(sum(resp), resp)


train_records_count = int(len(x_train)*0.9)
test_records_count = int(len(x_train)*0.1)
train_data={'inputs':x_train_flattened[:train_records_count], 'labels':one_hot_label[:train_records_count]}
val_data={'inputs':x_train_flattened[-test_records_count:], 'labels':one_hot_label[-test_records_count:]}
print('num train data:', len(train_data['inputs']), 'num val data:', len(val_data['inputs']))
nn.train(train_data=train_data, val_data=val_data, epochs=5, learning_rate=0.0005, optimizer='nadam', weight_decay=0.000, batch_size=50, print_every_epoch=1)

# for inp in train_data['inputs']:
#     print(inp, nn.feed_forward(inp, return_layer_outputs=False))

# print(nn.feed_forward([1, 2]))