print('Now starts the program...')

from FeedForward.NeuralNetwork import NeuralNetwork
layers = [{'num_neurons': 3, 'activation': 'relu'},
          {'num_neurons': 3, 'activation': 'relu'},
          {'num_neurons': 5, 'activation': 'relu'},
          {'num_neurons': 10, 'activation': 'relu'},
          {'num_neurons': 9, 'activation': 'relu'},
          {'num_neurons': 3, 'activation': 'linear'},]
nn = NeuralNetwork(input_dim=2, output_dim=1, nn_archtre=layers, last_layer_activation='sigmoid')

train_data={'inputs':[[0,0],[0,1],[1,0],[1,1]], 'labels':[[0],[1],[1],[0]]}

print(nn.feed_forward([1, 2], return_layer_outputs=False))
nn.train(train_data=train_data, epochs=5000, learning_rate=0.8)
for inp in train_data['inputs']:
    print(inp, nn.feed_forward(inp, return_layer_outputs=False))

# print(nn.feed_forward([1, 2]))