print('Now starts the program...')

from FeedForward.NeuralNetwork import NeuralNetwork
layers = [{'num_neurons': 3, 'activation': 'relu'},
          {'num_neurons': 3, 'activation': 'softmax'},]
nn = NeuralNetwork(input_dim=2, output_dim=5, network_layers_architecture=layers)

print(nn.feed_forward([1, 2]))
