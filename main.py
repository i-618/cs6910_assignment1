print('Now starts the program...')

from FeedForward.NeuralNetwork import NeuralNetwork
layers = [{'num_neurons': 3, 'activation': 'relu'},
          {'num_neurons': 3, 'activation': 'softmax'},]
nn = NeuralNetwork(input_dim=2, output_dim=5, nn_archtre=layers)



print(nn.feed_forward([1, 2]))
nn.train(train_data=[[1, 2], [3, 4]], epochs=100, learning_rate=0.01)
print(nn.feed_forward([1, 2]))