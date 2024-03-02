import numpy as np 


class NeuralNetwork:
    def __init__(self, input_dim, output_dim ,nn_archtre: list):

        self.num_layers = len(nn_archtre) + 1
        self.main_model = {'weights': [], 'biases': [], 'activation': []}
        # Initializing First Layer that takes in input_dim
        self.main_model['weights'].append(np.random.rand(input_dim, nn_archtre[0]['num_neurons']))
        self.main_model['biases'].append(np.random.rand())
        self.main_model['activation'].append(nn_archtre[0]['activation'])

        # Initializing N - 2 hidden layers 
        for index, layer in enumerate(nn_archtre[1:self.num_layers-1]):
            self.main_model['weights'].append(np.random.rand(nn_archtre[index]['num_neurons'], layer['num_neurons']))
            self.main_model['biases'].append(np.random.rand())
            self.main_model['activation'].append(layer['activation'])

        # Initializing the last output layer that gives out output_dim
        self.main_model['weights'].append(np.random.rand(nn_archtre[self.num_layers-2]['num_neurons'], output_dim))
        self.main_model['biases'].append(np.random.rand())
        self.main_model['activation'].append(nn_archtre[self.num_layers-2]['activation'])

        # [print(i.shape) for i in self.main_model['weights']]
        

    def feed_forward(self, input_data):
        # First Layer feed forward
        final_output = np.dot(input_data, self.main_model['weights'][0]) + self.main_model['biases'][0]
        activation = getattr(self, self.main_model['activation'][0], self.sigmoid)
        final_output = activation(final_output)
        # Rest of the layers feed forward
        for i in range(1, self.num_layers):
            final_output = np.dot(final_output, self.main_model['weights'][i]) + self.main_model['biases'][i]
            activation = getattr(self, self.main_model['activation'][i], self.sigmoid)
            final_output = activation(final_output)
        return final_output
    
    def back_propagation(self, train_data):
        delta_weights = [np.random.rand(*layer.shape) for layer in self.main_model['weights']]
        delta_biases = [np.random.rand() for _ in self.main_model['biases']]
        return delta_weights, delta_biases

    def softmax(self, pre_activation):
        return np.exp(pre_activation) / np.sum(np.exp(pre_activation), axis=0)
    
    def sigmoid(self, pre_activation):
        return 1 / (1 + np.exp(-pre_activation))
    
    def tanh(self, pre_activation):
        return np.tanh(pre_activation)
    
    def identity(self, pre_activation):
        return pre_activation

    def train(self, train_data, epochs, learning_rate):
        print('training started...')
        for epoch in range(epochs):
            delta_weight, delta_bias = self.back_propagation(train_data)
            self.main_model['weights'] = [self.main_model['weights'][i] - learning_rate * delta_weight[i] for i in range(self.num_layers)]
            self.main_model['biases'] = [self.main_model['biases'][i] - learning_rate * delta_bias[i] for i in range(self.num_layers)]
        print('training ended...')
        


            
    def loss_mse(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    def loss_cross_entropy(self, y_true, y_pred):
        return -np.sum(y_true * np.log(y_pred))

    def predict(self):
        pass