import numpy as np 


class NeuralNetwork:
    def __init__(self, input_dim, output_dim ,network_layers_architecture: list):

        self.main_model = []
        for index, layer in enumerate(network_layers_architecture):
            self.main_model.append({'weights': [], 'biases': [], 'activation': ''})
            if index == 0:
                self.main_model[index]['weights'] = np.random.rand(layer['num_neurons'], input_dim)
            elif index == len(network_layers_architecture) - 1:
                self.main_model[index]['weights'] = np.random.rand(output_dim, layer['num_neurons'])
            else:
                self.main_model[index]['weights'] = np.random.rand(layer['num_neurons'], layer[index-1]['num_neurons'])
            self.main_model[index]['activation'] =  layer['activation']
            self.main_model[index]['biases'] = np.random.rand(1)[0]

    def feed_forward(self, input_data):
        print(self.main_model[0]['weights'].shape)
        final_output = np.dot(self.main_model[0]['weights'], input_data) + self.main_model[0]['biases']
        activation = getattr(self, self.main_model[0]['activation'], 'sigmoid')
        final_output = activation(final_output)
        for layer in self.main_model[1:]:
            final_output = np.dot(layer['weights'], final_output) + layer['biases']
            activation = getattr(self, layer['activation'], 'sigmoid')
            final_output = activation(final_output)
        return final_output
    
    def relu(self, pre_activation):
        return np.maximum(0, pre_activation)

    def softmax(self, pre_activation):
        return np.exp(pre_activation) / np.sum(np.exp(pre_activation), axis=0)

    def train(self):
        pass

    def predict(self):
        pass