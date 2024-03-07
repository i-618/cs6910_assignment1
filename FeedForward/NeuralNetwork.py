import numpy as np 
class NeuralNetwork:
    def __init__(self, input_dim, output_dim ,nn_archtre: list, last_layer_activation='softmax'):

        self.num_layers = len(nn_archtre) + 1
        self.main_model = {'weights': [], 'biases': [], 'activation': []}
        # Initializing First Layer that takes in input_dim
        self.main_model['weights'].append(np.random.rand(nn_archtre[0]['num_neurons'], input_dim))
        self.main_model['biases'].append(np.random.rand())
        self.main_model['activation'].append(nn_archtre[0]['activation'])

        # Initializing N - 2 hidden layers 
        for index, layer in enumerate(nn_archtre[1:self.num_layers-1]):
            self.main_model['weights'].append(np.random.rand(layer['num_neurons'], nn_archtre[index]['num_neurons']))
            self.main_model['biases'].append(np.random.rand())
            self.main_model['activation'].append(layer['activation'])

        # Initializing the last output layer that gives out output_dim
        self.main_model['weights'].append(np.random.rand(output_dim, nn_archtre[self.num_layers-2]['num_neurons']))
        self.main_model['biases'].append(np.random.rand())
        self.main_model['activation'].append(last_layer_activation)

        [print(i.shape, j) for i, j in zip(self.main_model['weights'], self.main_model['activation'])]
        print('arr_len', len(self.main_model['activation']))
        print('num_layers', self.num_layers)

    def feed_forward(self, input_data, return_layer_outputs=True):
            layer_outputs = {'activation':[], 'pre_activation':[]} 
            # First Layer feed forward
            final_output = np.dot(self.main_model['weights'][0], input_data) + self.main_model['biases'][0]
            layer_outputs['pre_activation'].append(final_output)
            activation = getattr(self, self.main_model['activation'][0], self.sigmoid)
            final_output = activation(final_output)
            layer_outputs['activation'].append(final_output)
            # Rest of the layers feed forward
            for i in range(1, self.num_layers):
                final_output = np.dot(self.main_model['weights'][i], final_output) + self.main_model['biases'][i]
                layer_outputs['pre_activation'].append(final_output)
                activation = getattr(self, self.main_model['activation'][i], self.sigmoid)
                final_output = activation(final_output)
                layer_outputs['activation'].append(final_output)
            if return_layer_outputs:
                return layer_outputs
            else:
                return final_output
       
    
    def back_propagation(self, x, y):
        # delta_weights = [np.random.rand(*layer.shape) for layer in self.main_model['weights']]
        # delta_biases = [np.random.rand() for _ in self.main_model['biases']]
        delta_weights = [np.zeros_like(layer) for layer in self.main_model['weights']]
        delta_biases = [np.zeros_like(bias) for bias in self.main_model['biases']]
        # x, y = train_data['inputs'][0], train_data['labels'][0]

        
                # feed forward
        layer_outputs = self.feed_forward(x, return_layer_outputs=True)
        print('loss', self.loss_mse(y, layer_outputs['activation'][-1]))
        # print('self num of layers    ', self.num_layers)
        # print('shape of layer_outputs activation    ', list(map(lambda x: x.shape, layer_outputs['activation'])))
        # print('shape of layer_outputs pre-activation', list(map(lambda x: x.shape, layer_outputs['pre_activation'])))
        layer_ouput = layer_outputs['activation'][-1]
        dAL = - (np.divide(y, layer_ouput) - np.divide(1 - y, 1 - layer_ouput))
        dZ = dAL * self.sigmoid_derivative(layer_ouput)
        delta_weights[-1] = 1 / x.shape[0] * np.dot(dZ, layer_ouput.T)
        delta_biases[-1] = 1 / x.shape[0] * np.sum(dZ, keepdims=True)

        for i in range(2, self.num_layers):
            
            dA_prev = np.dot( self.main_model['weights'][-i+1].T, dZ)
            
            actv_derivative = getattr(self, self.main_model['activation'][-i] + '_derivative' , self.sigmoid_derivative)
            
            dZ = dA_prev * actv_derivative(layer_outputs['pre_activation'][-i])
            delta_weights[-i] = 1 / x.shape[0] * np.dot(dZ, layer_outputs['activation'][-i].T)
            delta_biases[-i] = 1 / x.shape[0] * np.sum(dZ, keepdims=True)
        
        

        return delta_weights, delta_biases

    def softmax(self, pre_activation):
        return np.exp(pre_activation) / np.sum(np.exp(pre_activation), axis=0)
    
    def softmax_derivative(self, pre_activation):
        jacobian_m = np.diag(pre_activation)

        for i in range(len(jacobian_m)):
            for j in range(len(jacobian_m)):
                if i == j:
                    jacobian_m[i][j] = pre_activation[i] * (1-pre_activation[i])
                else: 
                    jacobian_m[i][j] = -pre_activation[i]*pre_activation[j]
        return jacobian_m

    
    def sigmoid_derivative(self, pre_activation):
        return pre_activation * (1 - pre_activation)
    
    def linear_derivative(self, pre_activation):
        return 1
    
    def relu_derivative(self, pre_activation):
        return 1 * (pre_activation > 0)
    
    def sigmoid(self, pre_activation):
        return 1 / (1 + np.exp(-pre_activation))
    
    def tanh(self, pre_activation):
        return np.tanh(pre_activation)
    
    def identity(self, pre_activation):
        return pre_activation

    def train(self, train_data, epochs, learning_rate):
        print('training started...')
        if len(train_data['inputs'][0]) != self.main_model['weights'][0].shape[1]:
            raise ValueError('Input dimension does not match the input layer dimension')
        elif type(train_data['labels']) != np.ndarray and type(train_data['inputs']) != np.ndarray:
            train_data['labels'] = np.array(train_data['labels'])
            train_data['inputs'] = np.array(train_data['inputs'])
        for epoch in range(epochs):
            for x, y in zip(train_data['inputs'], train_data['labels']):
                delta_weight, delta_bias = self.back_propagation(x, y)
            # print('delta_weight', delta_weight)
                self.main_model['weights'] = [self.main_model['weights'][i] - learning_rate * delta_weight[i] for i in range(self.num_layers)]
                self.main_model['biases'] = [self.main_model['biases'][i] - learning_rate * delta_bias[i] for i in range(self.num_layers)]
        print('training ended...')
        


            
    def loss_mse(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    def loss_cross_entropy(self, y_true, y_pred):
        return -np.sum(y_true * np.log(y_pred))

    def predict(self):
        pass