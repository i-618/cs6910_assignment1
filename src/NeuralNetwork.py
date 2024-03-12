import numpy as np 
class NeuralNetwork:

    def __init__(self, input_dim ,nn_archtre: list, output_dim, last_layer_activation='softmax', weight_initializer='random'):

        self.num_layers = len(nn_archtre) + 1
        self.main_model = {'weights': [], 'biases': [], 'activation': []}
        self.last_layer_activation = last_layer_activation
        initializer = getattr(self, 'init_' + weight_initializer, self.init_random)
        # Initializing First Layer that takes in input_dim
        self.main_model['weights'].append(initializer(nn_archtre[0]['num_neurons'], input_dim))
        self.main_model['biases'].append(initializer(1, 1))
        self.main_model['activation'].append(nn_archtre[0]['activation'])

        # Initializing N minus 2 hidden layers 
        for index, layer in enumerate(nn_archtre[1:self.num_layers-1]):
            self.main_model['weights'].append(initializer(layer['num_neurons'], nn_archtre[index]['num_neurons']))
            self.main_model['biases'].append(initializer(1, 1))
            self.main_model['activation'].append(layer['activation'])

        # Initializing the last output layer that gives out output_dim
        self.main_model['weights'].append(initializer(output_dim, nn_archtre[self.num_layers-2]['num_neurons']))
        self.main_model['biases'].append(initializer(1, 1))
        self.main_model['activation'].append(last_layer_activation)

        [print(i.shape, j) for i, j in zip(self.main_model['weights'], self.main_model['activation'])]
        

    def feed_forward(self, input_data, return_layer_outputs=True):
            layer_outputs = {'activation':[input_data], 'pre_activation':[]} 
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
       
        delta_weights = [np.zeros_like(layer) for layer in self.main_model['weights']]
        delta_biases = [np.zeros_like(bias) for bias in self.main_model['biases']]
        layer_outputs = self.feed_forward(x, return_layer_outputs=True)
        cost_derivative = layer_outputs['activation'][-1] - y
        last_layer_derivative = getattr(self, self.last_layer_activation + '_derivative', self.sigmoid_derivative)
        delta = cost_derivative * last_layer_derivative(layer_outputs['pre_activation'][-1])
        
        delta_biases[-1] = delta
        delta_weights[-1] = np.dot(delta, layer_outputs['activation'][-2].transpose())
        
        for i in range(2, self.num_layers):
            activation_derivative = getattr(self, self.main_model['activation'][-i]+'_derivative', self.sigmoid_derivative)
            delta = np.dot(self.main_model['weights'][-i+1].transpose(), delta) * activation_derivative (layer_outputs['pre_activation'][-i])
            delta_biases[-i] = delta
            delta_weights[-i] = np.dot(delta, layer_outputs['activation'][-i-1].transpose())
        return delta_weights, delta_biases

    def softmax(self, pre_activation):
        return np.exp(pre_activation) / np.sum(np.exp(pre_activation), axis=0)
    
    def softmax_derivative(self, pre_activation):
        return pre_activation

    
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

    def stochastic_gradient_descent(self, delta_weight_complete, delta_bias_complete, learning_rate):
        self.main_model['weights'] = [self.main_model['weights'][i] - learning_rate * delta_weight_complete[i] for i in range(self.num_layers)]
        self.main_model['biases'] = [self.main_model['biases'][i] - learning_rate * delta_bias_complete[i] for i in range(self.num_layers)]

    def momentum_gradient_descent(self, delta_weight_complete, delta_bias_complete, learning_rate):
        self.main_model['weights'] = [self.main_model['weights'][i] - learning_rate * delta_weight_complete[i] for i in range(self.num_layers)]
        self.main_model['biases'] = [self.main_model['biases'][i] - learning_rate * delta_bias_complete[i] for i in range(self.num_layers)]



    def train(self, train_data, test_data, epochs, learning_rate, optimizer, weight_decay, batch_size):
        gradient_descent_optimizer = getattr(self, optimizer, self.stochastic_gradient_descent)
        for epoch in range(epochs):
            num_batches = len(train_data['inputs']) // batch_size
            for batch in range(num_batches):
                delta_weight_complete = [np.zeros_like(layer) for layer in self.main_model['weights']]
                delta_bias_complete = [np.zeros_like(bias) for bias in self.main_model['biases']]
                train_input_batch = train_data['inputs'][batch*batch_size: (batch+1)*batch_size]
                train_label_batch = train_data['labels'][batch*batch_size: (batch+1)*batch_size]
                for x, y in zip(train_input_batch, train_label_batch):
                    delta_weight, delta_bias = self.back_propagation(x, y)
                    delta_weight_complete = [delta_weight_complete[i] + delta_weight[i] for i in range(self.num_layers)]
                    delta_bias_complete = [delta_bias_complete[i] + delta_bias[i] for i in range(self.num_layers)]
                delta_weight_complete = [delta_weight_complete[i] / batch_size for i in range(self.num_layers)]
                delta_bias_complete = [delta_bias_complete[i] / batch_size for i in range(self.num_layers)]
            gradient_descent_optimizer(delta_weight_complete, delta_bias_complete, learning_rate)
            if epoch % 10 == 0:
                print('epoch:', epoch, 'loss:', self.total_loss(test_data))

        
    def init_xavier(self, input_dim, output_dim):
        return np.random.randn(input_dim, output_dim) * np.sqrt(1 / input_dim)
    
    def init_random(self, input_dim, output_dim):
        return np.random.randn(input_dim, output_dim)

    def total_loss(self, test_data, loss_type = 'mse'):
        loss_type = getattr(self, 'loss_' + loss_type, self.loss_mse)
        return np.sum([loss_type(y, self.feed_forward(x, return_layer_outputs=False)) for x, y in test_data])

            
    def loss_mse(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    def loss_cross_entropy(self, y_true, y_pred):
        return -np.sum(y_true * np.log(y_pred))

    def loss_mse_derivative(self, y_true, y_pred):
        return y_pred - y_true
    
    def loss_cross_entropy_derivative(self, y_true, y_pred):
        return y_pred - y_true
    
