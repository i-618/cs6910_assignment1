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

        self.mom_grad_descent_attr = {'previous_delta_weight': [np.zeros_like(layer) for layer in self.main_model['weights']], 
                                      'previous_delta_bias': [np.zeros_like(bias) for bias in self.main_model['biases']],
                                      'previous_learnrate_delta_weight': [np.zeros_like(layer) for layer in self.main_model['weights']], 
                                      'previous_learnrate_delta_bias': [np.zeros_like(bias) for bias in self.main_model['biases']],
                                      }
        

    def feed_forward(self, input_data, return_layer_outputs=False, specific_model=None):
            # Using the latest weights to caluculate the output
            main_model = self.main_model
            # Specific model is used for NAG Optimizer where weights are different from current weights
            if specific_model:
                main_model = specific_model
            layer_outputs = {'activation':[input_data], 'pre_activation':[]} 
            # First Layer feed forward
            final_output = np.dot(main_model['weights'][0], input_data) + main_model['biases'][0]
            layer_outputs['pre_activation'].append(final_output)
            activation = getattr(self, main_model['activation'][0], self.sigmoid)
            final_output = activation(final_output)
            layer_outputs['activation'].append(final_output)
            # Rest of the layers feed forward
            for i in range(1, self.num_layers):
                final_output = np.dot(main_model['weights'][i], final_output) + main_model['biases'][i]
                layer_outputs['pre_activation'].append(final_output)
                activation = getattr(self, main_model['activation'][i], self.sigmoid)
                final_output = activation(final_output)
                layer_outputs['activation'].append(final_output)
            if return_layer_outputs:
                return layer_outputs
            else:
                return final_output
       
    
    def back_propagation(self, x, y, specific_model=None):
        # refered http://neuralnetworksanddeeplearning.com/chap2.html
        
        # Using the latest weights to calculate the gradient
        main_model = self.main_model
        # Specific model is used for NAG Optimizer where weights are different from current weights
        if specific_model:
            main_model = specific_model

       
        delta_weights = [np.zeros_like(layer) for layer in main_model['weights']]
        delta_biases = [np.zeros_like(bias) for bias in main_model['biases']]
        layer_outputs = self.feed_forward(x, return_layer_outputs=True, specific_model=specific_model)
        cost_derivative = layer_outputs['activation'][-1] - y
        last_layer_derivative = getattr(self, self.last_layer_activation + '_derivative', self.sigmoid_derivative)
        delta = cost_derivative * last_layer_derivative(layer_outputs['pre_activation'][-1])
        
        delta_biases[-1] = delta
        delta_weights[-1] = np.dot(delta, layer_outputs['activation'][-2].transpose())
        
        for i in range(2, self.num_layers):
            activation_derivative = getattr(self, main_model['activation'][-i]+'_derivative', self.sigmoid_derivative)
            delta = np.dot(main_model['weights'][-i+1].transpose(), delta) * activation_derivative (layer_outputs['pre_activation'][-i])
            delta_biases[-i] = delta
            delta_weights[-i] = np.dot(delta, layer_outputs['activation'][-i-1].transpose())
        return delta_weights, delta_biases



    def softmax(self, pre_activation):
        return np.exp(pre_activation) / np.sum(np.exp(pre_activation), axis=0)
    
    def softmax_derivative(self, pre_activation):
        return pre_activation
    
    def sigmoid(self, pre_activation):
        return 1 / (1 + np.exp(-pre_activation))

    def sigmoid_derivative(self, pre_activation):
        return pre_activation * (1 - pre_activation)
    
    def linear_derivative(self, pre_activation):
        return 1
    
    def relu_derivative(self, pre_activation):
        return 1 * (pre_activation > 0)
    
    def tanh(self, pre_activation):
        return np.tanh(pre_activation)
    
    def identity(self, pre_activation):
        return pre_activation
    
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


    def train(self, train_data, test_data, epochs, learning_rate, optimizer, weight_decay, batch_size):
        gradient_descent_optimizer = getattr(self, optimizer, self.stochastic_gradient_descent)
        for epoch in range(epochs):
            num_batches = len(train_data['inputs']) // batch_size
            for batch in range(num_batches):
                gradient_descent_optimizer(train_data, batch_size, batch, learning_rate)
            if epoch % 10 == 0:
                print('epoch:', epoch, 'loss:', self.total_loss(test_data))

    
    def stochastic_gradient_descent(self, train_data, batch_size, batch, learning_rate):
        delta_weight_complete, delta_bias_complete = self.gradient_batch_data(train_data, batch_size, batch)

        self.main_model['weights'] = [self.main_model['weights'][i] - learning_rate * delta_weight_complete[i] for i in range(self.num_layers)]
        self.main_model['biases'] = [self.main_model['biases'][i] - learning_rate * delta_bias_complete[i] for i in range(self.num_layers)]

    def momentum_gradient_descent(self, train_data, batch_size, batch, learning_rate, beta):
        delta_weight_complete, delta_bias_complete = self.gradient_batch_data(train_data, batch_size, batch)

        momentum_gradient_weights = [beta * self.mom_grad_descent_attr['previous_delta_weight'][i] + learning_rate * delta_weight_complete[i] for i in range(self.num_layers)]
        momentum_gradient_bias = [beta * self.mom_grad_descent_attr['previous_delta_bias'][i] + learning_rate * delta_bias_complete[i] for i in range(self.num_layers)]
        
        self.main_model['weights'] = [self.main_model['weights'][i] - momentum_gradient_weights[i] for i in range(self.num_layers)]
        self.main_model['biases'] = [self.main_model['biases'][i] - momentum_gradient_bias[i] for i in range(self.num_layers)]

        self.mom_grad_descent_attr['previous_delta_weight'] = momentum_gradient_weights
        self.mom_grad_descent_attr['previous_delta_bias'] = momentum_gradient_bias

    def nag_gradient_descent(self, train_data, batch_size, batch, learning_rate, beta):
        look_ahead_weights = [self.main_model['weights'][i] - beta * self.mom_grad_descent_attr['previous_delta_weight'][i] for i in range(self.num_layers)]
        look_ahead_bias = [self.main_model['biases'][i] - beta * self.mom_grad_descent_attr['previous_delta_bias'][i] for i in range(self.num_layers)]
        look_ahead_model = {'weights': look_ahead_weights, 'biases': look_ahead_bias}

        delta_weight_complete, delta_bias_complete = self.gradient_batch_data(train_data, batch_size, batch, look_ahead_model)

        nestrov_delta_weight = [beta * self.mom_grad_descent_attr['previous_delta_weight'][i] + learning_rate * delta_weight_complete[i] for i in range(self.num_layers)]
        nestrov_delta_bias = [beta * self.mom_grad_descent_attr['previous_delta_bias'][i] + learning_rate * delta_bias_complete[i] for i in range(self.num_layers)]

        self.main_model['weights'] = [self.main_model['weights'][i] - nestrov_delta_weight[i] for i in range(self.num_layers)]
        self.main_model['biases'] = [self.main_model['biases'][i] - nestrov_delta_bias[i] for i in range(self.num_layers)]

        self.mom_grad_descent_attr['previous_delta_weight'] = nestrov_delta_weight
        self.mom_grad_descent_attr['previous_delta_bias'] = nestrov_delta_bias

    def rms_gradient_descent(self, train_data, batch_size, batch, learning_rate, beta, epsilon):
        delta_weight_complete, delta_bias_complete = self.gradient_batch_data(train_data, batch_size, batch)

        rms_gradient_weights = [beta * self.mom_grad_descent_attr['previous_delta_weight'][i] + (1 - beta) * np.square(delta_weight_complete[i]) for i in range(self.num_layers)]
        rms_gradient_bias = [beta * self.mom_grad_descent_attr['previous_delta_bias'][i] + (1 - beta) * np.square(delta_bias_complete[i]) for i in range(self.num_layers)]

        self.main_model['weights'] = [self.main_model['weights'][i] - learning_rate * delta_weight_complete[i] / (np.sqrt(rms_gradient_weights[i]) + epsilon) for i in range(self.num_layers)]
        self.main_model['biases'] = [self.main_model['biases'][i] - learning_rate * delta_bias_complete[i] / (np.sqrt(rms_gradient_bias[i]) + epsilon) for i in range(self.num_layers)]

        self.mom_grad_descent_attr['previous_delta_weight'] = rms_gradient_weights
        self.mom_grad_descent_attr['previous_delta_bias'] = rms_gradient_bias

    def adam_gradient_descent(self, train_data, batch_size, batch, epoch, learning_rate, beta1, beta2, epsilon):
        delta_weight_complete, delta_bias_complete = self.gradient_batch_data(train_data, batch_size, batch)

        self.mom_grad_descent_attr['previous_delta_weight'] = [beta1 * self.mom_grad_descent_attr['previous_delta_weight'][i] + (1 - beta1) * delta_weight_complete[i] for i in range(self.num_layers)]
        delta_weight_bias_correction = [self.mom_grad_descent_attr['previous_delta_weight'][i] / (1 - beta1 ** (epoch + 1)) for i in range(self.num_layers)]
        self.mom_grad_descent_attr['previous_delta_bias'] = [beta1 * self.mom_grad_descent_attr['previous_delta_bias'][i] + (1 - beta1) * delta_bias_complete[i] for i in range(self.num_layers)]
        delta_bias_bias_correction = [self.mom_grad_descent_attr['previous_delta_bias'][i] / (1 - beta1 ** (epoch + 1)) for i in range(self.num_layers)]    

        self.mom_grad_descent_attr['previous_learnrate_delta_weight'] = [beta2 * self.mom_grad_descent_attr['previous_learnrate_delta_weight'][i] + (1 - beta2) * np.square(delta_weight_complete[i]) for i in range(self.num_layers)]
        learnrate_delta_weight_bias_correction = [self.mom_grad_descent_attr['previous_learnrate_delta_weight'][i] / (1 - beta2 ** (epoch + 1)) for i in range(self.num_layers)]
        self.mom_grad_descent_attr['previous_learnrate_delta_bias'] = [beta2 * self.mom_grad_descent_attr['previous_learnrate_delta_bias'][i] + (1 - beta2) * np.square(delta_bias_complete[i]) for i in range(self.num_layers)]
        learnrate_delta_bias_bias_correction = [self.mom_grad_descent_attr['previous_learnrate_delta_bias'][i] / (1 - beta2 ** (epoch + 1)) for i in range(self.num_layers)]

        self.main_model['weights'] = [self.main_model['weights'][i] - learning_rate * delta_weight_bias_correction[i] / (np.sqrt(learnrate_delta_weight_bias_correction[i]) + epsilon) for i in range(self.num_layers)]
        self.main_model['biases'] = [self.main_model['biases'][i] - learning_rate * delta_bias_bias_correction[i] / (np.sqrt(learnrate_delta_bias_bias_correction[i]) + epsilon) for i in range(self.num_layers)]

    def nadam_gradient_descent(self, train_data, batch_size, batch, learning_rate, epoch, beta1, beta2, epsilon):
        # refered https://github.com/TannerGilbert/Machine-Learning-Explained/blob/master/Optimizers/nadam/code/nadam.py
        delta_weight_complete, delta_bias_complete = self.gradient_batch_data(train_data, batch_size, batch)

        self.mom_grad_descent_attr['previous_delta_weight'] = [beta1 * self.mom_grad_descent_attr['previous_delta_weight'][i] + (1 - beta1) * delta_weight_complete[i] for i in range(self.num_layers)]
        delta_weight_bias_correction = [self.mom_grad_descent_attr['previous_delta_weight'][i] / (1 - beta1 ** (epoch + 1)) for i in range(self.num_layers)]
        self.mom_grad_descent_attr['previous_delta_bias'] = [beta1 * self.mom_grad_descent_attr['previous_delta_bias'][i] + (1 - beta1) * delta_bias_complete[i] for i in range(self.num_layers)]
        delta_bias_bias_correction = [self.mom_grad_descent_attr['previous_delta_bias'][i] / (1 - beta1 ** (epoch + 1)) for i in range(self.num_layers)]    

        self.mom_grad_descent_attr['previous_learnrate_delta_weight'] = [beta2 * self.mom_grad_descent_attr['previous_learnrate_delta_weight'][i] + (1 - beta2) * np.square(delta_weight_complete[i]) for i in range(self.num_layers)]
        learnrate_delta_weight_bias_correction = [self.mom_grad_descent_attr['previous_learnrate_delta_weight'][i] / (1 - beta2 ** (epoch + 1)) for i in range(self.num_layers)]
        self.mom_grad_descent_attr['previous_learnrate_delta_bias'] = [beta2 * self.mom_grad_descent_attr['previous_learnrate_delta_bias'][i] + (1 - beta2) * np.square(delta_bias_complete[i]) for i in range(self.num_layers)]
        learnrate_delta_bias_bias_correction = [self.mom_grad_descent_attr['previous_learnrate_delta_bias'][i] / (1 - beta2 ** (epoch + 1)) for i in range(self.num_layers)]

        self.main_model['weights'] = [self.main_model['weights'][i] - learning_rate * (beta1 * delta_weight_bias_correction[i] + (1 - beta1) * delta_weight_complete[i] / (1 - beta1 ** (epoch + 1))) / (np.sqrt(learnrate_delta_weight_bias_correction[i]) + epsilon) for i in range(self.num_layers)]
        self.main_model['biases'] = [self.main_model['biases'][i] - learning_rate * (beta1 * delta_bias_bias_correction[i] + (1 - beta1) * delta_bias_complete[i] / (1 - beta1 ** (epoch + 1))) / (np.sqrt(learnrate_delta_bias_bias_correction[i]) + epsilon) for i in range(self.num_layers)]


    

    def gradient_batch_data(self, train_data, batch_size, batch, specific_model=None):
        delta_weight_complete = [np.zeros_like(layer) for layer in self.main_model['weights']]
        delta_bias_complete = [np.zeros_like(bias) for bias in self.main_model['biases']]
        train_input_batch = train_data['inputs'][batch*batch_size: (batch+1)*batch_size]
        train_label_batch = train_data['labels'][batch*batch_size: (batch+1)*batch_size]
        for x, y in zip(train_input_batch, train_label_batch):
            delta_weight, delta_bias = self.back_propagation(x, y, specific_model=specific_model)
            delta_weight_complete = [delta_weight_complete[i] + delta_weight[i] for i in range(self.num_layers)]
            delta_bias_complete = [delta_bias_complete[i] + delta_bias[i] for i in range(self.num_layers)]
        delta_weight_complete = [delta_weight_complete[i] / batch_size for i in range(self.num_layers)]
        delta_bias_complete = [delta_bias_complete[i] / batch_size for i in range(self.num_layers)]
        return delta_weight_complete,delta_bias_complete

        
    
    
