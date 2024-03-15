import numpy as np 
import time


class NeuralNetwork:

    def __init__(self, input_dim ,nn_archtre: list, output_dim, last_layer_activation='softmax', weight_initializer='random'):

        self.num_layers = len(nn_archtre) + 1
        self.main_model = {'weights': [], 'biases': [], 'activation': []}
        self.last_layer_activation = last_layer_activation
        self.weight_initializer = weight_initializer
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
        self.history_loss = []
        self.mom_grad_descent_attr = {'previous_delta_weight': [np.zeros_like(layer) for layer in self.main_model['weights']], 
                                      'previous_delta_bias': [np.zeros_like(bias) for bias in self.main_model['biases']],
                                      'previous_learnrate_delta_weight': [np.zeros_like(layer) for layer in self.main_model['weights']], 
                                      'previous_learnrate_delta_bias': [np.zeros_like(bias) for bias in self.main_model['biases']],
                                      }
        
    def reinitialize_weights(self):
        initializer = getattr(self, 'init_' + self.weight_initializer, self.init_random)
        self.main_model['weights'] = [initializer(*layer.shape) * np.random.randn()*6 for layer in self.main_model['weights']]
        self.main_model['biases'] = [initializer(*bias.shape) * np.random.randn()*6 for bias in self.main_model['biases']]
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
       
    
    def back_propagation(self, x, y, loss, specific_model=None):
        # refered http://neuralnetworksanddeeplearning.com/chap2.html
        
        # Using the latest weights to calculate the gradient
        main_model = self.main_model
        # Specific model is used for NAG Optimizer where weights are different from current weights
        if specific_model:
            main_model = specific_model

       
        delta_weights = [np.zeros_like(layer) for layer in main_model['weights']]
        delta_biases = [np.zeros_like(bias) for bias in main_model['biases']]
        layer_outputs = self.feed_forward(x, return_layer_outputs=True, specific_model=specific_model)

        loss_derivative = getattr(self, 'loss_' + loss + '_derivative', self.mse_derivative)
        cost_derivative = loss_derivative(y, layer_outputs['activation'][-1])


        last_layer_derivative = getattr(self, self.last_layer_activation + '_derivative', self.sigmoid_derivative)
        delta = last_layer_derivative(layer_outputs['pre_activation'][-1], cost_derivative)
        
        delta_biases[-1] = delta
        delta_weights[-1] = np.dot(delta, layer_outputs['activation'][-2].transpose())
        
        for i in range(2, self.num_layers):
            activation_derivative = getattr(self, main_model['activation'][-i]+'_derivative', self.sigmoid_derivative)
            delta = np.dot(main_model['weights'][-i+1].transpose(), delta) * activation_derivative(layer_outputs['pre_activation'][-i])
            delta_biases[-i] = delta
            delta_weights[-i] = np.dot(delta, layer_outputs['activation'][-i-1].transpose())
        return delta_weights, delta_biases

    def train(self, train_data, val_data, epochs, learning_rate, optimizer, weight_decay, batch_size, loss='cross_entrophy', print_every_epoch=10, **kwargs):
        time_start, time_per_epoch, time_per_batch = [time.time()]*3
        
        gradient_descent_optimizer = getattr(self, optimizer + '_gradient_descent', self.stochastic_gradient_descent)
        num_batches = len(train_data['inputs']) // batch_size
        for epoch in range(epochs):
            
            for batch in range(num_batches):
                
                train_input_batch = train_data['inputs'][batch*batch_size: (batch+1)*batch_size]
                train_label_batch = train_data['labels'][batch*batch_size: (batch+1)*batch_size]
                train_data_batch = {'inputs': train_input_batch, 'labels': train_label_batch}

                gradient_descent_optimizer(train_data=train_data_batch, batch=batch, learning_rate=learning_rate, loss=loss, weight_decay=weight_decay, epoch=epoch+batch*10/num_batches, **kwargs)
                if batch % int(num_batches/5) == 0:
                    print('Seconds taken', round((time.time() - time_per_batch), 2),'batch:', f'{batch}/{num_batches}', 'train_loss_acc:', self.total_loss_accuracy(train_data), 'val_loss_acc:', self.total_loss_accuracy(val_data))
                    time_per_batch = time.time()
            if epoch % print_every_epoch == 0:
                print('Mins taken', round((time.time() - time_per_epoch)/60, 2),'epoch:', epoch, 'train_loss_acc:', self.total_loss_accuracy(train_data), 'val_loss_acc:', self.total_loss_accuracy(val_data))
                time_per_epoch = time.time()
        
        print('Total time taken for training:', round((time.time() - time_start)/60, 2), ' mins')
    

    def softmax(self, pre_activation):
        # clipping to prevent overflow
        cliped_preactivation = np.clip(pre_activation, -700, 700)
        return np.exp(cliped_preactivation) / np.sum(np.exp(cliped_preactivation), axis=0)
    
    def softmax_derivative(self, pre_activation, cost_derivative=1):
        # refered from https://neuralthreads.medium.com/backpropagation-made-super-easy-for-you-part-2-7b2a06f25f3c
        identity_matrix = np.eye(pre_activation.shape[0])
        softmax_derivative = self.softmax(pre_activation) * (identity_matrix - self.softmax(pre_activation).T)
        delta = cost_derivative * softmax_derivative
        delta = np.sum(delta, axis=0).reshape(-1, 1)
        return delta

        
    
    def sigmoid(self, pre_activation):
        # clipping to prevent overflow
        cliped_preactivation = np.clip(pre_activation, -700, 700)
        return 1.0 / (1.0 + np.exp(-cliped_preactivation))

    def tanh(self, pre_activation):
        return np.tanh(pre_activation)
    
    def sigmoid_derivative(self, pre_activation, cost_derivative=1):
        return self.sigmoid(pre_activation) * (1 - self.sigmoid(pre_activation)) * cost_derivative
    
    def linear_derivative(self, pre_activation, cost_derivative=1):
        return 1.0 * cost_derivative
    
    def relu_derivative(self, pre_activation, cost_derivative=1):
        return 1.0 * (pre_activation > 0) * cost_derivative
    
    
    def tanh_derivative(self, pre_activation, cost_derivative=1):
        return 1.0 - np.tanh(pre_activation) ** 2 * cost_derivative
    
    def identity(self, pre_activation):
        return pre_activation
    
    def init_xavier(self, input_dim, output_dim):
        return np.random.randn(input_dim, output_dim) * np.sqrt(6 / (input_dim + output_dim))
    
    def init_random(self, input_dim, output_dim):
        return np.random.randn(input_dim, output_dim)

    def total_loss_accuracy(self, test_data, loss_type = 'mse'):
        loss_type = getattr(self, 'loss_' + loss_type, self.loss_mse)
        size_data = len(test_data['inputs'])
        model_output = [self.feed_forward(test_data['inputs'][i]) for i in range(size_data)]
        total_loss = np.sum([loss_type(test_data['labels'][i], model_output[i]) for i in range(size_data) ])
        total_accuracy = np.sum([np.argmax(test_data['labels'][i]) == np.argmax(model_output[i]) for i in range(size_data) ])
        # total_loss = total_loss / size_data
        total_accuracy = total_accuracy / size_data
        self.history_loss.append(round(total_loss, 2))
        self.history_loss = self.history_loss[-6:]
        if len(self.history_loss) == 6 and len(set(self.history_loss)) == 2 or np.isnan(total_loss):
            print('loss isnan or not changing for past 3 epochs reinitializing weights')
            self.reinitialize_weights()
        return round(total_loss, 2), round(total_accuracy, 3)

    def loss_mse(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    def loss_cross_entropy(self, y_true, y_pred):
        return -np.sum(y_true * np.log(y_pred + 1e-08))

    def mse_derivative(self, y_true, y_pred):
        return y_pred - y_true
    
    def cross_entropy_derivative(self, y_true, y_pred):
        return -y_true/(y_pred + 1e-08)

    def add_l2_regularization_penalty(self, delta_weight, delta_bias, weight_decay):
        delta_weight = [delta_weight[i] + weight_decay * np.sum(self.main_model['weights'][i]) for i in range(self.num_layers)]
        delta_bias = [delta_bias[i] + weight_decay * np.sum(self.main_model['biases'][i]) for i in range(self.num_layers)]
        return delta_weight, delta_bias

  
    def stochastic_gradient_descent(self, train_data,  learning_rate, loss, weight_decay, **kwargs):
        
        delta_weight_complete, delta_bias_complete = self.gradient_batch_wise_data(train_data,  loss=loss)

        delta_weight_complete, delta_bias_complete = self.add_l2_regularization_penalty(delta_weight_complete, delta_bias_complete, weight_decay)

        self.main_model['weights'] = [self.main_model['weights'][i] - learning_rate * delta_weight_complete[i] for i in range(self.num_layers)]
        self.main_model['biases'] = [self.main_model['biases'][i] - learning_rate * delta_bias_complete[i] for i in range(self.num_layers)]

    def momentum_gradient_descent(self, train_data,  learning_rate, loss, weight_decay, beta=0.9, **kwargs):
        
        delta_weight_complete, delta_bias_complete = self.gradient_batch_wise_data(train_data,  loss=loss)

        momentum_gradient_weights = [beta * self.mom_grad_descent_attr['previous_delta_weight'][i] + learning_rate * delta_weight_complete[i] for i in range(self.num_layers)]
        momentum_gradient_bias = [beta * self.mom_grad_descent_attr['previous_delta_bias'][i] + learning_rate * delta_bias_complete[i] for i in range(self.num_layers)]
        
        momentum_gradient_weights, momentum_gradient_bias = self.add_l2_regularization_penalty(momentum_gradient_weights, momentum_gradient_bias, weight_decay)

        self.main_model['weights'] = [self.main_model['weights'][i] - momentum_gradient_weights[i] for i in range(self.num_layers)]
        self.main_model['biases'] = [self.main_model['biases'][i] - momentum_gradient_bias[i] for i in range(self.num_layers)]

        self.mom_grad_descent_attr['previous_delta_weight'] = momentum_gradient_weights
        self.mom_grad_descent_attr['previous_delta_bias'] = momentum_gradient_bias

    def nag_gradient_descent(self, train_data,  learning_rate, loss, weight_decay, beta=0.9, **kwargs):
        
        look_ahead_weights = [self.main_model['weights'][i] - beta * self.mom_grad_descent_attr['previous_delta_weight'][i] for i in range(self.num_layers)]
        look_ahead_bias = [self.main_model['biases'][i] - beta * self.mom_grad_descent_attr['previous_delta_bias'][i] for i in range(self.num_layers)]
        look_ahead_model = {'weights': look_ahead_weights, 'biases': look_ahead_bias, 'activation': self.main_model['activation']}

        delta_weight_complete, delta_bias_complete = self.gradient_batch_wise_data(train_data,  look_ahead_model, loss=loss)

        nestrov_delta_weight = [beta * self.mom_grad_descent_attr['previous_delta_weight'][i] + learning_rate * delta_weight_complete[i] for i in range(self.num_layers)]
        nestrov_delta_bias = [beta * self.mom_grad_descent_attr['previous_delta_bias'][i] + learning_rate * delta_bias_complete[i] for i in range(self.num_layers)]
        
        nestrov_delta_weight, nestrov_delta_bias = self.add_l2_regularization_penalty(nestrov_delta_weight, nestrov_delta_bias, weight_decay)
        
        self.main_model['weights'] = [self.main_model['weights'][i] - nestrov_delta_weight[i] for i in range(self.num_layers)]
        self.main_model['biases'] = [self.main_model['biases'][i] - nestrov_delta_bias[i] for i in range(self.num_layers)]

        self.mom_grad_descent_attr['previous_delta_weight'] = nestrov_delta_weight
        self.mom_grad_descent_attr['previous_delta_bias'] = nestrov_delta_bias

    def rmsprop_gradient_descent(self, train_data,  learning_rate,loss, weight_decay, beta=0.99, epsilon=1e-08, **kwargs):
        
        delta_weight_complete, delta_bias_complete = self.gradient_batch_wise_data(train_data,  loss=loss)

        rms_gradient_weights = [beta * self.mom_grad_descent_attr['previous_delta_weight'][i] + (1 - beta) * np.square(delta_weight_complete[i]) for i in range(self.num_layers)]
        rms_gradient_bias = [beta * self.mom_grad_descent_attr['previous_delta_bias'][i] + (1 - beta) * np.square(delta_bias_complete[i]) for i in range(self.num_layers)]

        delta_weight_complete, delta_bias_complete = self.add_l2_regularization_penalty(delta_weight_complete, delta_bias_complete, weight_decay)

        self.main_model['weights'] = [self.main_model['weights'][i] - learning_rate * delta_weight_complete[i] / (np.sqrt(rms_gradient_weights[i]) + epsilon) for i in range(self.num_layers)]
        self.main_model['biases'] = [self.main_model['biases'][i] - learning_rate * delta_bias_complete[i] / (np.sqrt(rms_gradient_bias[i]) + epsilon) for i in range(self.num_layers)]

        self.mom_grad_descent_attr['previous_delta_weight'] = rms_gradient_weights
        self.mom_grad_descent_attr['previous_delta_bias'] = rms_gradient_bias

    def adam_gradient_descent(self, train_data,  learning_rate, weight_decay, epoch, loss, beta1=0.9, beta2=0.999, epsilon=1e-08, **kwargs):
        
        delta_weight_complete, delta_bias_complete = self.gradient_batch_wise_data(train_data,  loss=loss)

        self.mom_grad_descent_attr['previous_delta_weight'] = [beta1 * self.mom_grad_descent_attr['previous_delta_weight'][i] + (1 - beta1) * delta_weight_complete[i] for i in range(self.num_layers)]
        delta_weight_bias_correction = [self.mom_grad_descent_attr['previous_delta_weight'][i] / (1 - beta1 ** (epoch + 1)) for i in range(self.num_layers)]
        self.mom_grad_descent_attr['previous_delta_bias'] = [beta1 * self.mom_grad_descent_attr['previous_delta_bias'][i] + (1 - beta1) * delta_bias_complete[i] for i in range(self.num_layers)]
        delta_bias_bias_correction = [self.mom_grad_descent_attr['previous_delta_bias'][i] / (1 - beta1 ** (epoch + 1)) for i in range(self.num_layers)]    

        delta_weight_bias_correction, delta_bias_bias_correction = self.add_l2_regularization_penalty(delta_weight_bias_correction, delta_bias_bias_correction, weight_decay)
        
        self.mom_grad_descent_attr['previous_learnrate_delta_weight'] = [beta2 * self.mom_grad_descent_attr['previous_learnrate_delta_weight'][i] + (1 - beta2) * np.square(delta_weight_complete[i]) for i in range(self.num_layers)]
        learnrate_delta_weight_bias_correction = [self.mom_grad_descent_attr['previous_learnrate_delta_weight'][i] / (1 - beta2 ** (epoch + 1)) for i in range(self.num_layers)]
        self.mom_grad_descent_attr['previous_learnrate_delta_bias'] = [beta2 * self.mom_grad_descent_attr['previous_learnrate_delta_bias'][i] + (1 - beta2) * np.square(delta_bias_complete[i]) for i in range(self.num_layers)]
        learnrate_delta_bias_bias_correction = [self.mom_grad_descent_attr['previous_learnrate_delta_bias'][i] / (1 - beta2 ** (epoch + 1)) for i in range(self.num_layers)]

        self.main_model['weights'] = [self.main_model['weights'][i] - learning_rate * delta_weight_bias_correction[i] / (np.sqrt(learnrate_delta_weight_bias_correction[i]) + epsilon) for i in range(self.num_layers)]
        self.main_model['biases'] = [self.main_model['biases'][i] - learning_rate * delta_bias_bias_correction[i] / (np.sqrt(learnrate_delta_bias_bias_correction[i]) + epsilon) for i in range(self.num_layers)]

    def nadam_gradient_descent(self, train_data,  learning_rate, weight_decay, epoch, loss, beta1=0.9, beta2=0.999, epsilon=1e-08, **kwargs):
        
        # refered https://github.com/TannerGilbert/Machine-Learning-Explained/blob/master/Optimizers/nadam/code/nadam.py
        delta_weight_complete, delta_bias_complete = self.gradient_batch_wise_data(train_data,  loss=loss)

        self.mom_grad_descent_attr['previous_delta_weight'] = [beta1 * self.mom_grad_descent_attr['previous_delta_weight'][i] + (1 - beta1) * delta_weight_complete[i] for i in range(self.num_layers)]
        delta_weight_bias_correction = [self.mom_grad_descent_attr['previous_delta_weight'][i] / (1 - beta1 ** (epoch + 1)) for i in range(self.num_layers)]
        self.mom_grad_descent_attr['previous_delta_bias'] = [beta1 * self.mom_grad_descent_attr['previous_delta_bias'][i] + (1 - beta1) * delta_bias_complete[i] for i in range(self.num_layers)]
        delta_bias_bias_correction = [self.mom_grad_descent_attr['previous_delta_bias'][i] / (1 - beta1 ** (epoch + 1)) for i in range(self.num_layers)]    

        delta_weight_bias_correction, delta_bias_bias_correction = self.add_l2_regularization_penalty(delta_weight_bias_correction, delta_bias_bias_correction, weight_decay)

        self.mom_grad_descent_attr['previous_learnrate_delta_weight'] = [beta2 * self.mom_grad_descent_attr['previous_learnrate_delta_weight'][i] + (1 - beta2) * np.square(delta_weight_complete[i]) for i in range(self.num_layers)]
        learnrate_delta_weight_bias_correction = [self.mom_grad_descent_attr['previous_learnrate_delta_weight'][i] / (1 - beta2 ** (epoch + 1)) for i in range(self.num_layers)]
        self.mom_grad_descent_attr['previous_learnrate_delta_bias'] = [beta2 * self.mom_grad_descent_attr['previous_learnrate_delta_bias'][i] + (1 - beta2) * np.square(delta_bias_complete[i]) for i in range(self.num_layers)]
        learnrate_delta_bias_bias_correction = [self.mom_grad_descent_attr['previous_learnrate_delta_bias'][i] / (1 - beta2 ** (epoch + 1)) for i in range(self.num_layers)]

        self.main_model['weights'] = [self.main_model['weights'][i] - learning_rate * (beta1 * delta_weight_bias_correction[i] + (1 - beta1) * delta_weight_complete[i] / (1 - beta1 ** (epoch + 1))) / (np.sqrt(learnrate_delta_weight_bias_correction[i]) + epsilon) for i in range(self.num_layers)]
        self.main_model['biases'] = [self.main_model['biases'][i] - learning_rate * (beta1 * delta_bias_bias_correction[i] + (1 - beta1) * delta_bias_complete[i] / (1 - beta1 ** (epoch + 1))) / (np.sqrt(learnrate_delta_bias_bias_correction[i]) + epsilon) for i in range(self.num_layers)]


    

    def gradient_batch_wise_data(self, train_data, loss, specific_model=None):
        delta_weight_complete = [np.zeros_like(layer) for layer in self.main_model['weights']]
        delta_bias_complete = [np.zeros_like(bias) for bias in self.main_model['biases']]

        for x, y in zip(train_data['inputs'], train_data['labels']):
            delta_weight, delta_bias = self.back_propagation(x, y, loss, specific_model=specific_model)
            delta_weight_complete = [delta_weight_complete[i] + delta_weight[i] for i in range(self.num_layers)]
            delta_bias_complete = [delta_bias_complete[i] + delta_bias[i] for i in range(self.num_layers)]
        # delta_weight_complete = [delta_weight_complete[i]  for i in range(self.num_layers)]
        # delta_bias_complete = [delta_bias_complete[i]  for i in range(self.num_layers)]
        return delta_weight_complete,delta_bias_complete

        
    
    
