import numpy as np 
import time
import wandb


class NeuralNetwork:
    """
    A class representing a Neural Netork written from scratch only using numpy and nothing else, this 
    code might not be good at performace as the focus is on understanding the math and concepts of internal functionings 
    of a neural network.

    Attributes:
    - input_dim (int): The dimension of the input layer, images have to be flattened to a 1d array.
    - nn_archtre (list): A list of dictionaries representing the architecture of hidden layers, got as input.
    - output_dim (int): The dimension of the output layer, number of classes for classification problems.
    - last_layer_activation (str): The activation function used in the last layer, softmax is preferred for classification problems.
    - weight_initializer (str): The weight initialization method used in the neural network, can be random or xavier.
    - main_model (dict): A dictionary containing the weights, biases, and activation functions, which are the three major info regarding a layer.
    - history_loss (list): A list to store the loss values during training to check if loss is decreasing or not.
    - mom_grad_descent_attr (dict): A dictionary to store historical gradient attributes used in momentum gradient descent, NAG, RMSProp, Adam and Nadam.

    Methods:
    - __init__(self, input_dim, nn_archtre, output_dim, last_layer_activation='softmax', weight_initializer='random'):
        Initializes the neural network with the given parameters.

    - feed_forward(self, input_data, return_layer_outputs=False, specific_model=None):
        Performs the feed-forward operation of the neural network.

    - train(self, train_data, val_data, epochs, learning_rate, optimizer, weight_decay, batch_size, loss='cross_entropy', print_every_epoch=10, **kwargs):
        Trains the neural network using the specified training data and parameters.
    """

    def __init__(self, input_dim:int ,nn_archtre: list[dict], output_dim:int, last_layer_activation:str='softmax', weight_initializer:str='random'):

        self.num_layers = len(nn_archtre) + 1
        self.main_model = {'weights': [], 'biases': [], 'activation': []}
        self.last_layer_activation = last_layer_activation
        self.weight_initializer = weight_initializer
        initializer = getattr(self, f'_init_{weight_initializer}', self._init_random)
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

        # [print(i.shape, j) for i, j in zip(self.main_model['weights'], self.main_model['activation'])]
        self.history_loss = []
        self.mom_grad_descent_attr = {'previous_delta_weight': [np.zeros_like(layer) for layer in self.main_model['weights']], 
                                      'previous_delta_bias': [np.zeros_like(bias) for bias in self.main_model['biases']],
                                      'previous_learnrate_delta_weight': [np.zeros_like(layer) for layer in self.main_model['weights']], 
                                      'previous_learnrate_delta_bias': [np.zeros_like(bias) for bias in self.main_model['biases']],
                                      }
        
    def feed_forward(self, input_data, return_layer_outputs=False):
        """Performs the feed-forward operation of the neural network.
        Used for inference to get the output of the neural network for the given input data.
        There are two main modes, first is to get just the final output and the second is to get the output of each layer.
        The second mode is used during training to get the output of each layer to calculate the gradients. The first mode 
        is enabled by default, return_layer_outputs should be set to True to get the output of each layer.

        Args:
            input_data (ndarray): A numpy array representing the input data, should be flattened to 1d array.
            return_layer_outputs (bool, optional): Returns layerwise ouputs, used in back propagation. Defaults to False.

        Returns:
            ndarray: The output of the neural network for the given input data. Or Layerwise outputs if return_layer_outputs is set to True.
        """
        layer_outputs = {'activation':[input_data], 'pre_activation':[]} 
        # First Layer feed forward, this is unique as the input is the input_data
        final_output = np.dot(self.main_model['weights'][0], input_data) + self.main_model['biases'][0]
        layer_outputs['pre_activation'].append(final_output)
        activation = getattr(self, f"_{self.main_model['activation'][0]}", self._sigmoid)
        final_output = activation(final_output)
        layer_outputs['activation'].append(final_output)
        # Rest of the layers feed forward where the input is the output of the previous layer
        for i in range(1, self.num_layers):
            final_output = np.dot(self.main_model['weights'][i], final_output) + self.main_model['biases'][i]
            layer_outputs['pre_activation'].append(final_output)
            activation = getattr(self, f"_{self.main_model['activation'][i]}", self._sigmoid)
            final_output = activation(final_output)
            layer_outputs['activation'].append(final_output)
        if return_layer_outputs:
            return layer_outputs
        else:
            return final_output
            
    def train(self, train_data, val_data, epochs, learning_rate, optimizer, weight_decay, batch_size, loss='cross_entropy', print_every_epoch=10, **kwargs):
        """Trains the neural network using the specified training data and hyperparameters.
        The training is done using the specified optimizer, loss function, and learning rate. The training data is split into batches
        and the gradients are calculated for each batch and the weights and biases are updated using the gradients.
        If a Vanilla Gradient Descent is to be performed, then the optimizer should be set to 'stochastic' 
        and the batch_size should be set to the size of the training data, meaning a single batch for whole data.
        The model is supposed to increase its accuracy of predicting the outputs after training.


        Args:
            train_data (dict): A dictionary containing the inputs and labels for the training data.
            val_data (dict): A dictionary containing the inputs and labels for the validation data.
            epochs (int): The number of iterations to go over the training data.
            learning_rate (float): The weightage given to the gradients while updating the weights and biases.
            optimizer (str): The optimizer used to update the weights and biases, can be 'stochastic', 'momentum', 'nag', 'rmsprop', 'adam', 'nadam'.
            weight_decay (float): The weight decay used to regularize the model in L2 regularization.
            batch_size (int): The size of each batch used to split the training data into batches.
            loss (str, optional): Loss type to calculate the difference between the predicted and actual values. Defaults to 'cross_entropy'.
            print_every_epoch (int, optional): Just a parameter to print loss to keep track of training. Defaults to 10.
        """

        time_start, time_per_epoch, time_per_batch = [time.perf_counter()]*3
        
        gradient_descent_optimizer = getattr(self, f'_{optimizer}_gradient_descent', self._stochastic_gradient_descent)
        print(f'Using {gradient_descent_optimizer.__name__} for traing optimization')
        
        num_batches = len(train_data['inputs']) // batch_size
        for epoch in range(epochs):
            
            for batch in range(num_batches):
                
                train_input_batch = train_data['inputs'][batch*batch_size: (batch+1)*batch_size]
                train_label_batch = train_data['labels'][batch*batch_size: (batch+1)*batch_size]
                train_data_batch = {'inputs': train_input_batch, 'labels': train_label_batch}

                gradient_descent_optimizer(train_data=train_data_batch, batch=batch, learning_rate=learning_rate, loss=loss, weight_decay=weight_decay, epoch=epoch+batch*10/num_batches, **kwargs)
                if batch % int(num_batches/2) == 0:
                    val_loss_acc = self._total_loss_accuracy(val_data, weight_decay=weight_decay)
                    print('Seconds taken', round((time.perf_counter() - time_per_batch), 2),'batch:', f'{batch + 1}/{num_batches}', 'val_loss_acc:', val_loss_acc)
                    time_per_batch = time.perf_counter()
            if False and epoch % print_every_epoch == 0:
                train_loss_acc = self._total_loss_accuracy(train_data, weight_decay=weight_decay)
                val_loss_acc = self._total_loss_accuracy(val_data, weight_decay=weight_decay)
                print('Mins taken', round((time.perf_counter() - time_per_epoch)/60, 2),'epoch:', epoch+1, 'train_loss_acc:', train_loss_acc, 'val_loss_acc:', val_loss_acc)
                time_per_epoch = time.perf_counter()
            
            trn_loss, trn_accuracy = self._total_loss_accuracy(train_data, weight_decay=weight_decay)
            val_loss, val_accuracy = self._total_loss_accuracy(val_data, weight_decay=weight_decay)
            print('Mins taken', round((time.perf_counter() - time_per_epoch)/60, 2),{'loss':trn_loss, 'accuracy': trn_accuracy, 'val_loss': val_loss, 'val_accuracy':val_accuracy, 'epoch': epoch})
            time_per_epoch = time.perf_counter()
            wandb.log({'loss':trn_loss, 'accuracy': trn_accuracy, 'val_loss': val_loss, 'val_accuracy':val_accuracy, 'epoch': epoch})
        
        print('Total time taken for training:', round((time.perf_counter() - time_start)/60, 2), ' mins')
    
    def _back_propagation(self, x, y, loss):
        """The heart of the Neural Network, it calculates the gradients for the weights and biases using backpropagation.
        This is where the actual learning happens, the gradients are calculated using the chain rule of calculus.
        It is called back propagation because the gradients are calculated from the last layer to the first layer.
        """
        # refered http://neuralnetworksanddeeplearning.com/chap2.html

       
        delta_weights = [np.zeros_like(layer) for layer in self.main_model['weights']]
        delta_biases = [np.zeros_like(bias) for bias in self.main_model['biases']]
        layer_outputs = self.feed_forward(x, return_layer_outputs=True)

        loss_derivative = getattr(self, f'_loss_{loss}_derivative', self._mse_derivative)
        cost_derivative = loss_derivative(y, layer_outputs['activation'][-1])


        last_layer_derivative = getattr(self, f'_{self.last_layer_activation}_derivative', self._sigmoid_derivative)
        delta = last_layer_derivative(layer_outputs['pre_activation'][-1], cost_derivative)
        
        delta_biases[-1] = delta
        delta_weights[-1] = np.dot(delta, layer_outputs['activation'][-2].transpose())
        
        for i in range(2, self.num_layers):
            activation_derivative = getattr(self, f"_{self.main_model['activation'][-i]}_derivative", self._sigmoid_derivative)
            delta = np.dot(self.main_model['weights'][-i+1].transpose(), delta) * activation_derivative(layer_outputs['pre_activation'][-i])
            delta_biases[-i] = delta
            delta_weights[-i] = np.dot(delta, layer_outputs['activation'][-i-1].transpose())
        return delta_weights, delta_biases

       
    def _reinitialize_weights(self):
        """ It reinitializes the weights and biases of the neural network if the loss is not changing for past 3 epochs.
            The monitoring is done in the _total_loss_accuracy function where val and train loss is checked.
            This feature is added to prevent the model from getting stuck in a local minima.
        """        
        initializer = getattr(self, f'_init_{self.weight_initializer}', self._init_random)
        self.main_model['weights'] = [initializer(*layer.shape) * np.random.randn()*6 for layer in self.main_model['weights']]
        self.main_model['biases'] = [initializer(*bias.shape) * np.random.randn()*6 for bias in self.main_model['biases']]
        self.mom_grad_descent_attr = {'previous_delta_weight': [np.zeros_like(layer) for layer in self.main_model['weights']], 
                                      'previous_delta_bias': [np.zeros_like(bias) for bias in self.main_model['biases']],
                                      'previous_learnrate_delta_weight': [np.zeros_like(layer) for layer in self.main_model['weights']], 
                                      'previous_learnrate_delta_bias': [np.zeros_like(bias) for bias in self.main_model['biases']],
                                      }

    

###############################################################
################ Activation Functions START ###################
    def _softmax(self, pre_activation):
        # clipping to prevent overflow in case of exploding gradients
        cliped_preactivation = np.clip(pre_activation, -700, 700)
        return np.exp(cliped_preactivation) / np.sum(np.exp(cliped_preactivation), axis=0)
    
    def _softmax_derivative(self, pre_activation, cost_derivative=1):
        """ There were lot of opinions regarding the derivative of softmax, some say to just use the same preactivation
        but that did not work well in practice. So, I referred to the below link to implement it.
        """
        # Refernce https://neuralthreads.medium.com/backpropagation-made-super-easy-for-you-part-2-7b2a06f25f3c
        identity_matrix = np.eye(pre_activation.shape[0])
        softmax_derivative = self._softmax(pre_activation) * (identity_matrix - self._softmax(pre_activation).T)
        delta = cost_derivative * softmax_derivative
        delta = np.sum(delta, axis=0).reshape(-1, 1)
        return delta

    def _sigmoid(self, pre_activation):
        # clipping to prevent overflow
        cliped_preactivation = np.clip(pre_activation, -700, 700)
        return 1.0 / (1.0 + np.exp(-cliped_preactivation))

    def _sigmoid_derivative(self, pre_activation, cost_derivative=1):
        return self._sigmoid(pre_activation) * (1 - self._sigmoid(pre_activation)) * cost_derivative
    
    def _tanh(self, pre_activation):
        return np.tanh(pre_activation)
    
    def _tanh_derivative(self, pre_activation, cost_derivative=1):
        return 1.0 - np.tanh(pre_activation) ** 2 * cost_derivative
    
    def _identity(self, pre_activation):
        return pre_activation
    
    def _identity_derivative(self, pre_activation, cost_derivative=1):
        return 1.0 * cost_derivative
    
    def _relu(self, pre_activation):
        return np.maximum(0, pre_activation)

    def _relu_derivative(self, pre_activation, cost_derivative=1):
        return 1.0 * (pre_activation > 0) * cost_derivative

################ Activation Functions END ######################
################################################################
    
################################################################
################ Weight Initialization START ###################

    def _init_xavier(self, input_dim, output_dim):
        return np.random.randn(input_dim, output_dim) * np.sqrt(6 / (input_dim + output_dim))
    
    def _init_random(self, input_dim, output_dim):
        return np.random.randn(input_dim, output_dim)

################ Weight Initialization END ###################
##############################################################


##############################################################
################ Loss Functions START ########################
    
    def _total_loss_accuracy(self, test_data, weight_decay, loss_type = 'cross_entropy'):
        """Calculates the total loss and accuracy for the given test data. It consumes a lot of time if the 
        test data is large, so it is better to use a small subset of the test data to check the loss and accuracy.
        There is also a feature to reinitialize the weights if the loss is not changing for past 3 epochs.
        The loss and accuracy are returned after being rounded to 2 decimal places.
        """
        loss_type = getattr(self, f'_loss_{loss_type}', self._loss_mse)
        size_data = len(test_data['inputs'])
        model_output = [self.feed_forward(test_data['inputs'][i]) for i in range(size_data)]
        total_loss = np.sum([loss_type(test_data['labels'][i], model_output[i], weight_decay) for i in range(size_data) ])
        total_accuracy = np.sum([np.argmax(test_data['labels'][i]) == np.argmax(model_output[i]) for i in range(size_data) ])
        # total_loss = total_loss / size_data
        total_accuracy = total_accuracy / size_data
        self.history_loss.append(round(total_loss, 2))
        self.history_loss = self.history_loss[-6:]
        if len(self.history_loss) == 6 and len(set(self.history_loss)) == 2 or np.isnan(total_loss):
            print('loss isnan or not changing for past 3 epochs reinitializing weights')
            self._reinitialize_weights()
        return round(total_loss, 2), round(total_accuracy, 3)

    def _loss_mse(self, y_true, y_pred, weight_decay):
        l2_loss = weight_decay/2/self.num_layers * sum([np.sum(np.square(np.clip(self.main_model['weights'][i], -500, 500))) for i in range(self.num_layers)])/self.num_layers
        return np.mean((y_true - y_pred) ** 2) + l2_loss

    def _loss_cross_entropy(self, y_true, y_pred, weight_decay):
        l2_loss = weight_decay/2/self.num_layers * sum([np.sum(np.square(np.clip(self.main_model['weights'][i], -500, 500))) for i in range(self.num_layers)])/self.num_layers
        return -np.sum(y_true * np.log(y_pred + 1e-08)) + l2_loss

    def _mse_derivative(self, y_true, y_pred):
        return y_pred - y_true 
    
    def _cross_entropy_derivative(self, y_true, y_pred):
        return -y_true/(y_pred + 1e-08) 

    def _add_l2_regularization_penalty(self, delta_weight, delta_bias, weight_decay = 0):
        delta_weight_with_penalty = [delta_weight[i] + weight_decay/self.num_layers * self.main_model['weights'][i] for i in range(self.num_layers)]
        return delta_weight_with_penalty, delta_bias

################ Loss Functions END ##########################
##############################################################
    
###############################################################
################ Gradient Descent Optimizers START ############
  
    def _gradient_batch_wise_data(self, train_data, loss, weight_decay):
        """Common function used by all the gradient descent optimizers to calculate the gradients for the batch of data.

        Args:
            train_data (dict): A dictionary containing the inputs and labels for the batch of data.
            loss ([str]): Loss type either mse or cross_entropy
            weight_decay (float): The weight decay used to regularize the model in L2 regularization.

        Returns:
            [delta_weight_complete, delta_bias_complete]: A list of layer wise gradient weights and biases for the batch of data.
        """        
        delta_weight_complete = [np.zeros_like(layer) for layer in self.main_model['weights']]
        delta_bias_complete = [np.zeros_like(bias) for bias in self.main_model['biases']]

        for x, y in zip(train_data['inputs'], train_data['labels']):
            delta_weight, delta_bias = self._back_propagation(x, y, loss)
            delta_weight_complete = [delta_weight_complete[i] + delta_weight[i] for i in range(self.num_layers)]
            delta_bias_complete = [delta_bias_complete[i] + delta_bias[i] for i in range(self.num_layers)]
        
        delta_weight_complete, delta_bias_complete = self._add_l2_regularization_penalty(delta_weight_complete, delta_bias_complete, weight_decay)

        return delta_weight_complete,delta_bias_complete
    
    def _stochastic_gradient_descent(self, train_data,  learning_rate, loss, weight_decay, **kwargs):
        """ The most simplest of all the gradient descent optimizers, 
        it updates the weights and biases with the gradients calculated for the batch of data.
        """
        # Calculate the gradients for the batch of data
        delta_weight_complete, delta_bias_complete = self._gradient_batch_wise_data(train_data, weight_decay=weight_decay, loss=loss)
        # Update the weights and biases with the gradients
        self.main_model['weights'] = [self.main_model['weights'][i] - learning_rate * delta_weight_complete[i] for i in range(self.num_layers)]
        self.main_model['biases'] = [self.main_model['biases'][i] - learning_rate * delta_bias_complete[i] for i in range(self.num_layers)]

    def _momentum_gradient_descent(self, train_data,  learning_rate, loss, weight_decay, momentum_beta=0.9, **kwargs):
        """ The momentum gradient descent optimizer uses the historical gradients to update the weights and biases.
        It is like a ball rolling down the hill, it gains momentum with each iteration and moves faster towards the minima.
        The momentum_beta is the hyperparameter that controls the momentum of the ball, that is the weightage given to historical data.
        """
        delta_weight_complete, delta_bias_complete = self._gradient_batch_wise_data(train_data, weight_decay=weight_decay, loss=loss)
        
        momentum_gradient_weights = [momentum_beta * self.mom_grad_descent_attr['previous_delta_weight'][i] + learning_rate * delta_weight_complete[i] for i in range(self.num_layers)]
        momentum_gradient_bias = [momentum_beta * self.mom_grad_descent_attr['previous_delta_bias'][i] + learning_rate * delta_bias_complete[i] for i in range(self.num_layers)]
        
        self.main_model['weights'] = [self.main_model['weights'][i] - momentum_gradient_weights[i] for i in range(self.num_layers)]
        self.main_model['biases'] = [self.main_model['biases'][i] - momentum_gradient_bias[i] for i in range(self.num_layers)]

        self.mom_grad_descent_attr['previous_delta_weight'] = momentum_gradient_weights
        self.mom_grad_descent_attr['previous_delta_bias'] = momentum_gradient_bias

    def _nag_gradient_descent(self, train_data,  learning_rate, loss, weight_decay, momentum_beta=0.9, **kwargs):
        """ The Nesterov Accelerated Gradient (NAG) Descent is one step improvement over the momentum gradient descent.
        It uses the look ahead gradient to update the weights and biases instead of the current gradient, so saving one step.
        """
        
        # Temporarily update the weights and biases to get look ahead gradient
        self.main_model['weights'] = [self.main_model['weights'][i] - momentum_beta * self.mom_grad_descent_attr['previous_delta_weight'][i] for i in range(self.num_layers)]
        self.main_model['biases'] = [self.main_model['biases'][i] - momentum_beta * self.mom_grad_descent_attr['previous_delta_bias'][i] for i in range(self.num_layers)]
        # Get the look ahead gradient 
        delta_weight_complete, delta_bias_complete = self._gradient_batch_wise_data(train_data, weight_decay=weight_decay , loss=loss)
        # Revert back to original weights and biases
        self.main_model['weights'] = [self.main_model['weights'][i] + momentum_beta * self.mom_grad_descent_attr['previous_delta_weight'][i] for i in range(self.num_layers)]
        self.main_model['biases'] = [self.main_model['biases'][i] + momentum_beta * self.mom_grad_descent_attr['previous_delta_bias'][i] for i in range(self.num_layers)]
        
        # Momentum based gradients with look ahead gradient instead of current gradient
        nestrov_delta_weight = [momentum_beta * self.mom_grad_descent_attr['previous_delta_weight'][i] + learning_rate * delta_weight_complete[i] for i in range(self.num_layers)]
        nestrov_delta_bias = [momentum_beta * self.mom_grad_descent_attr['previous_delta_bias'][i] + learning_rate * delta_bias_complete[i] for i in range(self.num_layers)]
        
        # Update the weights and biases with the momentum based gradients
        self.main_model['weights'] = [self.main_model['weights'][i] - nestrov_delta_weight[i] for i in range(self.num_layers)]
        self.main_model['biases'] = [self.main_model['biases'][i] - nestrov_delta_bias[i] for i in range(self.num_layers)]

        # Saving the look ahead gradients for next iteration
        self.mom_grad_descent_attr['previous_delta_weight'] = nestrov_delta_weight
        self.mom_grad_descent_attr['previous_delta_bias'] = nestrov_delta_bias

    def _rmsprop_gradient_descent(self, train_data,  learning_rate,loss, weight_decay, beta=0.99, epsilon=1e-08, **kwargs):
        """ The RMSProp Gradient Descent optimizer also uses the historical gradients to update the weights and biases.
        But here the historical gradients are used to reduce the learning rate so that the learning is more stable.
        The beta is the hyperparameter that controls the weightage given to historical gradients.
        """
        
        delta_weight_complete, delta_bias_complete = self._gradient_batch_wise_data(train_data, weight_decay=weight_decay, loss=loss)

        rms_gradient_weights = [beta * self.mom_grad_descent_attr['previous_delta_weight'][i] + (1 - beta) * np.square(delta_weight_complete[i]) for i in range(self.num_layers)]
        rms_gradient_bias = [beta * self.mom_grad_descent_attr['previous_delta_bias'][i] + (1 - beta) * np.square(delta_bias_complete[i]) for i in range(self.num_layers)]

        self.main_model['weights'] = [self.main_model['weights'][i] - learning_rate * delta_weight_complete[i] / (np.sqrt(rms_gradient_weights[i]) + epsilon) for i in range(self.num_layers)]
        self.main_model['biases'] = [self.main_model['biases'][i] - learning_rate * delta_bias_complete[i] / (np.sqrt(rms_gradient_bias[i]) + epsilon) for i in range(self.num_layers)]

        self.mom_grad_descent_attr['previous_delta_weight'] = rms_gradient_weights
        self.mom_grad_descent_attr['previous_delta_bias'] = rms_gradient_bias

    def _adam_gradient_descent(self, train_data,  learning_rate, weight_decay, epoch, loss, beta1=0.9, beta2=0.999, epsilon=1e-08, **kwargs):
        """ The Adam Gradient Descent optimizer also uses the historical gradients to update the weights and biases.
        It is a combination of momentum gradient descent and RMSProp gradient descent, it uses the historical gradients
        to reduce the learning rate and also to gain momentum. There are two hyperparameters beta1 and beta2 that control
        how much the learning rate is reduced and how much momentum is gained. epsilon is used to prevent division by zero.
        """
        
        delta_weight_complete, delta_bias_complete = self._gradient_batch_wise_data(train_data, weight_decay=weight_decay, loss=loss)

        # Calculating the momentum based gradients that gain momentum with each iteration of similar gradients
        self.mom_grad_descent_attr['previous_delta_weight'] = [beta1 * self.mom_grad_descent_attr['previous_delta_weight'][i] + (1 - beta1) * delta_weight_complete[i] for i in range(self.num_layers)]
        delta_weight_bias_correction = [self.mom_grad_descent_attr['previous_delta_weight'][i] / (1 - beta1 ** (epoch + 1)) for i in range(self.num_layers)]
        self.mom_grad_descent_attr['previous_delta_bias'] = [beta1 * self.mom_grad_descent_attr['previous_delta_bias'][i] + (1 - beta1) * delta_bias_complete[i] for i in range(self.num_layers)]
        delta_bias_bias_correction = [self.mom_grad_descent_attr['previous_delta_bias'][i] / (1 - beta1 ** (epoch + 1)) for i in range(self.num_layers)]    
        
        # Calculating the RMSProp based gradients that reduce the learning rate with each iteration 
        self.mom_grad_descent_attr['previous_learnrate_delta_weight'] = [beta2 * self.mom_grad_descent_attr['previous_learnrate_delta_weight'][i] + (1 - beta2) * np.square(delta_weight_complete[i]) for i in range(self.num_layers)]
        learnrate_delta_weight_bias_correction = [self.mom_grad_descent_attr['previous_learnrate_delta_weight'][i] / (1 - beta2 ** (epoch + 1)) for i in range(self.num_layers)]
        self.mom_grad_descent_attr['previous_learnrate_delta_bias'] = [beta2 * self.mom_grad_descent_attr['previous_learnrate_delta_bias'][i] + (1 - beta2) * np.square(delta_bias_complete[i]) for i in range(self.num_layers)]
        learnrate_delta_bias_bias_correction = [self.mom_grad_descent_attr['previous_learnrate_delta_bias'][i] / (1 - beta2 ** (epoch + 1)) for i in range(self.num_layers)]

        self.main_model['weights'] = [self.main_model['weights'][i] - learning_rate * delta_weight_bias_correction[i] / (np.sqrt(learnrate_delta_weight_bias_correction[i]) + epsilon) for i in range(self.num_layers)]
        self.main_model['biases'] = [self.main_model['biases'][i] - learning_rate * delta_bias_bias_correction[i] / (np.sqrt(learnrate_delta_bias_bias_correction[i]) + epsilon) for i in range(self.num_layers)]

    def _nadam_gradient_descent(self, train_data,  learning_rate, weight_decay, epoch, loss, beta1=0.9, beta2=0.999, epsilon=1e-08, **kwargs):
        """ The Nadam Gradient Descent optimizer is an improvement over the Adam Gradient Descent optimizer.
        It uses the Nesterov Accelerated Gradient (NAG) Descent along with the Adam Gradient Descent. 
        The formulas were a bit complex, I got confused with the implementation, so I referred the code from below github link to implement it.
        """
        # reference https://github.com/TannerGilbert/Machine-Learning-Explained/blob/master/Optimizers/nadam/code/nadam.py
        delta_weight_complete, delta_bias_complete = self._gradient_batch_wise_data(train_data, weight_decay=weight_decay, loss=loss)

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


################ Gradient Descent Optimizers END ##############
###############################################################