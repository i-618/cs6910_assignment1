# cs6910_assignment1
---
## Assignment Solved by Likhith Kumara (NS24Z066) for the course CS6910 Fundamentals of Deep Learning
---
Three Main files for **Assignment Grading**:
- **train.py** - Accepts the command args for training 
- **NeuralNetwork.py** - Has the actual code for the Neural Network
- **FDL_CS6910_Assigment_1.ipynb** - Has the code for each of the questions, it used NeuralNetwork.py for solving the questions.

The code is very simple and has inline comments and documentations to help understand the workings and usage of the functions. 

The whole code for Neural Network is written in NeuralNetwork.py, it is a single class file which has all the functions and methods required to train and test the neural network. train.py is a wrapper script which uses the NeuralNetwork class to train the network and upload the reports to wandb. 

The code is written in a modular way so as to make it easy to implement any new feature like activation functions, loss functions, optimizers, etc.


For Initializing the neural network:
```python
from NeuralNetwork import NeuralNetwork
layers = [
          {'num_neurons': 50, 'activation': 'relu'},
          {'num_neurons': 50, 'activation': 'tanh'},
          {'num_neurons': 50, 'activation': 'sigmoid'},
          ]
nn = NeuralNetwork(input_dim=x_train.shape[1], output_dim=y_train.shape[1], nn_archtre=layers, last_layer_activation='softmax', weight_initializer='xavier')

```

For Getting predictions via forward propagation:
```python
y_pred = nn.forward_propagation(x_test)
```
For Training the neural network:
```python
nn.train(train_data=train_data, val_data=val_data, epochs=10, learning_rate=0.001,
                 optimizer='adam', weight_decay=0, batch_size=64, print_every_epoch=1)
```