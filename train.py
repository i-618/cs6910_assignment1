import argparse
import numpy as np
import wandb
from NeuralNetwork import NeuralNetwork



parser = argparse.ArgumentParser(description='Neural Network Training Configuration')


parser.add_argument('-wp', '--wandb_project', type=str, default='myprojectname', help='Project name used to track experiments in Weights & Biases dashboard')
parser.add_argument('-we', '--wandb_entity', type=str, default='myname', help='Wandb Entity used to track experiments in the Weights & Biases dashboard.')
parser.add_argument('-sid', '--wandb_sweepid', type=str, default=None, help='Wandb Sweep Id to log in sweep runs the Weights & Biases dashboard.')
parser.add_argument('-d', '--dataset', type=str, default='fashion_mnist', choices=["mnist", "fashion_mnist"], help='Dataset choices: ["mnist", "fashion_mnist"]')
parser.add_argument('-e', '--epochs', type=int, default=1, help='Number of epochs to train neural network.')
parser.add_argument('-b', '--batch_size', type=int, default=4, help='Batch size used to train neural network.')
parser.add_argument('-l', '--loss', type=str, default='cross_entropy', choices=["mean_squared_error", "cross_entropy"], help='Loss function choices: ["mean_squared_error", "cross_entropy"]')
parser.add_argument('-o', '--optimizer', type=str, default='stochastic', choices=["stochastic", "momentum", "nag", "rmsprop", "adam", "nadam"], help='Optimizer choices: ["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"]')
parser.add_argument('-lr', '--learning_rate', type=float, default=0.1, help='Learning rate used to optimize model parameters')
parser.add_argument('-m', '--momentum', type=float, default=0.5, help='Momentum used by momentum and nag optimizers.')
parser.add_argument('-beta', '--beta', type=float, default=0.5, help='Beta used by rmsprop optimizer')
parser.add_argument('-beta1', '--beta1', type=float, default=0.5, help='Beta1 used by adam and nadam optimizers.')
parser.add_argument('-beta2', '--beta2', type=float, default=0.5, help='Beta2 used by adam and nadam optimizers.')
parser.add_argument('-eps', '--epsilon', type=float, default=0.000001, help='Epsilon used by optimizers.')
parser.add_argument('-w_d', '--weight_decay', type=float, default=0.0, help='Weight decay used by optimizers.')
parser.add_argument('-w_i', '--weight_init', type=str, default='random', choices=["random", "xavier"], help='Weight initialization choices: ["random", "xavier"]')
parser.add_argument('-nhl', '--num_layers', type=int, default=1, help='Number of hidden layers used in feedforward neural network.')
parser.add_argument('-sz', '--hidden_size', type=int, default=4, help='Number of hidden neurons in a feedforward layer.')
parser.add_argument('-a', '--activation', type=str, default='sigmoid', choices=["identity", "sigmoid", "tanh", "ReLU"], help='Activation function choices: ["identity", "sigmoid", "tanh", "ReLU"]')


args = parser.parse_args()

train_data, test_data = None, None
if args.dataset == 'mnist':
    from keras.datasets import mnist
    train_data, test_data = mnist.load_data()
elif args.dataset == 'fashion_mnist':
    from keras.datasets import fashion_mnist
    train_data, test_data = fashion_mnist.load_data()



X_train, y_train = train_data

X_train_flattened = X_train.reshape(-1, 784, 1)/255.0
one_hot_label = np.zeros([y_train.shape[0], len(np.unique(y_train)), 1], dtype=int)
for index, item in enumerate(y_train):
  one_hot_label[index, item] = [1]


train_records_count = int(len(X_train_flattened)*0.9)
train_data={'inputs':X_train_flattened[:train_records_count], 'labels':one_hot_label[:train_records_count]}
val_data={'inputs':X_train_flattened[train_records_count:], 'labels':one_hot_label[train_records_count:]}




def train(config=None):
  # Initialize a new wandb run
    with wandb.init(config=config) as run:

        # If called by wandb.agent, as below,
        # this config will be set by Sweep Controller
        config = wandb.config
        
        run_name = str(config).replace("': '", ' ').replace("'", '')
        print(run_name)
        run.name = run_name
        layers = [{'num_neurons': config.hidden_size, 'activation': config.activation}] * config.num_layers
        nn = NeuralNetwork(input_dim=X_train_flattened.shape[1], output_dim=one_hot_label.shape[1], nn_archtre=layers, 
                   last_layer_activation='softmax', weight_initializer=config.weight_init)
        
        

        nn.train(train_data=train_data, val_data=val_data, epochs=config.epochs, learning_rate=config.learning_rate,
                 optimizer=config.optimizer, weight_decay=config.weight_decay, batch_size=config.batch_size, print_every_epoch=1)


        
        
wandb.agent(args.wandb_sweepid, project=args.wandb_project, function=train)