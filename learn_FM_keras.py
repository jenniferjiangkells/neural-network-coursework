from __future__ import print_function
import numpy as np
import csv

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.wrappers.scikit_learn import KerasRegressor
from keras.optimizers import SGD
from keras.constraints import maxnorm

from sklearn.model_selection import GridSearchCV

from nn_lib import Preprocessor
#    MultiLayerNetwork,
#    Trainer,
#    save_network,
#    load_network,
#)
from illustrate import illustrate_results_FM
from sklearn.metrics import mean_squared_error


def main(_neurons, _activationFunctionHidden, _activationFunctionOutput, _lossFunction, _batchSize, _learningRate,
         _numberOfEpochs, _writeToCSV=False):

    dataset = np.loadtxt("FM_dataset.dat")

    #######################################################################
    #                       ** START OF YOUR CODE **
    #######################################################################
    # Setup hyperparameters and neural network
    input_dim = 3  # CONSTANT: Stated in specification
    neurons = _neurons
    activations = _activationFunctionHidden
    #net = MultiLayerNetwork(input_dim, neurons, activations)
    np.random.shuffle(dataset)

    # Separate data columns into x (input features) and y (output)
    x = dataset[:, :input_dim]
    y = dataset[:, input_dim:]

    split_idx = int(0.8 * len(x))

    # Split data by rows into a training set and a validation set
    x_train = x[:split_idx]
    y_train = y[:split_idx]
    x_val = x[split_idx:]
    y_val = y[split_idx:]

    # Apply preprocessing to the data
    prep_input = Preprocessor(x_train)
    x_train_pre = prep_input.apply(x_train)
    x_val_pre = prep_input.apply(x_val)

    # fix random seed for reproducibility
    seed = 7
    np.random.seed(seed)

    #create model
    model = KerasRegressor(build_fn=create_model, nb_epoch=_numberOfEpochs,
                            batch_size=_batchSize)

    # Use scikit-learn to grid search - these are all possible paramaters, takes a long time so I only left in few values
    batch_size = [8, 16] #32
    activation =  ['relu', 'tanh', 'sigmoid']
    epochs = [1, 10] #100, 250, 500, 1000?
    learn_rate = [1e-1, 1e-3, 1e-6]
    dropout_rate = [0.0, 0.5, 0.9]
    neurons = [1, 5, 10, 15]
    optimizer = [ 'SGD', 'RMSprop', 'Adam']

    param_grid = dict(epochs=epochs, batch_size=batch_size, learn_rate=learn_rate,
                        activation=activation, neurons=neurons)

    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=10)

    grid_result = grid.fit(x_train_pre, y_train)

    # summarize results

    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print ("%f (%f) with: %r" % (mean, stdev, param))


    ##########this is not working yet, find out how to predict with keras models
    # Evaluate the neural network
    #preds = grid.predict(x_val_pre)
    #targets = y_val
    #mse = evaluate_architecture(targets, preds)
    print("Mean squared error:", mse)

    predict_hidden(dataset)

    #######################################################################
    #                       ** END OF YOUR CODE **
    #######################################################################
    # illustrate_results_FM(net, prep)

#create model for KerasRegressor - right now the network is just input output layer, figure out
#how to modify number of hidden layers maybe
def create_model(neurons=1, learn_rate=0.01, activation='relu'):
    # default values
    dropout_rate=0.0 # or 0.2
    optimizer='adam' # or SGD
    input_dim = 3

    # create model
    model = Sequential()
    model.add(Dense(3, input_dim=input_dim, activation=activation))
    model.add(Dense(3))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam')

    return model

#how to load best parameter in here?
def predict_hidden(dataset):

    filePath = "trained_ROI.pickle"
    input_dim = 3

    # Pre-process data
    xData = inputData[:, :input_dim]
    yData = inputData[:, input_dim:]
    prep_input = Preprocessor(xData)
    x_data_pre = prep_input.apply(xData)

    # Load the network
    #net = load_network(filePath)

    # Generate the output
    #preds = net(x_data_pre) #use keras to make prediction


    return preds


# Evaluate the arhcitecture using MSE
def evaluate_architecture(y_true, y_pred):
    return mean_squared_error(y_true, y_pred)


if __name__ == "__main__":
    # Setup for the hyperparameters for main()
    neurons = []
    activationFunctions = []
    outputDimension = 3

    # Modify any of the following hyperparameters
    numOfHiddenLayers = 3  # Does not count input/output layer
    numOfNeuronsPerHiddenLayer = 5  # Configures all hidden layers to have the same number of neurons
    activationHidden = "relu"  # Does not apply for input/output layer
    activationOutput = "identity"
    lossFunction = "mse"
    batchSize = 64
    learningRate = 1e-5
    numberOfEpochs = 1000

    # Call the main function to train and evaluate the neural network
    main(neurons, activationFunctions, activationOutput, lossFunction, batchSize, learningRate, numberOfEpochs)
