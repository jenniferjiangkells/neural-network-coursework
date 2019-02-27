import numpy as np
import csv
from sklearn.model_selection import GridSearchCV

from nn_lib import (
    MultiLayerNetwork,
    Trainer,
    Preprocessor,
    save_network,
    load_network,
)
from illustrate import illustrate_results_FM
from sklearn.metrics import mean_squared_error


def main(_neurons, _activationFunctionHidden, _activationFunctionOutput, _lossFunction, _batchSize, _learningRate, _numberOfEpochs):   #, _writeToCSV=False
    dataset = np.loadtxt("FM_dataset.dat")
    #######################################################################
    #                       ** START OF YOUR CODE **
    #######################################################################
    # Setup hyperparameters and neural network
    input_dim = 3  # CONSTANT: Stated in specification
    neurons = _neurons
    activations = _activationFunctionHidden
    net = MultiLayerNetwork(input_dim, neurons, activations)

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

    trainer = Trainer(
        network=net,
        batch_size=_batchSize,
        nb_epoch=_numberOfEpochs,
        learning_rate=_learningRate,
        loss_fun=_lossFunction,
        shuffle_flag=True,
    )

    # Train the neural network
    trainer.train(x_train_pre, y_train)
    print("Train loss = ", trainer.eval_loss(x_train_pre, y_train))
    print("Validation loss = ", trainer.eval_loss(x_val_pre, y_val))

    # Evaluate the neural network
    preds = net(x_val_pre)
    targets = y_val
    mse = evaluate_architecture(targets, preds)
    print("Mean squared error:", mse)

    parameters = {'batchSize': [8, 16, 32],
                  'epochs': [100, 500],
                  'numOfHiddenLayer': [3, 4, 5],  # Does not count input/output layer
                  'numOfNeuronsPerHiddenLayer': [10, 51, 5],
                  # Configures all hidden layers to have the same number of neurons
                  'learningRate': [1e-1, 1e-3, 1e-6],
                  'numberOfEpochs': [5, 10, 20]}

    grid_search = GridSearchCV(estimator=net,
                               param_grid=parameters,
                               n_jobs=-1,
                               cv=10
                               )
    grid_search = grid_search.fit(x_train_pre, y_train)
    best_parameters = grid_search.best_params_

    print("best_parameters:", best_parameters)

    predict_hidden(dataset)

    #######################################################################
    #                       ** END OF YOUR CODE **
    #######################################################################
    # illustrate_results_FM(net, prep)

def predict_hidden(dataset):
    input_dim = 3
    np.random.shuffle(dataset)

    x = dataset[:, :input_dim]
    y = dataset[:, input_dim:]

    split_idx = int(0.8 * len(x))
    augmentedTrainingData = augment_data_oversample(dataset[:split_idx, :], input_dim, label1=0.25, label2=0.25, label3=0.25, label4=0.25)
    x_train = augmentedTrainingData[:, :input_dim]
    y_train = augmentedTrainingData[:, input_dim:]

    x_val = x[split_idx:]
    y_val = y[split_idx:]

    prep_input = Preprocessor(x_train)
    x_train_pre = prep_input.apply(x_train)
    x_val_pre = prep_input.apply(x_val)

    neurons = [16, 3]
    activations = ["relu", "identity"]
    net = MultiLayerNetwork(input_dim, neurons, activations)

    parameters = {'batchSize': [8, 16, 32],
               'epochs': [100, 500],
               'optimizer': ['adam', 'rmsprop'],
               'numOfHiddenLayer': [3, 4, 5],              # Does not count input/output layer
               'numOfNeuronsPerHiddenLayer': [10, 51,5],      # Configures all hidden layers to have the same number of neurons
               'learningRate': [1e-1, 1e-3, 1e-6],
               'numberOfEpochs': [5, 10, 20]}

    grid_search = GridSearchCV(estimator = net,
                            param_grid = parameters,
                            n_jobs=-1,
                            cv = 10
             )
    grid_search = grid_search.fit(x_train_pre, y_train)
    best_parameters = grid_search.best_params_

#    _batchSize = best_parameters[0]
#    _learningRate = best_parameters[5]
#    _numberOfEpochs = best_parameters[6]

#    with open('ROI_results.csv','a') as file:
            # No. of hidden layers, no. of neurons per hidden layer, activation in hidden layer, activation in output layer,
            # batch size, learning rate, number of epochs, accuracy, confusionMatrix, labelDict
#        csvList = [len(neurons) - 1, neurons[0], activations[0], _activationFunctionOutput, _batchSize,
#        _learningRate, _numberOfEpochs]
#        csvRow = str(csvList).strip("[]")
#        csvRow += "\n"
#        file.write(csvRow)

#    load_network(ROI_results.csv)

    y_pred = grid_search.predict(x_val_pre)

    return y_pred

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
