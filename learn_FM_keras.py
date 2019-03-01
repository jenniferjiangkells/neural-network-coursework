from __future__ import print_function
import numpy as np
import csv
import pickle

import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers.normalization import BatchNormalization
from keras.wrappers.scikit_learn import KerasRegressor
from keras.optimizers import SGD
from keras.constraints import maxnorm

from sklearn.model_selection import RandomizedSearchCV

from nn_lib import Preprocessor

from illustrate import illustrate_results_FM
from sklearn.metrics import mean_squared_error


def main(_neurons, _activationFunctionHidden, _activationFunctionOutput, _lossFunction, _batchSize, _learningRate,
         _numberOfEpochs, _writeToCSV=False, _hyperparameterTuning=False):

    dataset = np.loadtxt("FM_dataset.dat")

    #######################################################################
    #                       ** START OF YOUR CODE **
    #######################################################################

    input_dim = 3 # CONSTANT: Stated in specification

    #shuffle the data
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
    x_prep_input = Preprocessor(x_train)
    y_prep_input = Preprocessor(y_train)

    x_train_pre = x_prep_input.apply(x_train)
    y_train_pre = y_prep_input.apply(y_train)

    x_val_pre = x_prep_input.apply(x_val)
    y_val_pre = y_prep_input.apply(y_val)


    # fix random seed for reproducibility
    seed = 7
    np.random.seed(seed)



    if _hyperparameterTuning == True:

        #create model
        model = KerasRegressor(build_fn=create_model,
                                nb_epoch=_numberOfEpochs,
                                batch_size=_batchSize
                                )

        # Use scikit-learn to grid search
        epochs = [1] #10, 100, 250, 500, 1000?
        learn_rate = [1e-1, 1e-3, 1e-6]
        neurons = [1, 5, 15]
        hidden_layers = [3, 4, 5]
        #activation =  ['relu', 'sigmoid'] #tanh

        #optimizer = [ 'SGD', 'RMSprop', 'Adam']
        #dropout_rate = [0.0, 0.5, 0.9]

        param_grid = dict(epochs=epochs,
                            batch_size=batch_size,
                            learn_rate=learn_rate,
                            neurons=neurons,
                            hidden_layers=hidden_layers)

        #perform grid search with 10-fold cross validation
        grid = RandomizedSearchCV(estimator=model,
                            param_distributions=param_grid,
                            n_jobs=-1,
                            cv=10)

        grid_result = grid.fit(x_train_pre, y_train_pre)


        #summarize results of hyperparameter search
        print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
        means = grid_result.cv_results_['mean_test_score']
        stds = grid_result.cv_results_['std_test_score']
        params = grid_result.cv_results_['params']
        for mean, stdev, param in zip(means, stds, params):
            print ("%f (%f) with: %r" % (mean, stdev, param))


        #extract the best model
        best_model = grid.best_estimator_.model

        #Evaluate the best model
        preds = best_model.predict(x_val_pre)
        targets = y_val_pre
        mse = evaluate_architecture(targets, preds)
        print("Mean squared error of best model:", mse)

        #save the best model
        filename = 'trained_FM.pickle'
        pickle.dump(best_model, open(filename, 'wb'))

    else:

        model = create_model()
        history = model.fit(x_train_pre, y_train_pre,
                    batch_size=_batchSize,
                    epochs=numberOfEpochs,
                    verbose=1,
                    validation_data=(x_val_pre, y_val_pre))

        #model.fit(x_train_pre,y_train_pre)
        score = model.evaluate(x_val_pre, y_val_pre, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])

    #predict hidden dataset using best model
    predictions = predict_hidden(dataset)
    print(predictions)

    #######################################################################
    #                       ** END OF YOUR CODE **
    #######################################################################
    # illustrate_results_FM(net, prep)

#create keras model
def create_model(neurons=4, learn_rate=0.01, activation='relu', hidden_layers=1):
    # default values
    input_dim = 3  # CONSTANT: Stated in specification
    output_dim = 3
    # create model
    model = Sequential()
    #add input layer with batch normalization
    model.add(Dense(input_dim, input_dim=input_dim))
    model.add(BatchNormalization())
    model.add(Activation(activation))
    #add hidden layers
    for i in range(hidden_layers):
        model.add(Dense(neurons))
        model.add(BatchNormalization())
        model.add(Activation(activation))

    #add output layer
    model.add(Dense(output_dim))
    model.add(BatchNormalization())

    #compile model
    model.compile(loss='mean_squared_error',
                optimizer='adam',
                metrics=['accuracy'])

    model.summary()

    return model


def predict_hidden(dataset):

    input_dim = 3

    # Pre-process data
    x_data = dataset[:, :input_dim]
    y_data = dataset[:, input_dim:]
    prep_input = Preprocessor(x_data)
    x_data_pre = prep_input.apply(x_data)

    # Load the network
    filename = open("trained_FM.pickle", 'rb')
    model = pickle.load(filename)
    filename.close()

    # Generate the output
    preds = model.predict(x_data_pre) #use keras to make prediction


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
    numberOfEpochs = 50

    # Call the main function to train and evaluate the neural network
    main(neurons, activationFunctions, activationOutput, lossFunction, batchSize, learningRate, numberOfEpochs)
