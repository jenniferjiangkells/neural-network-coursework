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

from sklearn.model_selection import GridSearchCV

from nn_lib import Preprocessor

from illustrate import illustrate_results_FM
from sklearn.metrics import mean_squared_error


def main(_neurons, _activationFunctionHidden, _activationFunctionOutput, _lossFunction, _batchSize, _learningRate,
         _numberOfEpochs, _writeToCSV=False, _hyperparameterTuning=True):

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
        batch_size = [32]
        epochs = [100,250,500,1000] #10, 100, 250, 500, 1000?
        learn_rate = [1e-3]
        neurons = [5]
        hidden_layers = [3]
        #activation =  ['relu', 'sigmoid'] #tanh

        #optimizer = [ 'SGD', 'RMSprop', 'Adam']
        #dropout_rate = [0.0, 0.5, 0.9]

        param_grid = dict(epochs=epochs,
                            batch_size=batch_size,
                            learn_rate=learn_rate,
                            neurons=neurons,
                            hidden_layers=hidden_layers)

        #perform grid search with 10-fold cross validation
        grid = GridSearchCV(estimator=model,
                            param_grid=param_grid,
                            n_jobs=-1,
                            cv=5)

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

        GridSearch_table_plot(grid, batch_size,
                                  num_results=15,
                                  negative=True,
                                  graph=True,
                                  display_all_params=True)

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

def GridSearch_table_plot(grid_clf, param_name,
                          num_results=15,
                          negative=True,
                          graph=True,
                          display_all_params=True):

    '''Display grid search results

    Arguments
    ---------

    grid_clf           the estimator resulting from a grid search
                       for example: grid_clf = GridSearchCV( ...

    param_name         a string with the name of the parameter being tested

    num_results        an integer indicating the number of results to display
                       Default: 15

    negative           boolean: should the sign of the score be reversed?
                       scoring = 'neg_log_loss', for instance
                       Default: True

    graph              boolean: should a graph be produced?
                       non-numeric parameters (True/False, None) don't graph well
                       Default: True

    display_all_params boolean: should we print out all of the parameters, not just the ones searched for?
                       Default: True

    Usage
    -----

    GridSearch_table_plot(grid_clf, "min_samples_leaf")

                          '''
    from matplotlib      import pyplot as plt
    from IPython.display import display
    import pandas as pd

    clf = grid_clf.best_estimator_
    clf_params = grid_clf.best_params_
    if negative:
        clf_score = -grid_clf.best_score_
    else:
        clf_score = grid_clf.best_score_
    clf_stdev = grid_clf.cv_results_['std_test_score'][grid_clf.best_index_]
    cv_results = grid_clf.cv_results_

    print("best parameters: {}".format(clf_params))
    print("best score:      {:0.5f} (+/-{:0.5f})".format(clf_score, clf_stdev))
    if display_all_params:
        import pprint
        pprint.pprint(clf.get_params())

    # pick out the best results
    # =========================
    scores_df = pd.DataFrame(cv_results).sort_values(by='rank_test_score')

    best_row = scores_df.iloc[0, :]
    if negative:
        best_mean = -best_row['mean_test_score']
    else:
        best_mean = best_row['mean_test_score']
    best_stdev = best_row['std_test_score']
    best_param = best_row['param_' + param_name]

    # display the top 'num_results' results
    # =====================================
    display(pd.DataFrame(cv_results) \
            .sort_values(by='rank_test_score').head(num_results))

    # plot the results
    # ================
    scores_df = scores_df.sort_values(by='param_' + param_name)

    if negative:
        means = -scores_df['mean_test_score']
    else:
        means = scores_df['mean_test_score']
    stds = scores_df['std_test_score']
    params = scores_df['param_' + str(param_name[0])]

    # plot
    if graph:
        plt.figure(figsize=(8, 8))
        plt.errorbar(params, means, yerr=stds)

        plt.axhline(y=best_mean + best_stdev, color='red')
        plt.axhline(y=best_mean - best_stdev, color='red')
        plt.plot(best_param, best_mean, 'or')

        plt.title(param_name + " vs Score\nBest Score {:0.5f}".format(clf_score))
        plt.xlabel(param_name)
        plt.ylabel('Score')
        plt.show()
        plt.savefig('graph.png')


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
