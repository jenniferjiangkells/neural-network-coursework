from __future__ import print_function
import numpy as np
import csv
import pickle

import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers.normalization import BatchNormalization
from keras.wrappers.scikit_learn import KerasClassifier
from keras.optimizers import SGD
from keras.constraints import maxnorm

from sklearn.model_selection import RandomizedSearchCV

from nn_lib import Preprocessor

from illustrate import illustrate_results_ROI
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# Global variables to stores values that will be used for plotting
xValues = []
yValues = [[], [], [], [], []]      # label1(f1), label2(f1), label3(f1), label4(f1), accuracy

# Main function that calls other functions to train and evaluate the neural network
def main(_neurons, _activationFunctionHidden, _activationFunctionOutput, _lossFunction, _batchSize, _learningRate,
         _numberOfEpochs, _writeToCSV=False, _Train=True):
    dataset = np.loadtxt("ROI_dataset.dat")
    #######################################################################
    #                       ** START OF YOUR CODE **
    #######################################################################
    # Setup hyperparameters and neural network
    if _Train == True:
       input_dim = 3       # CONSTANT: Stated in specification


       np.random.shuffle(dataset)
    #numOfRows = int(0.8*dataset.shape[0])
    #output = predict_hidden(dataset[:numOfRows, :])
    #print(output)
    # Separate data columns into x (input features) and y (output)
       x = dataset[:, :input_dim]
       y = dataset[:, input_dim:]

       split_idx = int(0.8 * len(x))

    # Split data by rows into a training set and a validation set. We then augment the training data into the desired proportions
       x_train = x[:split_idx]
       y_train = y[:split_idx]
    # Validation dataset
       x_val = x[split_idx:]
       y_val = y[split_idx:]

    # Apply preprocessing to the data
       x_prep_input = Preprocessor(x_train)
       #y_prep_input = Preprocessor(y_train)

       x_train_pre = x_prep_input.apply(x_train)
       #y_train_pre = y_prep_input.apply(y_train)
       y_train_pre = y_train

       x_val_pre = x_prep_input.apply(x_val)
       #y_val_pre = y_prep_input.apply(y_val)
       y_val_pre = y_val

       seed = 7
       np.random.seed(seed)

    #create model
       model = KerasClassifier(build_fn=create_model,
                            nb_epoch=_numberOfEpochs,
                            batch_size=_batchSize)

    # Use scikit-learn to grid search - these are all possible paramaters, takes a long time so I only left in few values
       batch_size = [8, 16, 32] #32
       epochs = [10] #10, 100, 250, 500, 1000?
       learn_rate = [1e-1, 1e-3, 1e-6]
       neurons = [1, 5, 15]
       hidden_layers = [3, 4, 5]

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

       print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
       best_model = grid.best_estimator_.model



    # Evaluate the neural network
       preds = best_model.predict(x_val_pre)
       targets = y_val_pre
       accuracy, confusionMatrix, labelDict = evaluate_architecture(targets, preds)
    # Optional: Print results
       print(confusionMatrix)
       for i in range(len(labelDict)):
           key = "label" + str(i + 1)
           print(key, labelDict[key])
       print("Accuracy: ", accuracy)

    # Optional: Append x and y values, to be plotted at the end
       global xValues, yValues
       xValues.append(len(neurons) - 1)
       for i in range(len(labelDict)):
           key = "label" + str(i + 1)
           metric = "f1"
           yValues[i].append(labelDict[key][metric])
           yValues[len(yValues) - 1].append(accuracy)

       filename = 'trained_ROI.pickle'
       pickle.dump(best_model, open(filename, 'wb'))


       #predict hidden dataset using best model
       predictions = predict_hidden(dataset)
       print(predictions)

    #######################################################################
    #                       ** END OF YOUR CODE **
    #######################################################################
    # illustrate_results_ROI(network, prep)
def create_model(neurons=1, learn_rate=0.01, activation='relu', hidden_layers=1):
    # default values
    input_dim = 3  # CONSTANT: Stated in specification
    # create model
    model = Sequential()
    #add input layer with batch normalization
    model.add(Dense(3, input_dim=input_dim))
    model.add(BatchNormalization())
    model.add(Activation(activation))
    #add hidden layers
    for i in range(hidden_layers):
        model.add(Dense(4))
        model.add(BatchNormalization())
        model.add(Activation(activation))

    #add output layer
    model.add(Dense(4))
    model.add(BatchNormalization())

    #compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model
# Augments the data into the desired proportion. The size of the new dataset will be the same as the input dataset
def predict_hidden(dataset):
    input_dim = 3
    x_data = dataset[:, :input_dim]
    y_data = dataset[:, input_dim:]
    prep_input = Preprocessor(x_data)
    x_data_pre = prep_input.apply(x_data)

    # Load the network
    filename = open("trained_ROI.pickle", 'rb')
    model = pickle.load(filename)
    filename.close()

    # Generate the output
    pred = model.predict(x_data_pre) #use keras to make prediction
    oneHotEncoding = one_hot_encode(pred, pred.shape[1])
    return oneHotEncoding


def one_hot_encode(output, numOfPossibleLabels):
    indices = extract_indices(output)
    encodedOutput = np.empty([0, numOfPossibleLabels])

    for i in range(indices.shape[0]):
        newRow = np.zeros((1, numOfPossibleLabels))
        newRow[:, indices[i]] = 1
        encodedOutput = np.append(encodedOutput, newRow, axis=0)

    return encodedOutput
# First create the confusion matrix (predicted x expected)
# Then, evaluate the architecture using accuracy, precision, recall and F1 score
def evaluate_architecture(y_true, y_pred):
    # Generate and populate the confusion matrix
    confusionMatrix = populate_confusion_matrix(y_true, y_pred)

    # Stores data on recall, precision and f1 for each label
    labelDict = dict.fromkeys({"label1", "label2", "label3", "label4"})

    # Compute and store the metrics
    index = 0
    totalErrors = 0
    numOfRows = y_true.shape[0]
    for i in range(len(labelDict)):
        truePositive, falsePositive, falseNegative = calculate_metrics(confusionMatrix, index)
        if truePositive + falseNegative == 0:
            recall=0
        recall= truePositive / (truePositive + falseNegative)

        if truePositive + falsePositive == 0:
            precision = 0
        precision = truePositive / (truePositive + falsePositive)

        if precision + recall == 0:
            f1 = 0
        f1 = 2 * (precision * recall) / (precision + recall)

        totalErrors += falsePositive

        key = "label" + str(index + 1)
        labelDict[key] = {"recall": recall, "precision": precision, "f1": f1}
        index += 1

        if numOfRows == 0:
           accuracy = 0
        accuracy = (numOfRows - totalErrors) / numOfRows

    # Return metrics
    return accuracy, confusionMatrix, labelDict

# Populates the confusion matrix (predicted x expected) based on y_true and y_pred
def populate_confusion_matrix(y_true, y_pred):
    # Create an empty confusion matrix filled with 0s
    numOfRows = y_pred.shape[1]
    numOfColumns = y_true.shape[1]
    confusionMatrix = create_confusion_matrix(numOfRows, numOfColumns)

    # Start populating the confusion matrix
    row = extract_indices(y_pred)
    col = extract_indices(y_true)
    for i in range(y_true.shape[0]):
        confusionMatrix[row[i], col[i]] += 1

    return confusionMatrix

# Create confusion matrix (predicted x expected) with all 0s
def create_confusion_matrix(numOfRows, numOfColumns):
    matrix = np.zeros(shape = (numOfRows, numOfColumns))

    return matrix

# Given a matrix of values, returns as an array the indices of the maximum value for each row
# E.g. [0, 1, 0, 0] returns 1
def extract_indices(data):
    indices = np.argmax(data, axis=1)

    return indices

# Metric: true positive, false positive, false negative
def calculate_metrics(matrix, index):
    numOfRows = matrix.shape[0]
    truePositive = matrix[index, index]
    falsePositive = 0
    falseNegative = 0

    # Calculate false positive and false negative
    for currentIndex in range(numOfRows):
        if currentIndex == index:
            continue
        falsePositive += matrix[index, currentIndex]
        falseNegative += matrix[currentIndex, index]

    return truePositive, falsePositive, falseNegative




if __name__ == "__main__":
    neurons = []
    activationFunctions = []
    outputDimension = 4

    # Modify any of the following hyperparameters
    numOfHiddenLayers = 4              # Does not count input/output layer
    numOfNeuronsPerHiddenLayer = 35      # Configures all hidden layers to have the same number of neurons
    activationHidden = "relu"          # Does not apply for input/output layer
    activationOutput = "sigmoid"
    lossFunction = "mse"
    batchSize = 64
    learningRate = 1e-3
    numberOfEpochs = 1000



    # Call the main function to train and evaluate the neural network
main(neurons, activationFunctions, activationOutput, lossFunction, batchSize, learningRate, numberOfEpochs)
