import numpy as np
from sklearn.model_selection import GridSearchCV

from nn_lib import (
    MultiLayerNetwork,
    Trainer,
    Preprocessor,
    save_network,
    load_network,
)

from illustrate import illustrate_results_ROI
import matplotlib.pyplot as plt

# Global variables to stores values that will be used for plotting
xValues = []
yValues = [[], [], [], [], []]      # label1(f1), label2(f1), label3(f1), label4(f1), accuracy

# Main function that calls other functions to train and evaluate the neural network
def main(_neurons, _activationFunctionHidden, _activationFunctionOutput, _lossFunction, _batchSize, _learningRate, _numberOfEpochs, _writeToCSV = False):
    dataset = np.loadtxt("ROI_dataset.dat")
    #######################################################################
    #                       ** START OF YOUR CODE **
    #######################################################################
    # Setup hyperparameters and neural network
    input_dim = 3       # CONSTANT: Stated in specification
    neurons = _neurons
    activations = _activationFunctionHidden
    net = MultiLayerNetwork(input_dim, neurons, activations)

    np.random.shuffle(dataset)

    # Separate data columns into x (input features) and y (output)
    x = dataset[:, :input_dim]
    y = dataset[:, input_dim:]

    split_idx = int(0.8 * len(x))

    # Split data by rows into a training set and a validation set. We then augment the training data into the desired proportions
    # Use this for original dataset (training)
    # x_train = x[:split_idx]
    # y_train = y[:split_idx]

    # Use this for augmented dataset (training)
    augmentedTrainingData = augment_data_oversample(dataset[:split_idx, :], input_dim, label1=0.25, label2=0.25, label3=0.25, label4=0.25)
    x_train = augmentedTrainingData[:, :input_dim]
    y_train = augmentedTrainingData[:, input_dim:]

    # Validation dataset
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
    accuracy, confusionMatrix, labelDict = evaluate_architecture(targets, preds)

    # Optional: Print results
    print_results(confusionMatrix, labelDict, accuracy)

    # Optional: Append x and y values, to be plotted at the end
    global xValues, yValues
    xValues.append(neurons[0])
    for i in range(len(labelDict)):
        key = "label" + str(i + 1)
        metric = "f1"
        yValues[i].append(labelDict[key][metric])
    yValues[len(yValues) - 1].append(accuracy)

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

    print("best_parameters:", best_parameters)


    predict_hidden(dataset)

    #######################################################################
    #                       ** END OF YOUR CODE **
    #######################################################################
    # illustrate_results_ROI(network, prep)

def predict_hidden(dataset)
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

     _batchSize = best_parameters[0]
     _learningRate = best_parameters[5]
     _numberOfEpochs = best_parameters[6]
    with open('ROI_results.csv','a') as file:
            # No. of hidden layers, no. of neurons per hidden layer, activation in hidden layer, activation in output layer,
            # batch size, learning rate, number of epochs, accuracy, confusionMatrix, labelDict
        csvList = [len(neurons) - 1, neurons[0], activations[0], _activationFunctionOutput, _batchSize,
        _learningRate, _numberOfEpochs]
        csvRow = str(csvList).strip("[]")
        csvRow += "\n"
        file.write(csvRow)

    load_network(ROI_results.csv)

    y_pred = grid_search.predict(x_val_pre)

    return y_pred

# Augments the data into the desired proportion. The size of the new dataset will be larger than the input dataset
# Each and every label will be oversampled relative to the size as that of the label with the largest sample size
# No data will be shrunk/discarded
def augment_data_oversample(dataset, inputDim, label1=0.25, label2=0.25, label3=0.25, label4=0.25):
    # Calculate the relative proportions based on the input arguments
    label1 /= (label1 + label2 + label3 + label4)
    label2 /= (label1 + label2 + label3 + label4)
    label3 /= (label1 + label2 + label3 + label4)
    label4 /= (label1 + label2 + label3 + label4)
    listOfLabelProportions = [label1, label2, label3, label4]

    # Get the counts of each label in the input dataset and store as a dictionary (key = index, value = count)
    indices = np.argmax(dataset[:, inputDim:], axis=1)
    unique, counts = np.unique(indices, return_counts=True)
    countsDict = dict(zip(unique, counts))

    # Segregate the dataset according to the label
    numOfRows = dataset.shape[0]
    numOfColumns = dataset.shape[1]
    labelData = np.empty([0, numOfColumns])
    listOfLabelData = [labelData, labelData, labelData, labelData]      # Index 0 = label1 data, index 1 = label2 data, etc.
    for i in range(numOfRows):
        labelIndex = np.argmax(dataset[i, inputDim:])       # Get the index with the maximum value out of indices 0 to 3
        listOfLabelData[labelIndex] = np.append(listOfLabelData[labelIndex], [dataset[i, :]], axis=0)

    # Augment the dataset
    minNumberOfDataKey = max(countsDict, key=lambda i: countsDict[i])   # Get the label with the most data
    minNumberOfDataValue = countsDict[minNumberOfDataKey]
    newDataset = np.empty([0, numOfColumns])
    for i in range(len(listOfLabelData)):
        numOfDataNeeded = int((listOfLabelProportions[i] / listOfLabelProportions[minNumberOfDataKey]) * countsDict[minNumberOfDataKey])
        if numOfDataNeeded <= listOfLabelData[i].shape[0]:
            newDataset = np.append(newDataset, listOfLabelData[i][:numOfDataNeeded, :], axis=0)
        else:
            numOfDuplicationsNeeded = int(numOfDataNeeded / listOfLabelData[i].shape[0])
            numOfRemaindersNeeded = int(numOfDataNeeded % listOfLabelData[i].shape[0])
            for j in range(numOfDuplicationsNeeded):
                newDataset = np.append(newDataset, listOfLabelData[i][:, :], axis=0)
            newDataset = np.append(newDataset, listOfLabelData[i][:numOfRemaindersNeeded, :], axis=0)

    return newDataset

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
        recall = calculate_recall(truePositive, falseNegative)
        precision = calculate_precision(truePositive, falsePositive)
        f1 = calculate_f1(recall, precision)
        totalErrors += falsePositive

        key = "label" + str(index + 1)
        labelDict[key] = {"recall": recall, "precision": precision, "f1": f1}
        index += 1

    accuracy = calculate_classification_rate(numOfRows, totalErrors)

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

# Metric: recall = true pos / (true pos + false neg)
def calculate_recall(truePositive, falseNegative):
    if truePositive + falseNegative == 0:
        return 0
    return truePositive / (truePositive + falseNegative)

# Metric: precision = true pos / (true pos + false pos)
def calculate_precision(truePositive, falsePositive):
    if truePositive + falsePositive == 0:
        return 0
    return truePositive / (truePositive + falsePositive)

# Metric: F1 = 2 * (prec * rec) / (prec + rec)
def calculate_f1(precision, recall):
    if precision + recall == 0:
        return 0
    return 2 * (precision * recall) / (precision + recall)

# Metric: classification rate = 1 - classification error
def calculate_classification_rate(numOfRows, totalErrors):
    if numOfRows == 0:
        return 0
    return (numOfRows - totalErrors) / numOfRows

# Prints the metrics
def print_results(confusionMatrix, labelDict, accuracy):
    print(confusionMatrix)
    for i in range(len(labelDict)):
        key = "label" + str(i + 1)
        print(key, labelDict[key])
    print("Accuracy: ", accuracy)

# Plot a line graph of y against x
def plot_data(x, y):
    # Define the box area for the main plot
    ax = plt.subplot(111)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    # Set the data we want to be plotted
    for i in range(len(y)):
        labelName = "Label " + str(i + 1) + " (f1)"
        if i == len(y) - 1:
            plt.plot(x, y[i], marker="x", label="Accuracy")
        else:
            plt.plot(x, y[i], marker="x", label=labelName)

    # Set axes scales
    plt.ylim(0.0, 1.0)

    # Set label and title names
    xLabel = "Number of neurons per hidden layer (X hidden layers)"
    yLabel = "F1 + Accuracy"
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.title(yLabel + " vs " + xLabel)
    plt.grid(True)

    # Set legend to be outside of the plot
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.show()


if __name__ == "__main__":
    #for iteratedValue in range(10, 51, 5):
        # Setup for the hyperparameters for main()
        neurons = []
        activationFunctions = []
        outputDimension = 4

        # Modify any of the following hyperparameters
        numOfHiddenLayers = 3              # Does not count input/output layer
        numOfNeuronsPerHiddenLayer = 5     # Configures all hidden layers to have the same number of neurons
        activationHidden = "relu"          # Does not apply for input/output layer
        activationOutput = "sigmoid"
        lossFunction = "mse"
        batchSize = 64
        learningRate = 1e-3
        numberOfEpochs = 10
        
        # Optional: Write results to csv
        writeToCSV = True


        # Call the main function to train and evaluate the neural network
        main(neurons, activationFunctions, activationOutput, lossFunction, batchSize, learningRate, numberOfEpochs, writeToCSV)
