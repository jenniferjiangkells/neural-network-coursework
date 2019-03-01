# Group 69 Neural Networks CBC

This repository contains nn_lib, our implementation of a mini-neural network library, which will load the dataset from iris.dat and train a neural network based on the default parameters loaded in main,
and learn_FM, learn_ROI, which simulates a robotic arm and makes predictions on the coordinate or region of the robotic arm.
learn_FM and learn_ROI will load the default datasets (FM_dataset and ROI dataset), train a neural network based on a default architecture, and display evaluation matrices.
It will then predict on the "hidden dataset" using a model that has been hyperparameter-tuned (trained_FM and trained_ROI pickle files) and output the predicted dataset of NxM, where
N is the number of observations in the dataset and M is the output dimension.

## Prerequisites

Our code depends on the latest versions of numpy, keras, and scikit_learn, and pickle (for loading pre-trained model) and requires the installation of these modules,
which can be installed with the commands given below (per the specification).

```bash

export PYTHONUSERBASE=/vol/bitbucket/nuric/pypi

```

## Running the code

Our code runs on Python 3. To run, simply type

```bash

python3 nn_lib
python3 learn_FM.py
python3 learn_ROI.py

```
