from collections import defaultdict

import numpy as np

import loglinear as ll
import random

import utils

STUDENT = {'name': 'YOUR NAME',
           'ID': 'YOUR ID NUMBER'}


def feats_to_vec(features):
    # YOUR CODE HERE.
    # Should return a numpy vector of features.
    length = len(features)
    vec = np.zeros(len(utils.vocab))
    for feature in features:
        if feature in utils.F2I:
            vec[utils.F2I[feature]] += 1
    vec = vec / length
    return vec


def accuracy_on_dataset(dataset, params):
    good = bad = 0.0
    for label, features in dataset:
        # YOUR CODE HERE
        # Compute the accuracy (a scalar) of the current parameters
        # on the dataset.
        # accuracy is (correct_predictions / all_predictions)
        x = feats_to_vec(features)
        y = utils.CLASS_TO_INDEX[label]
        if y == ll.predict(x, params):
            good += 1
        else:
            bad += 1
    return good / (good + bad)


def train_classifier(train_data, dev_data, num_iterations, learning_rate, params):
    """
    Create and train a classifier, and return the parameters.

    train_data: a list of (label, feature) pairs.
    dev_data  : a list of (label, feature) pairs.
    num_iterations: the maximal number of training iterations.
    learning_rate: the learning rate to use.
    params: list of parameters (initial values)
    """
    for I in range(num_iterations):
        cum_loss = 0.0  # total loss in this iteration.
        random.shuffle(train_data)
        for label, features in train_data:
            x = feats_to_vec(features)  # convert features to a vector.
            y = utils.CLASS_TO_INDEX[label]  # convert the label to number if needed.
            loss, grads = ll.loss_and_gradients(x, y, params)
            cum_loss += loss
            # YOUR CODE HERE
            # update the parameters according to the gradients
            # and the learning rate.
            params[0] -= learning_rate * grads[0]
            params[1] -= learning_rate * grads[1]

        train_loss = cum_loss / len(train_data)
        train_accuracy = accuracy_on_dataset(train_data, params)
        dev_accuracy = accuracy_on_dataset(dev_data, params)
        print(I, train_loss, train_accuracy, dev_accuracy)
    return params


if __name__ == '__main__':
    # YOUR CODE HERE
    # write code to load the train and dev sets, set up whatever you need,
    # and call train_classifier.

    # ...
    in_dim = len(utils.vocab)
    out_dim = len(utils.LANGUAGES)
    num_iterations = 1000
    learning_rate = 0.01
    train_data, dev_data = utils.TRAIN, utils.DEV
    weights = ll.create_classifier(in_dim, out_dim)
    trained_params = train_classifier(train_data, dev_data, num_iterations, learning_rate, weights)
