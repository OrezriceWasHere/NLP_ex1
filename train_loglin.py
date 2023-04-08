import numpy as np
import math
import loglinear as ll
import random

import utils

STUDENT = {'name': 'YOUR NAME',
           'ID': 'YOUR ID NUMBER'}


def feats_to_vec(features):
    # YOUR CODE HERE.
    # Should return a numpy vector of features.
    return np.fromiter((utils.F2I.get(feature, utils.MISFIT_INDEX) for feature in features), dtype=np.float32)


def accuracy_on_dataset(dataset, params):
    good = bad = 0.0
    for label, features in dataset:
        # YOUR CODE HERE
        # Compute the accuracy (a scalar) of the current parameters
        # on the dataset.
        # accuracy is (correct_predictions / all_predictions)
        good += 1 if utils.CLASS_TO_INDEX[label] == ll.predict(feats_to_vec(features), params) else 0
        bad += 1 if utils.CLASS_TO_INDEX[label] != ll.predict(feats_to_vec(features), params) else 0

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
        #random.shuffle(train_data)
        index = 0
        for label, features in train_data:
            index += 1
            x = feats_to_vec(features)  # convert features to a vector.
            y = utils.CLASS_TO_INDEX[label]  # convert the label to number if needed.
            loss, grads = ll.loss_and_gradients(x, y, params)
            cum_loss += loss
            # YOUR CODE HERE
            W, b = params
            params = [W - learning_rate * grads[0], b - learning_rate * grads[1]]

            # update the parameters according to the gradients
            # and the learning rate.

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


    train_data, dev_data = utils.TRAIN, utils.DEV
    learning_rate = 0.01
    num_iterations = 100
    in_dim, out_dim = utils.INPUT_SIZE, len(utils.OUTPUT_CLASSES)

    params = ll.create_classifier(in_dim, out_dim)
    trained_params = train_classifier(train_data, dev_data, num_iterations, learning_rate, params)
