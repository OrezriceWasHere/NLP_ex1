from collections import defaultdict

import numpy as np

import loglinear as ll
import random

import utils

import xor_data

STUDENT = {'name': 'YOUR NAME',
           'ID': 'YOUR ID NUMBER'}


def feats_to_vec(features, uni):
    # YOUR CODE HERE.
    # Should return a numpy vector of features.

    if uni == 2:
        return np.array(features)

    dict = utils.uni_F2I if uni == 1 else utils.F2I

    length = len(features)
    vec = np.zeros(len(utils.vocab))
    for feature in features:
        if feature in dict:
            vec[dict[feature]] += 1
    vec = vec / length
    return vec


def accuracy_on_dataset(dataset, params, module, uni):
    good = bad = 0.0
    for label, features in dataset:
        # YOUR CODE HERE
        # Compute the accuracy (a scalar) of the current parameters
        # on the dataset.
        # accuracy is (correct_predictions / all_predictions)
        x = feats_to_vec(features, uni)
        y = utils.CLASS_TO_INDEX[label] if type(label) is not int else label
        if y == module.predict(x, params):
            good += 1
        else:
            bad += 1
    return good / (good + bad)


def train_classifier(train_data, dev_data, num_iterations, learning_rate, params, module=ll, uni=0):
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
            x = feats_to_vec(features, uni)  # convert features to a vector.
            y = utils.CLASS_TO_INDEX[label] if type(
                label) is not int else label  # convert the label to number if needed.
            loss, grads = module.loss_and_gradients(x, y, params)
            cum_loss += loss
            # YOUR CODE HERE
            # update the parameters according to the gradients
            # and the learning rate.

            for i in range(len(params)):
                params[i] -= learning_rate * grads[i]

            # print(grads[1])

        train_loss = cum_loss / len(train_data)
        train_accuracy = accuracy_on_dataset(train_data, params, module, uni)
        dev_accuracy = accuracy_on_dataset(dev_data, params, module, uni)
        print(I, train_loss, train_accuracy, dev_accuracy)
    return params


if __name__ == '__main__':
    # YOUR CODE HERE
    # write code to load the train and dev sets, set up whatever you need,
    # and call train_classifier.

    # ...

    uni = 0

    in_dim = len(utils.vocab) if uni < 2 else 2
    out_dim = len(utils.LANGUAGES) if uni < 2 else 1
    num_iterations = 3000
    learning_rate = 0.05
    train_data, dev_data = (utils.uni_TRAIN, utils.uni_DEV) if uni else (utils.TRAIN, utils.DEV)
    weights = ll.create_classifier(in_dim, out_dim)
    trained_params = train_classifier(train_data, dev_data, num_iterations, learning_rate, weights, ll, uni)

    with open('test.pred', 'w') as file:
        for label, features in utils.TEST:
            x = feats_to_vec(features, uni)
            file.write(utils.LANGUAGES[ll.predict(x, trained_params)] + '\n')
