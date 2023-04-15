from collections import defaultdict

import numpy as np

import train_loglin
import mlp1
import random

import utils

STUDENT = {'name': 'YOUR NAME',
           'ID': 'YOUR ID NUMBER'}


if __name__ == '__main__':
    # YOUR CODE HERE
    # write code to load the train and dev sets, set up whatever you need,
    # and call train_classifier.

    # ...

    uni = True

    in_dim = len(utils.vocab)
    hidden_dim = 50 # for 100 it was 0.85, 50 was 0.863
    out_dim = len(utils.LANGUAGES)
    num_iterations = 1000
    learning_rate = 0.01
    train_data, dev_data = (utils.uni_TRAIN, utils.uni_DEV) if uni else (utils.TRAIN, utils.DEV)
    weights = mlp1.create_classifier(in_dim, hidden_dim, out_dim)
    trained_params = train_loglin.train_classifier(train_data, dev_data, num_iterations, learning_rate, weights, mlp1, uni)
