from collections import defaultdict

import numpy as np

import train_loglin
import mlp1
import random

import utils

import xor_data

STUDENT = {'name': 'YOUR NAME',
           'ID': 'YOUR ID NUMBER'}


if __name__ == '__main__':
    # YOUR CODE HERE
    # write code to load the train and dev sets, set up whatever you need,
    # and call train_classifier.

    # ...

    uni = 0

    in_dim = len(utils.vocab) if uni < 2 else 2
    out_dim = len(utils.LANGUAGES) if uni < 2 else 2

    hidden_dim = 20
    num_iterations = 1000
    learning_rate = 0.1
    train_data, dev_data = (utils.uni_TRAIN, utils.uni_DEV) if uni == 1 else ((utils.TRAIN, utils.DEV) if uni == 0 else (xor_data.data, xor_data.data))
    weights = mlp1.create_classifier(in_dim, hidden_dim, out_dim)
    trained_params = train_loglin.train_classifier(train_data, dev_data, num_iterations, learning_rate, weights, mlp1, uni)
