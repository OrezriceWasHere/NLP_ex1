import numpy as np

import loglinear
import mlp1

STUDENT = {'name': 'YOUR NAME',
           'ID': 'YOUR ID NUMBER'}


def classifier_output(x, params):
    tuples = list(zip(params[0::2], params[1::2]))

    h = x
    for t in tuples[:-1]:
        h = mlp1.hidden_output(h, t)

    return loglinear.classifier_output(h, tuples[-1])


def predict(x, params):
    return np.argmax(classifier_output(x, params))


def loss_and_gradients(x, y, params):
    """
    params: a list as created by create_classifier(...)

    returns:
        loss,[gW1, gb1, gW2, gb2, ...]

    loss: scalar
    gW1: matrix, gradients of W1
    gb1: vector, gradients of b1
    gW2: matrix, gradients of W2
    gb2: vector, gradients of b2
    ...

    (of course, if we request a linear classifier (ie, params is of length 2),
    you should not have gW2 and gb2.)
    """

    tuples = list(zip(params[0::2], params[1::2]))

    hs = [x]
    for i, t in enumerate(tuples[:-1]):
        hs.append(mlp1.hidden_output(hs[-1], t))

    loss, (gU, gb_tag, gh) = loglinear.loss_and_gradients(hs[-1], y, tuples[-1])

    gradients = []

    for x, h, t in reversed(list(zip(hs, hs[1:], tuples))):
        W, b = t
        z = np.dot(x, W) + b

        sech = 1 / np.cosh(z) ** 2

        dL_du = gh * sech

        gW = np.outer(x, dL_du)

        gb = dL_du

        gh = gb @ np.transpose(W)

        gradients.insert(0, gb)
        gradients.insert(0, gW)

    return loss, gradients + [gU, gb_tag]


def create_classifier(dims):
    """
    returns the parameters for a multi-layer perceptron with an arbitrary number
    of hidden layers.
    dims is a list of length at least 2, where the first item is the input
    dimension, the last item is the output dimension, and the ones in between
    are the hidden layers.
    For example, for:
        dims = [300, 20, 30, 40, 5]
    We will have input of 300 dimension, a hidden layer of 20 dimension, passed
    to a layer of 30 dimensions, passed to learn of 40 dimensions, and finally
    an output of 5 dimensions.
    
    Assume a tanh activation function between all the layers.

    return:
    a flat list of parameters where the first two elements are the W and b from input
    to first layer, then the second two are the matrix and vector from first to
    second layer, and so on.
    """

    params = sum([loglinear.create_classifier(i, o) for i, o in zip(dims, dims[1:])], [])
    return params
