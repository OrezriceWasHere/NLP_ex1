import numpy as np

import loglinear

STUDENT = {'name': 'YOUR NAME',
           'ID': 'YOUR ID NUMBER'}


def hidden_output(x, params):
    W, b = params[:2]
    return np.tanh(np.dot(x, W) + b)


def classifier_output(x, params):
    return loglinear.classifier_output(hidden_output(x, params), params[2:])


def predict(x, params):
    """
    params: a list of the form [W, b, U, b_tag]
    """
    return np.argmax(classifier_output(x, params))


def loss_and_gradients(x, y, params):
    """
    params: a list of the form [W, b, U, b_tag]

    returns:
        loss,[gW, gb, gU, gb_tag]

    loss: scalar
    gW: matrix, gradients of W
    gb: vector, gradients of b
    gU: matrix, gradients of U
    gb_tag: vector, gradients of b_tag
    """

    W, b = params[:2]
    z = np.dot(x, W) + b
    hidden = np.tanh(z)

    loss, (gU, gb_tag, gh) = loglinear.loss_and_gradients(hidden, y, params[2:])

    sech = 1 / np.cosh(z) ** 2

    dL_du = gh * sech

    gW = np.outer(x, dL_du)

    gb = dL_du

    return loss, [gW, gb, gU, gb_tag]


def create_classifier(in_dim, hid_dim, out_dim):
    """
    returns the parameters for a multi-layer perceptron,
    with input dimension in_dim, hidden dimension hid_dim,
    and output dimension out_dim.

    return:
    a flat list of 4 elements, W, b, U, b_tag.
    """
    params = loglinear.create_classifier(in_dim, hid_dim) + loglinear.create_classifier(hid_dim, out_dim)
    return params
