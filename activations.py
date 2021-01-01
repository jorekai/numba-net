import numpy as np
from numba import guvectorize, float32, float64


def gu_activate(func, *args, **kwargs):
    """
    The guvectorize decorated method, runs in nopython mode and fastmaths which favors speed over precision
    since this is a DL library everything is pretty unprecise and fastmaths performs well enough
    :param func: a activation method or its derivative method
    :param target: choose between | 1: None -> serial | 2: 'parallel' -> parallel | execution method
    :return: the decorated activation function
    """
    kwargs_ = {k: v for k, v in kwargs.items() if v is not None}
    return guvectorize([(float32[:], float32[:]),
                        (float64[:], float64[:])],
                       '(n)->(n)', nopython=True, fastmath=True, *args, **kwargs_)(func)


def relu(x, y):
    """
    The Rectified Linear Unit Activation
    :param x: input vector
    :param y: output vector
    :return: y
    """
    for i in range(x.shape[0]):
        y[i] = max(0, x[i])


def relu_d(x, y):
    """
    The Derivative of the Rectified Linear Unit Activation
    :param x: input vector
    :param y: output vector
    :return: y
    """
    for i in range(x.shape[0]):
        y[i] = 1 if x[i] > 0 else 0


def tanh(x, y):
    """
    Tangens Hyperbolicus Activation Function
    :param x: input vector
    :param y: output vector
    :return: y
    """
    for i in range(x.shape[0]):
        y[i] = np.tanh(x[i])


def tanh_d(x, y):
    """
    The Derivative of Tangens Hyperbolicus Activation Function
    :param x: input vector
    :param y: output vector
    :return: y
    """
    for i in range(x.shape[0]):
        y[i] = 1 - x[i] ** 2


def sigmoid(x, y):
    """
    The Sigmoid Activation Function
    :param x: input vector
    :param y: output vector
    :return: y
    """
    for i in range(x.shape[0]):
        y[i] = 1. / (1 + np.exp(-x[i]))


def sigmoid_d(x, y):
    """
    The Derivative of the Sigmoid Activation Function
    :param x: input vector
    :param y: output vector
    :return: y
    """
    for i in range(x.shape[0]):
        y[i] = x[i] - x[i] ** 2


"""
Below are the exported default methods
ATM only serialized and parallel implementation is supported
"""
s_relu = gu_activate(relu)
p_relu = gu_activate(relu, target='parallel')
s_relu_d = gu_activate(relu_d)
p_relu_d = gu_activate(relu_d, target='parallel')

s_tanh = gu_activate(tanh)
p_tanh = gu_activate(tanh, target='parallel')
s_tanh_d = gu_activate(tanh_d)
p_tanh_d = gu_activate(tanh_d, target='parallel')

s_sigmoid = gu_activate(sigmoid)
p_sigmoid = gu_activate(sigmoid, target='parallel')
s_sigmoid_d = gu_activate(sigmoid_d)
p_sigmoid_d = gu_activate(sigmoid_d, target='parallel')
