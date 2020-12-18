from math import fabs
from random import normalvariate

import numpy as np
from numba import float32, guvectorize, float64, prange, njit  # import the types


def gu_random_normal(func, *args, **kwargs):
    """
    The guvectorize decorated method, runs in nopython mode
    :param func: a loss error method
    :param target: choose between | 1: None -> serial | 2: 'parallel' -> parallel | execution method
    :return: the decorated loss, error function
    """
    kwargs_ = {k: v for k, v in kwargs.items() if v is not None}
    return guvectorize([(float32, float32, float32[:, :], float32[:, :]),
                        (float64, float64, float64[:, :], float64[:, :])], '(),(),(m,n)->(m,n)', nopython=True,
                       fastmath=True, *args, **kwargs_)(func)


def gu_random_normal_bias(func, *args, **kwargs):
    """
    The guvectorize decorated method, runs in nopython mode
    :param func: a loss error method
    :param target: choose between | 1: None -> serial | 2: 'parallel' -> parallel | execution method
    :return: the decorated loss, error function
    """
    kwargs_ = {k: v for k, v in kwargs.items() if v is not None}
    return guvectorize([(float32, float32, float32[:], float32[:]),
                        (float64, float64, float64[:], float64[:])], '(),(),(m)->(m)', nopython=True, fastmath=True,
                       *args, **kwargs_)(func)


def random_normal(mu, sigma, arr, out):
    dim1 = arr[0].shape
    dim2 = arr[1].shape
    for i in prange(*dim1):
        for j in prange(*dim2):
            out[i][j] = normalvariate(mu, sigma)


def random_normal_bias(mu, sigma, arr, out):
    dim1 = arr.shape
    for i in prange(*dim1):
        out[i] = fabs(normalvariate(mu, sigma))


@njit(fastmath=True, nopython=True)
def matmul(a, b):
    return a @ b


@njit(fastmath=True)
def numba_predict(x, W, b):
    return matmul(x, W) + b


def numba_backward(x, W, de_dy, batch_size=24):
    de_dW = matmul(x.T, de_dy) / batch_size
    de_db = np.mean(de_dy, axis=0)
    return de_dW, de_db, matmul(de_dy, W.T)


s_rnd_norm_W = gu_random_normal(random_normal)
p_rnd_norm_W = gu_random_normal(random_normal, target='parallel')
s_rnd_norm_b = gu_random_normal(random_normal)
p_rnd_norm_b = gu_random_normal(random_normal, target='parallel')
