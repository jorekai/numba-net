from math import fabs
from random import normalvariate

from numba import float32, guvectorize, float64, prange  # import the types


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


s_rnd_norm_W = gu_random_normal(random_normal)
p_rnd_norm_W = gu_random_normal(random_normal, target='parallel')
s_rnd_norm_b = gu_random_normal(random_normal)
p_rnd_norm_b = gu_random_normal(random_normal, target='parallel')
