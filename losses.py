import numpy as np
from numba import guvectorize, float32, float64


def gu_loss(func, *args, **kwargs):
    """
    The guvectorize decorated method, runs in nopython mode
    :param func: a loss error method
    :param target: choose between | 1: None -> serial | 2: 'parallel' -> parallel | execution method
    :return: the decorated loss, error function
    """
    kwargs_ = {k: v for k, v in kwargs.items() if v is not None}
    return guvectorize([(float32[:], float32[:], float32[:], float32[:]),
                        (float64[:], float64[:], float64[:], float64[:])],
                       '(n),(n)->(),(n)', nopython=True, *args, **kwargs_)(func)


def mse(x, y, loss, error):
    for i in range(y.shape[0]):
        error[i] = y[i] - x[i]
    loss[0] = 0.5 * np.sum(error[:] ** 2)


def cross_entropy(x, y, loss, derivation):
    l = np.log(y[:])
    for i in range(y.shape[0]):
        derivation[i] = -(x[i] / y[i])
    loss[0] = -np.sum(x[:] * l)


s_mse = gu_loss(mse)
p_mse = gu_loss(mse, target='parallel')
s_ce = gu_loss(cross_entropy)
p_ce = gu_loss(cross_entropy, target='parallel')
