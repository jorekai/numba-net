import numpy as np
from numba import guvectorize, float32, float64, prange


def gu_optimizer(func, *args, **kwargs):
    """
    The guvectorize decorated method, runs in nopython mode
    :param func: a loss error method
    :param target: choose between | 1: None -> serial | 2: 'parallel' -> parallel | execution method
    :return: the decorated loss, error function
    """
    kwargs_ = {k: v for k, v in kwargs.items() if v is not None}
    return guvectorize([(float32[:], float32[:], float32[:], float32, float32, float32, float32[:], float32[:]),
                        (float64[:], float64[:], float64[:], float64, float64, float64, float64[:], float64[:])],
                       '(n),(n),(m),(),(),()->(n),(n)', nopython=True, fastmath=True, *args, **kwargs_)(func)


def sgd(x, y, aux, lr, mu, decay, out, init):
    init[:] = aux[:]
    for i in prange(x.shape[0]):
        out[i] = (1 - decay) * x[i]
        init[i] = mu * init[i] - lr * y[i]
        out[i] = np.add(out[i], init[i])


s_sgd = gu_optimizer(sgd)
p_sgd = gu_optimizer(sgd, target='parallel')
