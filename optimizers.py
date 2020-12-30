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
    """
    :param x: input vector
    :param y: target vector
    :param aux: auxilliary vector
    :param lr: learning rate alpha
    :param mu: factor auxilliaray update
    :param decay: temporal decay factor
    :param out: update vector
    :param init: update auxilliary vector
    :return: (out, init)
    """
    init[:] = aux[:]
    for i in prange(x.shape[0]):
        out[i] = (1 - decay) * x[i]
        init[i] = mu * init[i] - lr * y[i]
        out[i] = np.add(out[i], init[i])


def gu_adam_optimizer(func, *args, **kwargs):
    kwargs_ = {k: v for k, v in kwargs.items() if v is not None}
    return guvectorize([(float32[:], float32[:], float32[:, :], float32, float32, float32, float32, float32, float32[:],
                         float32[:, :]),
                        (float64[:], float64[:], float64[:, :], float64, float64, float64, float64, float64, float64[:],
                         float64[:, :])],
                       '(n),(n),(m,n),(),(),(),(),()->(n),(m,n)', nopython=True, fastmath=True, *args, **kwargs_)(func)


def adam(x, y, aux, lr, beta1, beta2, eps, decay, out, init):
    """
    :param x: input vector
    :param y: target vector
    :param aux: auxilliary vector
    :param lr: learning rate alpha
    :param beta1: hyperparam1
    :param beta2: hyperparam2
    :param eps: avoid vanishing gradient/zero division
    :param decay: weight decay factor
    :param out: update vector
    :param init: update auxilliary vector
    :return: (out, init)
    """
    init[:][:] = aux[:][:]
    for i in prange(aux.shape[1]):
        init[0][i] = beta1 * aux[0][i] + (1 - beta1) * y[i]
    for j in prange(aux.shape[1]):
        init[1][j] = beta2 * aux[1][j] + (1 - beta2) * y[j] ** 2
    out[:] = (1 - decay) * x[:]
    out[:] = out[:] + (-lr * init[0] / (np.sqrt(init[1]) + eps))


s_sgd = gu_optimizer(sgd)
p_sgd = gu_optimizer(sgd, target='parallel')
s_adam = gu_adam_optimizer(adam)
p_adam = gu_adam_optimizer(adam, target='parallel')
