import numpy as np
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


def random_normal(mu, sigma, arr, out):
    dim1 = arr[0].shape
    dim2 = arr[1].shape
    for i in prange(*dim1):
        for j in prange(*dim2):
            out[i][j] = np.random.normal(mu, sigma)


s_random_normal = gu_random_normal(random_normal)
p_random_normal = gu_random_normal(random_normal, target='parallel')
