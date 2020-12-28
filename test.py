# %%

import sys;
# !/usr/bin/python3 -u
import time

import numpy as np
from numba import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings
sys.path.append('..')
from net import Sequential
from layers import Dense, Activation
from losses import s_mse, p_mse
from optimizers import adam


def train_xor_net():
    x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.double)
    y = np.array([[0], [1], [1], [0]], dtype=np.double)
    x[x == 0] = 0.01;
    x[x == 1] = 0.99;
    y[y == 0] = 0.01;
    y[y == 1] = 0.99;

    net = Sequential([
        Dense(input_dim=2, output_dim=7),
        Activation('tanh'),

        Dense(input_dim=7, output_dim=1),
        Activation('sigmoid')
    ])

    net.configure(batch_size=2, objective=s_mse,
                  optimizer=adam())
    net.train(x, y, epochs=10000, display=False)
    print(net.predict(x))


# %%
def ntime():
    start = time.time()
    train_xor_net()
    end = time.time()
    print("Elapsed Numba Pre = %s" % (end - start))
    start = time.time()
    train_xor_net()
    end = time.time()
    print("Elapsed Numba Post = %s" % (end - start))


if __name__ == '__main__':
    warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
    warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)
    ntime()
