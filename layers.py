import numpy as np

from optimizers import s_sgd
from utils import numba_backward, numba_predict


class Dense:

    def __init__(self, input_dim, output_dim, batch_size=256, optimizer=s_sgd):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.batch_size = batch_size
        self.optimizer = optimizer

        # layer params
        self.var = np.sqrt(2.0 / (self.input_dim + self.output_dim))
        self.W = np.random.normal(0, self.var, (self.input_dim, self.output_dim))
        self.b = np.abs(np.random.normal(0, self.var, (self.output_dim)))

        # init params
        self.x = np.zeros((batch_size, self.input_dim))
        self.auxW = np.zeros_like(self.W)
        self.auxb = np.zeros_like(self.b)

        # parameter derivatives
        self.de_dW = np.zeros_like(self.W)
        self.de_db = np.zeros_like(self.b)

    def forward(self, x):
        self.x = x
        return self.predict(x)

    def backward(self, de_dy):
        self.de_dW, self.de_db, de_dY = numba_backward(self.x.T, self.W, de_dy, self.batch_size)
        return de_dY

    def predict(self, x):
        return numba_predict(x, self.W, self.b)

    def update(self):
        self.W, self.aW = self.optimizer(self.W, self.de_dW, self.aW)
        self.b, self.ab = self.optimizer(self.b, self.de_db, self.ab)
