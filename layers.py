from activations import *
from optimizers import s_sgd
from utils import numba_backward, numba_predict


class Dense:

    def __init__(self, input_dim, output_dim, batch_size=24, optimizer=s_sgd):
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

    def configure(self, batch_size, optimizer):
        self.batch_size = batch_size
        self.optimizer = optimizer

    def forward(self, x):
        self.x = x
        return self.predict(x)

    def backward(self, de_dy):
        self.de_dW, self.de_db, de_dY = numba_backward(self.x, self.W, de_dy, self.batch_size)
        return de_dY

    def predict(self, x):
        return numba_predict(x, self.W, self.b)

    def update(self, lr=1e-2, mu=0.9, decay=0):
        self.W, self.auxW = self.optimizer(self.W, self.de_dW, self.auxW, lr, mu, decay)
        self.b, self.auxb = self.optimizer(self.b, self.de_db, self.auxb, lr, mu, decay)


class Activation:

    def __init__(self, name):
        self.fwd, self.bwd = {
            'relu': (s_relu, s_relu_d),
            'tanh': (s_tanh, s_tanh_d),
            'sigmoid': (s_sigmoid, s_sigmoid_d),
        }[name]
        self.y = None

    def configure(self, batch_size, optimizer):
        pass

    def forward(self, x):
        self.y = self.fwd(x)
        return self.y

    def backward(self, de_dy):
        return de_dy * self.bwd(self.y)

    def predict(self, x):
        return self.fwd(x)

    def update(self):
        pass


class Dropout:

    def __init__(self, ratio=0.4):
        self.ratio = ratio

    def configure(self, batch_size, optimizer):
        pass

    def forward(self, x):
        self.mask = 1.0 * (np.random.rand(*x.shape) > self.ratio)
        return x * self.mask

    def backward(self, de_dy):
        return de_dy * self.mask

    def predict(self, x):
        return x

    def update(self):
        pass
