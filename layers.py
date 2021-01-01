from activations import *
from optimizers import s_sgd
from utils import numba_backward, numba_predict

"""
The different Layers are implemented in this file.
Every sequential net consists of different/stacked layers
"""


class Dense:
    def __init__(self, input_dim, output_dim, batch_size=24, optimizer=s_sgd):
        """
        The Dense Layer is equal to a fully connected keras layer
        :param input_dim: Input dimension which is the size of the input vector
        :param output_dim: Output dimension which is the size of the output vector
        :param batch_size: the batch size which is fitted in one iteration at training time
        :param optimizer: the optimizer object implemented in optimizers.py
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.batch_size = batch_size
        self.optimizer = optimizer

        # layer parameters
        # variance of weights initialization
        # the weight vector initialized by a fast numba method
        # the bias vector added to each weight update
        self.var = np.sqrt(2.0 / (self.input_dim + self.output_dim))
        self.W = np.random.normal(0, self.var, (self.input_dim, self.output_dim))
        self.b = np.abs(np.random.normal(0, self.var, (self.output_dim)))

        # init parameters of input zeros and auxilliary weight/bias
        self.x = np.zeros((batch_size, self.input_dim))
        self.auxW = np.zeros_like(self.W)
        self.auxb = np.zeros_like(self.b)

        # init parameters derivatives
        self.de_dW = np.zeros_like(self.W)
        self.de_db = np.zeros_like(self.b)

    def configure(self, batch_size, optimizer):
        """
        The configuration is used to set optimizer method and batch update size
        :param batch_size: int > 0
        :param optimizer: optimizers.py
        :return: void
        """
        self.batch_size = batch_size
        self.optimizer = optimizer

    def forward(self, x):
        """
        The forward pass to predict values
        :param x: input vector
        :return: prediction vector
        """
        self.x = x
        return self.predict(x)

    def backward(self, de_dy):
        """
        The backward pass is needed to update our weights
        :param de_dy: the derivative is passed backwards to update the weights
        :return: derivative of the target vector
        """
        self.de_dW, self.de_db, de_dY = numba_backward(self.x, self.W, de_dy, self.batch_size)
        return de_dY

    def predict(self, x):
        """
        The prediction of the input vector x
        :param x: input vector
        :return: prediction values
        """
        return numba_predict(x, self.W, self.b)

    def update(self, lr=1e-2, mu=0.9, decay=0):
        """
        Update method by using our optimizer
        :param lr: learning rate alpha
        :param mu: mu parameter
        :param decay: weight update decay factor
        :return:
        """
        self.W, self.auxW = self.optimizer(self.W, self.de_dW, self.auxW, lr, mu, decay)
        self.b, self.auxb = self.optimizer(self.b, self.de_db, self.auxb, lr, mu, decay)


class Activation:
    def __init__(self, name):
        """
        The activation layer is treated as a layer even though it is only the activation function
        TODO: implement a choice to parallel acces of activation function
        :param name: choose->("relu","tanh", "sigmoid")
        """
        self.fwd, self.bwd = {
            'relu': (s_relu, s_relu_d),
            'tanh': (s_tanh, s_tanh_d),
            'sigmoid': (s_sigmoid, s_sigmoid_d),
        }[name]
        self.y = None

    def configure(self, batch_size, optimizer):
        pass

    def forward(self, x):
        """
        Forward here is equal to calculation of the activation function on vector
        :param x: input vector
        :return: f(x)
        """
        self.y = self.fwd(x)
        return self.y

    def backward(self, de_dy):
        """
        The derivative of the activation to update the Higher Order layer
        :param de_dy: derivative
        :return: calculate the derivative activation values
        """
        return de_dy * self.bwd(self.y)

    def predict(self, x):
        return self.fwd(x)

    def update(self):
        pass


class Dropout:

    def __init__(self, ratio=0.4):
        """
        The Dropout layer randomly masks weights in out network by a certain ratio
        :param ratio: 1 >= p >= 0
        """
        self.ratio = ratio

    def configure(self, batch_size, optimizer):
        """
        nothing to configure here
        """
        pass

    def forward(self, x):
        """
        The forward pass is the masking
        :param x: weight vector
        :return: masked weight vector
        """
        self.mask = 1.0 * (np.random.rand(*x.shape) > self.ratio)
        return x * self.mask

    def backward(self, de_dy):
        """
        backward pass of the masked weight vector for Higher Order layers
        :param de_dy: derivative vector
        :return: masked derivative vector
        """
        return de_dy * self.mask

    def predict(self, x):
        return x

    def update(self):
        pass
