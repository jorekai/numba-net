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


# FROM HERE ON NOT YET NUMBAD
class Softmax:
    # Still must be numba conformed
    def __init__(self):
        pass

    def forward(self, x):
        self.y = self.predict(x)
        return self.y

    def backward(self, de_dy):
        ey = de_dy * self.y
        ss = ey.sum(axis=1).reshape((-1, 1))
        return ey - self.y * ss

    def configure(self, batch_size, optimizer):
        pass

    def predict(self, x):
        xx = x.T
        e = np.exp(xx - xx.max(axis=0))
        return np.transpose(e / e.sum(axis=0))

    def update(self):
        pass


def get_im2col_indices(x_shape, field_height, field_width, padding=1, stride=1):
    # First figure out what the size of the output should be
    N, C, H, W = x_shape
    assert (H + 2 * padding - field_height) % stride == 0
    assert (W + 2 * padding - field_height) % stride == 0
    out_height = (H + 2 * padding - field_height) // stride + 1
    out_width = (W + 2 * padding - field_width) // stride + 1

    i0 = np.repeat(np.arange(field_height), field_width)
    i0 = np.tile(i0, C)
    i1 = stride * np.repeat(np.arange(out_height), out_width)
    j0 = np.tile(np.arange(field_width), field_height * C)
    j1 = stride * np.tile(np.arange(out_width), out_height)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)

    k = np.repeat(np.arange(C), field_height * field_width).reshape(-1, 1)

    return (k, i, j)


def im2col_indices(x, field_height, field_width, padding=1, stride=1):
    """ An implementation of im2col based on some fancy indexing """
    # Zero-pad the input
    p = padding
    x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')

    k, i, j = get_im2col_indices(x.shape, field_height, field_width, padding,
                                 stride)

    cols = x_padded[:, k, i, j]
    C = x.shape[1]
    cols = cols.transpose(1, 2, 0).reshape(field_height * field_width * C, -1)
    return cols


def col2im_indices(cols, x_shape, field_height=3, field_width=3, padding=1,
                   stride=1):
    """ An implementation of col2im based on fancy indexing and np.add.at """
    N, C, H, W = x_shape
    H_padded, W_padded = H + 2 * padding, W + 2 * padding
    x_padded = np.zeros((N, C, H_padded, W_padded), dtype=cols.dtype)
    k, i, j = get_im2col_indices(x_shape, field_height, field_width, padding,
                                 stride)
    cols_reshaped = cols.reshape(C * field_height * field_width, -1, N)
    cols_reshaped = cols_reshaped.transpose(2, 0, 1)
    np.add.at(x_padded, (slice(None), k, i, j), cols_reshaped)
    if padding == 0:
        return x_padded
    return x_padded[:, :, padding:-padding, padding:-padding]


class Conv:

    def __init__(self, input_dim, n_filters, size, stride=1, padding='same'):
        """
        Convolutional layer for neural network
        input_dim::Tuple{Int,Int,Int} = (channels, height, width)
        size::Tuple{Int,Int} = height, width of filter
        """

        self.h_filter = self.w_filter = size
        assert (self.h_filter % 2 == 1), "Filter height must be an odd integer"
        self.n_filters = n_filters  # number of conv kernels
        assert (len(input_dim) == 3), "Input shape must be (channels, height, width)"
        self.input_dim = input_dim  # channels, height, width
        self.stride = stride
        if padding == 'same':
            padding = size // 2
        self.padding = padding

        d_x, h_x, w_x = input_dim
        self.h_out = (h_x - self.h_filter + 2 * padding) // stride + 1
        self.w_out = (w_x - self.w_filter + 2 * padding) // stride + 1
        self.output_dim = (n_filters, self.h_out, self.w_out)

        f = d_x * self.h_filter * self.w_filter * n_filters
        self.W_col = np.random.normal(0, 1.0 / f, (n_filters, d_x * self.h_filter * self.w_filter))
        self.b = np.abs(np.random.normal(0, 1.0 / f, (n_filters, 1)))
        self.de_dW_col = None
        self.de_db = None
        self.X_col = None
        self.in_shape = None

    def configure(self, batch_size, optimizer):
        self.batch_size = batch_size

        aux_init, self.update_rule = optimizer
        self.dW_col = aux_init(self.W_col)
        self.db = aux_init(self.b)

    def forward(self, X):
        # efficient convolution operation using im2col
        X_col = im2col_indices(X, self.h_filter, self.w_filter,
                               padding=self.padding, stride=self.stride)
        self.X_col = X_col
        self.in_shape = X.shape  # batch_size, depth, height, width
        out = self.W_col.dot(X_col) + self.b
        out = out.reshape(self.n_filters, self.h_out, self.w_out, X.shape[0])
        out = out.transpose(3, 0, 1, 2)
        return out

    def backward(self, de_dy):
        db = np.mean(de_dy, axis=(0, 2, 3))
        self.de_db = db.reshape(self.n_filters, -1)

        dout_reshaped = de_dy.transpose(1, 2, 3, 0).reshape(self.n_filters, -1)
        self.de_dW_col = dout_reshaped.dot(self.X_col.T) / self.in_shape[0]
        dX_col = self.W_col.T.dot(dout_reshaped)
        de_dX = col2im_indices(dX_col, self.in_shape, self.h_filter, self.w_filter,
                               padding=self.padding, stride=self.stride)
        return de_dX

    def predict(self, X):
        X_col = im2col_indices(X, self.h_filter, self.w_filter,
                               padding=self.padding, stride=self.stride)
        out = self.W_col.dot(X_col) + self.b
        out = out.reshape(self.n_filters, self.h_out, self.w_out, X.shape[0])
        out = out.transpose(3, 0, 1, 2)
        return out

    def update(self):
        self.W_col, self.dW_col = self.update_rule(self.W_col, self.de_dW_col, self.dW_col)
        self.b, self.db = self.update_rule(self.b, self.de_db, self.db)


class Maxpool:

    def __init__(self, input_dim, size, stride):
        """maxpooling layer"""
        assert (len(input_dim) == 3)
        self.stride = stride
        self.size = size
        assert (type(size) == int)
        self.in_shape = None
        self.X_col = None
        self.max_idx = None
        assert (size == stride)
        d, h, w = input_dim
        assert (h % size == 0)
        assert (w % size == 0)

    def forward(self, X):
        n, d, h, w = X.shape
        h_out = h // self.size
        w_out = w // self.size
        self.in_shape = X.shape
        X_reshaped = X.reshape(n * d, 1, h, w)
        self.X_col = im2col_indices(X_reshaped, self.size, self.size, padding=0, stride=self.stride)
        self.max_idx = np.argmax(self.X_col, axis=0)
        out = self.X_col[self.max_idx, range(self.max_idx.size)]
        out = out.reshape(h_out, w_out, n, d)
        out = out.transpose(2, 3, 0, 1)
        return out

    def configure(self, batch_size, optimizer):
        pass

    def backward(self, de_dy):
        n, d, h, w = self.in_shape
        dX_col = np.zeros_like(self.X_col)
        dout_flat = de_dy.transpose(2, 3, 0, 1).ravel()
        dX_col[self.max_idx, range(self.max_idx.size)] = dout_flat
        de_dX = col2im_indices(dX_col, (n * d, 1, h, w), self.size,
                               self.size, padding=0, stride=self.stride)
        dX = de_dX.reshape(self.in_shape)
        return dX

    def predict(self, X):
        n, d, h, w = X.shape
        h_out = h // self.size
        w_out = w // self.size
        X_reshaped = X.reshape(n * d, 1, h, w)
        X_col = im2col_indices(X_reshaped, self.size, self.size, padding=0, stride=self.stride)
        max_idx = np.argmax(X_col, axis=0)
        out = X_col[max_idx, range(max_idx.size)]
        out = out.reshape(h_out, w_out, n, d)
        out = out.transpose(2, 3, 0, 1)
        return out

    def update(self):
        pass


####################################################################################
class Conv2Dense:

    def __init__(self, output_dim):
        assert (type(output_dim) == int)
        self.in_dim = None
        self.output_dim = output_dim

    def configure(self, batch_size, optimizer):
        pass

    def forward(self, x):
        self.in_dim = x.shape
        return x.reshape((-1, self.output_dim))

    def backward(self, de_dy):
        return de_dy.reshape(self.in_dim)

    def predict(self, x):
        return x.reshape((-1, self.output_dim))

    def update(self):
        pass
