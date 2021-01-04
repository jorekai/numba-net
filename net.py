## Neural net layers
from losses import s_mse


class Sequential:
    """
    The Sequential Class mainly inspired by keras sequential class
    Initialization requires layers, batch size and loss function
    """

    def __init__(self, layers=None, batch_size=24, loss=s_mse):
        """
        Init a sequential net
        :param layers: a List of Layers
        :param batch_size: int size
        :param loss: losses function
        """
        if type(layers) in [list, tuple]:
            self.layers = layers
        elif type(layers) == str:
            import dill
            self.layers = dill.load(open(layers, 'rb'))
        else:
            self.layers = None
        self.batch_size = batch_size
        self.loss = loss

    def configure(self, batch_size, optimizer, objective):
        """
        configure to init the sequential net
        :param batch_size: batch size for layers
        :param optimizer: optimizer method for gradient descent
        :param objective: the goal
        :return:
        """
        for layer in self.layers:
            layer.configure(batch_size=batch_size, optimizer=optimizer)
        self.batch_size = batch_size
        self.loss = objective

    def saveas(self, filename):
        """
        Save the sequential net with its layers
        :param filename: str
        :return: None
        """
        import dill
        dill.dump(self.layers, open(filename, 'wb'))

    def forward(self, x):
        """
        recursive forward pass
        :param x: input vector
        :return: result of forward pass
        """
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, de_dy):
        """
        recursive backward pass for each layer
        :param de_dy: the error, loss derivative
        :return: propagated derivative
        """
        for layer in self.layers[::-1]:
            de_dy = layer.backward(de_dy)
        return de_dy

    def predict(self, x):
        """
        predict on input vector, no update
        :param x: input vector
        :return: prediction value
        """
        for layer in self.layers:
            x = layer.predict(x)
        return x

    def update(self):
        """
        Updates recursively the weights
        :return: None
        """
        for layer in self.layers:
            layer.update()

    def train(self, x, y, epochs, display=True):
        """Train input on target vector for x epochs """
        bs = self.batch_size
        lim = x.shape[0] + (x.shape[0] % bs == 0) * bs
        try:
            for i in range(epochs):
                for j in range(bs, lim, bs):
                    yp = self.forward(x[j - bs:j])
                    loss, deriv = self.loss(y[j - bs:j], yp)
                    if display:
                        print('loss: {}'.format(loss.mean()))
                    self.backward(deriv)
                    self.update()
        except KeyboardInterrupt:
            return
