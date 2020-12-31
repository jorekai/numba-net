from activations import *


def test_relu_serialize():
    a = [-1, 0, 0.5, 1]
    b = s_relu(a)
    c = [0, 0, 0.5, 1]
    assert all(b == c)


def test_relu_parallel():
    a = [-1, 0, 0.5, 1]
    b = p_relu(a)
    c = [0, 0, 0.5, 1]
    assert all(b == c)


def test_relu_serialize_d():
    a = [-1, 0, 1]
    b = s_relu_d(a)
    c = [0, 0, 1]
    assert all(b == c)


def test_relu_parallel_d():
    a = [-1, 0, 1]
    b = p_relu_d(a)
    c = [0, 0, 1]
    assert all(b == c)


def test_sigmoid_serialize():
    a = [-1, 0, 1]
    b = s_sigmoid(a)
    c = [(1. / (1 + np.exp(-x))) for x in a]
    np.testing.assert_allclose(b, c)


def test_sigmoid_parallel():
    a = [-1, 0, 1]
    b = p_sigmoid(a)
    c = [(1. / (1 + np.exp(-x))) for x in a]
    np.testing.assert_allclose(b, c)


def test_sigmoid_serialize_d():
    a = [-1, 0, 1]
    b = s_sigmoid_d(a)
    c = [(x - x ** 2) for x in a]
    np.testing.assert_allclose(b, c)


def test_sigmoid_parallel_d():
    a = [-1, 0, 1]
    b = p_sigmoid_d(a)
    c = [(x - x ** 2) for x in a]
    np.testing.assert_allclose(b, c)


def test_tanh_serialize():
    a = [-1, 0, 1]
    b = s_tanh(a)
    c = np.tanh(a)
    np.testing.assert_allclose(b, c)


def test_tanh_parallel():
    a = [-1, 0, 1]
    b = p_tanh(a)
    c = np.tanh(a)
    np.testing.assert_allclose(b, c)
