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
