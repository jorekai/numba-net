from losses import *

x1 = np.array([-1, 0, 1], dtype=np.float32)
y1 = np.array([-1, 0, 1], dtype=np.float32)


def test_s_mse():
    loss, error = s_mse(x1, y1)
    assert loss == 0
    assert all(error == [0, 0, 0])
    assert type(error) == np.ndarray
    for x in error:
        assert type(x) == np.float32


def test_p_mse():
    loss, error = p_mse(x1, y1)
    assert loss == 0
    assert all(error == [0, 0, 0])
    assert type(error) == np.ndarray
    for x in error:
        assert type(x) == np.float32
