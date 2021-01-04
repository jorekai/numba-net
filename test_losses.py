from losses import *

x1 = np.array([-1, -1, -1], dtype=np.float32)
x2 = np.array([0, 0, 0], dtype=np.float32)
x3 = np.array([1, 1, 1], dtype=np.float32)

y1 = np.array([-1, -1, -1], dtype=np.float32)
y2 = np.array([0, 0, 0], dtype=np.float32)
y3 = np.array([1, 1, 1], dtype=np.float32)


def test_s_mse():
    loss, error = s_mse(x1, y1)
    assert loss == 0
    assert all(error == [0, 0, 0])
    assert type(error) == np.ndarray
