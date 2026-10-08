"""Reference CNN ops. Each fails for a real numerics defect."""
import numpy as np
from python.edge_npu.reference import relu, bias_add, conv2d_int8, max_pool, requantize


def test_relu_zeros_negatives():
    x = np.array([-2, -1, 0, 1, 2], dtype=np.int8)
    assert relu(x).tolist() == [0, 0, 0, 1, 2]


def test_bias_add_broadcasts():
    x = np.array([[1, 2], [3, 4]], dtype=np.int32)
    assert bias_add(x, np.array([10, 20], dtype=np.int32)).tolist() == [[11, 22], [13, 24]]


def test_conv2d_single_channel():
    x = np.arange(1, 10, dtype=np.int8).reshape(1, 3, 3)
    w = np.array([[[[1, 0], [0, -1]]]], dtype=np.int8)
    assert conv2d_int8(x, w).tolist() == [[[-4, -4], [-4, -4]]]


def test_max_pool_2x2():
    x = np.arange(1, 17, dtype=np.int8).reshape(1, 4, 4)
    assert max_pool(x, size=2, stride=2).tolist() == [[[6, 8], [14, 16]]]


def test_requantize_scales_and_clips():
    acc = np.array([100, -200, 100000], dtype=np.int32)
    assert requantize(acc, scale=0.1).tolist() == [10, -20, 127]
