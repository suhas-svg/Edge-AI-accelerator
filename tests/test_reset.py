import numpy as np
import pytest

from compiler.binary import write_model
from compiler.graph import Graph, Node
from python.edge_npu.commands import Command
from python.edge_npu.sdk import Device


def _tiny_model(path):
    w = np.ones((8, 8), dtype=np.int8)
    graph = Graph(nodes=(Node(op="matmul", inputs=("a", "w"), output="c"),))
    cmds = [Command(op="LOAD", address=0x1000, size=64),
            Command(op="LOAD", address=0x1040, size=64),
            Command(op="MATMUL", m=8, n=8, k=8),
            Command(op="STORE", address=0x1080, size=256)]
    write_model(path, graph, {"w": w}, cmds)


def test_reset_closes_device(tmp_path):
    path = str(tmp_path / "tiny.bin")
    _tiny_model(path)
    device = Device()
    model = device.load_model(path)
    x = np.ones((8, 8), dtype=np.int8)
    model.predict(x)
    device.reset()
    with pytest.raises(ValueError):
        device.load_model(path)
    with pytest.raises(ValueError):
        device.info()
    with pytest.raises(ValueError):
        device.get_stats()
    with pytest.raises(ValueError):
        device.benchmark(model, x)
    with pytest.raises(ValueError):
        device.reset()
    # An already-loaded model carries its own data and keeps predicting.
    model.predict(x)


def test_reopen_after_reset(tmp_path):
    path = str(tmp_path / "tiny.bin")
    _tiny_model(path)
    device = Device()
    device.reset()
    device.open()
    model = device.load_model(path)
    assert device.info()["device"] == "edge-npu-sim"
    model.predict(np.ones((8, 8), dtype=np.int8))
