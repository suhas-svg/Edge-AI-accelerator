import csv
import logging

import numpy as np
import pytest

from compiler.binary import write_model
from compiler.graph import Graph, Node
from python.edge_npu.cli import main
from python.edge_npu.commands import Command


def _tiny_model(path):
    w = np.ones((8, 8), dtype=np.int8)
    graph = Graph(nodes=(Node(op="matmul", inputs=("a", "w"), output="c"),))
    cmds = [Command(op="LOAD", address=0x1000, size=64),
            Command(op="LOAD", address=0x1040, size=64),
            Command(op="MATMUL", m=8, n=8, k=8),
            Command(op="STORE", address=0x1080, size=256)]
    write_model(path, graph, {"w": w}, cmds)


def test_predict_writes_bit_exact_output(tmp_path, capsys):
    model = str(tmp_path / "tiny.bin")
    _tiny_model(model)
    inp = str(tmp_path / "in.npy")
    np.save(inp, np.ones((8, 8), dtype=np.int8))
    out = str(tmp_path / "out.npy")
    assert main(["predict", model, inp, "--out", out]) == 0
    text = capsys.readouterr().out
    assert "cycles=8" in text
    np.testing.assert_array_equal(
        np.load(out), np.full((8, 8), 8, dtype=np.int32))


def test_benchmark_prints_median_and_cycles(tmp_path, capsys):
    model = str(tmp_path / "tiny.bin")
    _tiny_model(model)
    inp = str(tmp_path / "in.npy")
    np.save(inp, np.ones((8, 8), dtype=np.int8))
    assert main(["benchmark", model, inp, "--repeats", "2"]) == 0
    text = capsys.readouterr().out
    assert "median_ms=" in text
    assert "cycles=8" in text


def test_dashboard_prints_results_table(tmp_path, capsys):
    results = str(tmp_path / "results.csv")
    with open(results, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["case", "fp32_ms", "int8_ms",
                                          "edgenpu_sim_ms", "edgenpu_cycles",
                                          "busy_cycles", "dma_bytes",
                                          "command_counts"])
        w.writeheader()
        w.writerow({"case": "tiny-mm", "fp32_ms": "0.01", "int8_ms": "0.02",
                    "edgenpu_sim_ms": "0.05", "edgenpu_cycles": "8",
                    "busy_cycles": "8", "dma_bytes": "384",
                    "command_counts": "LOAD:2;MATMUL:1;STORE:1"})
    assert main(["dashboard", "--results", results]) == 0
    assert "tiny-mm" in capsys.readouterr().out


def test_info_prints_device(tmp_path, capsys):
    assert main(["info"]) == 0
    assert "edge-npu-sim" in capsys.readouterr().out


def test_load_model_emits_log_record(tmp_path, caplog):
    from python.edge_npu.sdk import Device
    model = str(tmp_path / "tiny.bin")
    _tiny_model(model)
    with caplog.at_level(logging.INFO, logger="edge_npu"):
        Device().load_model(model)
    assert any("load_model" in r.message for r in caplog.records)
