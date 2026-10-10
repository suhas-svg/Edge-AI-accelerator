from benchmarks.bench_matmul import conv_chain_row, matmul_row


def _keys(row):
    return {"case", "fp32_ms", "int8_ms", "edgenpu_sim_ms", "edgenpu_cycles",
            "busy_cycles", "dma_bytes", "command_counts"}


def test_matmul_row_reports_counters(tmp_path):
    row = matmul_row("tiny-mm", 8, 8, 8, str(tmp_path))
    assert set(row) == _keys(row)
    assert row["edgenpu_cycles"] == 8
    assert row["busy_cycles"] == 8
    assert row["dma_bytes"] == 64 + 64 + 256
    assert row["command_counts"] == "LOAD:2;MATMUL:1;STORE:1"
    assert row["fp32_ms"] != "" and row["int8_ms"] != ""


def test_odd_matmul_row_tiles(tmp_path):
    row = matmul_row("tiny-odd", 4, 6, 10, str(tmp_path))
    assert row["edgenpu_cycles"] == (8 * 8 * 16) // 64
    assert "MATMUL:1" in row["command_counts"].split(";")


def test_conv_chain_row_reports_counters(tmp_path):
    row = conv_chain_row(str(tmp_path))
    assert set(row) == _keys(row)
    assert row["case"] == "conv-chain-8ch"
    assert row["fp32_ms"] == ""
    assert row["edgenpu_cycles"] == (8 * 8 * 8 * 8 * 3 * 3) // 64
    assert "CONV2D:1" in row["command_counts"].split(";")
    assert "MAX_POOL:1" in row["command_counts"].split(";")
