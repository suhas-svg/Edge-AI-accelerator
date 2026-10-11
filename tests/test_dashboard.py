from benchmarks.dashboard import render


def _rows():
    return [
        {"case": "matmul-64", "fp32_ms": "0.016", "int8_ms": "0.175",
         "edgenpu_sim_ms": "3.15", "edgenpu_cycles": "4096",
         "busy_cycles": "4096", "dma_bytes": "24576",
         "command_counts": "LOAD:2;MATMUL:1;STORE:1"},
        {"case": "conv-chain-8ch", "fp32_ms": "", "int8_ms": "4.129",
         "edgenpu_sim_ms": "16.717", "edgenpu_cycles": "576",
         "busy_cycles": "576", "dma_bytes": "1888",
         "command_counts": "LOAD:2;CONV2D:1;MAX_POOL:1;STORE:1"},
    ]


def test_render_reports_latency_throughput_and_speedup():
    out = render(_rows())
    assert "matmul-64" in out
    assert "3.150 ms" in out
    assert "317.5 inf/s" in out
    assert "0.06x" in out


def test_render_skips_blank_reference_cells_honestly():
    out = render(_rows())
    assert "conv-chain-8ch" in out
    assert "N/A" in out
    assert "Memory bandwidth: N/A (no clock yet)" in out
    assert "FPGA utilization: N/A (partner synthesis pending)" in out


def test_render_reports_cycles_and_dma():
    out = render(_rows())
    assert "4096 cyc" in out
    assert "24576 B" in out
