from demo.demo_inference import render_dashboard


def test_dashboard_renders_key_lines():
    stats = {"cycles": 4096, "mac_utilization": 1.0}
    out = render_dashboard(0.175, 3.15, stats, 0.0012, 0)
    assert "EdgeNPU Dashboard" in out
    assert "DemoMatmul 64x64" in out
    assert "4096" in out
    assert "max diff 0 (exact)" in out


def test_dashboard_survives_stock_windows_console():
    stats = {"cycles": 4096, "mac_utilization": 1.0}
    out = render_dashboard(0.175, 3.15, stats, 0.0012, 0)
    out.encode("cp1252")
