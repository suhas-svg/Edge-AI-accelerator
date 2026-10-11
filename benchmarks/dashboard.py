"""Benchmark dashboard: render benchmarks/results.csv as a text table.

Wall-clock medians are human snapshots, never gates. Memory bandwidth
and FPGA utilization print N/A until the partner's clock and synthesis
numbers arrive — reported, not invented.
"""
from __future__ import annotations
import csv
import os

RESULTS = os.path.join(os.path.dirname(__file__), "results.csv")

FOOTER = ("\nMemory bandwidth: N/A (no clock yet)\n"
          "FPGA utilization: N/A (partner synthesis pending)")


def _fnum(v: str) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def render(rows: list[dict]) -> str:
    lines = ["case | sim latency | throughput | speedup vs int8 | "
             "cycles | dma | counts | fp32 ref"]
    for r in rows:
        sim = _fnum(r.get("edgenpu_sim_ms", ""))
        int8 = _fnum(r.get("int8_ms", ""))
        fp32 = _fnum(r.get("fp32_ms", ""))
        lat = f"{sim:.3f} ms" if sim else "N/A"
        tput = f"{1000 / sim:.1f} inf/s" if sim else "N/A"
        if sim and int8:
            speedup = f"{int8 / sim:.2f}x"
        else:
            speedup = "N/A"
        lines.append(
            f"{r.get('case', '?')} | {lat} | {tput} | {speedup} | "
            f"{r.get('edgenpu_cycles', '?')} cyc | "
            f"{r.get('dma_bytes', '?')} B | "
            f"{r.get('command_counts', '?')} | "
            f"{f'{fp32:.3f} ms' if fp32 else 'N/A'}")
    return "\n".join(lines) + "\n" + FOOTER + "\n"


def main() -> None:
    with open(RESULTS, newline="") as f:
        print(render(list(csv.DictReader(f))), end="")


if __name__ == "__main__":
    main()
