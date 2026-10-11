"""EdgeNPU command line: predict, benchmark, dashboard, info."""
from __future__ import annotations
import argparse
import csv
import logging
import sys

import numpy as np

from benchmarks.dashboard import RESULTS as DEFAULT_RESULTS
from benchmarks.dashboard import render
from python.edge_npu.sdk import Device

log = logging.getLogger("edge_npu")


def _load_input(path: str) -> np.ndarray:
    arr = np.load(path)
    if isinstance(arr, np.lib.npyio.NpzFile):
        key = "x" if "x" in arr.files else arr.files[0]
        return np.ascontiguousarray(arr[key])
    return np.ascontiguousarray(arr)


def _cmd_predict(args: argparse.Namespace) -> int:
    model = Device().load_model(args.model)
    out = model.predict(_load_input(args.input))
    if args.out:
        np.save(args.out, out)
    stats = model.get_stats()
    print(f"shape={out.shape} dtype={out.dtype} cycles={stats['cycles']}")
    return 0


def _cmd_benchmark(args: argparse.Namespace) -> int:
    model = Device().load_model(args.model)
    bench = Device().benchmark(model, _load_input(args.input),
                               repeats=args.repeats)
    print(f"median_ms={bench['median_ms']:.3f} repeats={bench['repeats']} "
          f"cycles={model.get_stats()['cycles']}")
    return 0


def _cmd_dashboard(args: argparse.Namespace) -> int:
    with open(args.results, newline="") as f:
        print(render(list(csv.DictReader(f))), end="")
    return 0


def _cmd_info(args: argparse.Namespace) -> int:
    print(Device().info())
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="edge-npu", description="EdgeNPU CLI")
    p.add_argument("-v", "--verbose", action="store_true",
                   help="log INFO instead of WARNING")
    sub = p.add_subparsers(dest="cmd", required=True)
    pp = sub.add_parser("predict", help="run inference, optionally save output")
    pp.add_argument("model")
    pp.add_argument("input", help=".npy, or .npz (key 'x' else first array)")
    pp.add_argument("--out", default=None, help="save output as .npy")
    pp.set_defaults(func=_cmd_predict)
    bp = sub.add_parser("benchmark", help="median latency over repeats")
    bp.add_argument("model")
    bp.add_argument("input")
    bp.add_argument("--repeats", type=int, default=5)
    bp.set_defaults(func=_cmd_benchmark)
    dp = sub.add_parser("dashboard", help="render benchmarks/results.csv")
    dp.add_argument("--results", default=DEFAULT_RESULTS)
    dp.set_defaults(func=_cmd_dashboard)
    ip = sub.add_parser("info", help="print device info")
    ip.set_defaults(func=_cmd_info)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(name)s %(levelname)s %(message)s")
    try:
        return args.func(args)
    except (ValueError, OSError) as e:
        print(f"edge-npu: error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
