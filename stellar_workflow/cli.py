"""Small public command interface; optional numerical backends load on demand."""
import argparse
from pathlib import Path

from .config import ROOT


def parser():
    p = argparse.ArgumentParser(prog="python -m stellar_workflow", description="Standalone SDSS stellar-regression workflow.")
    sub = p.add_subparsers(dest="command", required=True)
    for command in ("prepare-data", "train", "tune", "baselines", "ablations", "evaluate", "figures", "audit", "reproduce-paper", "smoke"):
        s = sub.add_parser(command)
        s.add_argument("--run-dir", type=Path, required=command not in ("prepare-data", "smoke"))
        s.add_argument("--data-dir", type=Path, default=ROOT / "data")
        s.add_argument("--config", type=Path)
        s.add_argument("--offline", action="store_true")
        s.add_argument("--row-limit", type=int)
        s.add_argument("--output-dir", type=Path)
        s.add_argument("--replace-derived", action="store_true")
        if command in ("train", "tune", "smoke", "baselines", "ablations"):
            s.add_argument("--epochs", type=int)
            s.add_argument("--batch-size", type=int)
            s.add_argument("--threads", type=int)
            s.add_argument("--seed", type=int)
        if command == "tune":
            s.add_argument("--max-trials", type=int, default=100)
            s.add_argument("--search-epochs", type=int, default=30)
        if command == "baselines":
            s.add_argument("--models", nargs="+", choices=("ols", "ridge", "cnn", "dnn", "transformer"), default=["ols", "ridge", "cnn", "dnn"])
            s.add_argument("--allow-cpu", action="store_true")
            s.add_argument("--pretrain-epochs", type=int, default=20)
        if command == "ablations":
            s.add_argument("--variants", nargs="+", default=["all"])
            s.add_argument("--low-memory", action="store_true")
        if command in ("evaluate", "audit"):
            s.add_argument("--comparators", nargs="*", choices=("ols", "ridge", "cnn", "dnn", "transformer"), default=[])
        if command in ("evaluate", "reproduce-paper", "smoke", "figures"):
            s.add_argument("--n-boot", type=int, default=10000)
        if command == "audit":
            s.add_argument("--verify-transformer", action="store_true")
        if command in ("figures", "reproduce-paper"):
            s.add_argument("--paper-dir", type=Path)
            s.add_argument("--paper-reference", action="store_true")
        if command == "figures":
            s.add_argument("--only-training", action="store_true")
        if command == "reproduce-paper":
            s.add_argument("--bundle", type=Path, help="Verify a locally staged release bundle instead of downloading.")
        if command == "smoke":
            s.add_argument("--online", action="store_true")
            s.add_argument("--extended", action="store_true")
    return p


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    if args.row_limit is not None and args.row_limit < 1:
        p.error("--row-limit must be positive.")
    if getattr(args, "online", False) and args.offline:
        p.error("--online and --offline are mutually exclusive.")
    try:
        from .pipeline import dispatch
        dispatch(args)
    except (OSError, ValueError, RuntimeError, ImportError) as error:
        p.exit(2, f"error: {error}\n")
