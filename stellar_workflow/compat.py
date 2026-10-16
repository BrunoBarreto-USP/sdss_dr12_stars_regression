"""Compatibility names, with one implementation behind every public command."""
import sys
from pathlib import Path


def legacy_main(command, argv=None):
    from .cli import main
    values = list(sys.argv[1:] if argv is None else argv)
    aliases = {"--out-dir": "--run-dir" if command == "train" else "--output-dir"}
    values = [aliases.get(value, value) for value in values]
    if "--summarize-only" in values:
        import argparse
        from .evaluation import ablation_reports
        from .runs import Run
        parser = argparse.ArgumentParser()
        parser.add_argument("--run-dir", type=Path, required=True)
        parser.add_argument("--summarize-only", action="store_true")
        parser.add_argument("--n-boot", type=int, default=10000)
        parser.add_argument("--output-dir", type=Path)
        parser.add_argument("--replace-derived", action="store_true")
        args = parser.parse_args(values)
        return ablation_reports(Run(args.run_dir), n_boot=args.n_boot, output_dir=args.output_dir, replace=args.replace_derived)
    # These historical options selected the same source tree, not numerical behavior.
    for obsolete in ("--project-root",):
        if obsolete in values:
            i = values.index(obsolete)
            del values[i:i+2]
    if command == "audit" and "--paper-dir" in values:
        i = values.index("--paper-dir")
        del values[i:i+2]
    return main([command, *values])
