"""Compatibility builder for the shared verified compact dataset implementation."""
from pathlib import Path
import argparse
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
DEFAULT_OUTPUT = PROJECT_ROOT / "data/sdss_dr12_processed_flux_benchmark.npz"


def build_final_dataset(*, output_path=DEFAULT_OUTPUT, row_limit=None, normalize_flux=False,
                        overwrite=False, data_dir=None, offline=False):
    from stellar_workflow.datasets import prepare_public, build_compact
    root = Path(data_dir or PROJECT_ROOT / "data")
    prepared = prepare_public(root, row_limit=row_limit, offline=offline)
    output_path = Path(output_path)
    if output_path == DEFAULT_OUTPUT:
        output_path = prepared
    if output_path.resolve() == prepared.resolve() and not normalize_flux:
        return prepared
    return build_compact(prepared.parent, source_kind="public_sample" if row_limit else "public",
                         row_limit=row_limit, normalize_flux=normalize_flux,
                         output_path=output_path, overwrite=overwrite)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--row-limit", type=int)
    parser.add_argument("--normalize-flux", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    print(build_final_dataset(output_path=args.out, data_dir=args.data_dir, row_limit=args.row_limit,
                              normalize_flux=args.normalize_flux, overwrite=args.overwrite, offline=args.offline))


if __name__ == "__main__":
    main()
