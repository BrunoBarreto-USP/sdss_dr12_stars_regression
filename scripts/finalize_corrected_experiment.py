"""Compatibility entry point for python -m stellar_workflow evaluate."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from stellar_workflow.compat import legacy_main


def main():
    return legacy_main("evaluate")


if __name__ == "__main__":
    main()
