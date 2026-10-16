"""Load trusted research checkpoints across Windows and POSIX hosts."""

from __future__ import annotations

import pickle
import types
from pathlib import PurePosixPath, PureWindowsPath

import torch


class _PortableUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # Archived argparse metadata may contain concrete paths from another OS.
        if module == "pathlib" and name in ("PosixPath", "WindowsPath"):
            return PurePosixPath if name == "PosixPath" else PureWindowsPath
        return super().find_class(module, name)


def load_checkpoint(path, *, map_location="cpu"):
    """Load a locally trusted checkpoint without modifying global pathlib classes."""
    portable_pickle = types.ModuleType("portable_pickle")
    portable_pickle.Unpickler = _PortableUnpickler
    portable_pickle.load = pickle.load
    portable_pickle.loads = pickle.loads
    return torch.load(
        path,
        map_location=map_location,
        pickle_module=portable_pickle,
        weights_only=False,
    )
