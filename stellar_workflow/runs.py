"""Stage-owned outputs, portable manifests and transactional derived reports."""
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

from filelock import FileLock, Timeout


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, values):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(values, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


class Run:
    def __init__(self, path):
        self.path = Path(path).resolve()
        self.path.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.path / "run_manifest.json"

    def read(self):
        if self.manifest_path.exists():
            return json.loads(self.manifest_path.read_text(encoding="utf-8"))
        return {"schema_version": 1, "stages": {}, "artifacts": {}}

    def resolve(self, relative):
        result = (self.path / relative).resolve()
        if not result.is_relative_to(self.path) or result == self.path:
            raise ValueError("An artifact path must remain within its run directory.")
        return result

    @contextmanager
    def stage(self, name, outputs):
        paths = [self.resolve(relative) for relative in outputs]
        try:
            lock = FileLock(self.path / ".stage.lock", timeout=0)
            lock.acquire()
        except Timeout as error:
            raise RuntimeError("Another stage is writing this run.") from error
        try:
            collisions = [p.relative_to(self.path).as_posix() for p in paths if p.exists()]
            if collisions:
                raise FileExistsError(f"Preserve existing stage outputs; use a fresh run: {collisions}")
            manifest = self.read()
            manifest["stages"][name] = {"status": "running"}
            write_json(self.manifest_path, manifest)
            try:
                yield self
                for p in paths:
                    if not p.exists():
                        raise RuntimeError(f"Stage did not produce its required output: {p.name}")
                    for artifact in ([p] if p.is_file() else sorted(q for q in p.rglob("*") if q.is_file())):
                        manifest["artifacts"][artifact.relative_to(self.path).as_posix()] = {
                            "sha256": sha256(artifact), "bytes": artifact.stat().st_size,
                        }
                manifest["stages"][name] = {"status": "complete"}
            except BaseException as error:
                manifest["stages"][name] = {"status": "failed", "error": str(error)}
                write_json(self.manifest_path, manifest)
                raise
            write_json(self.manifest_path, manifest)
        finally:
            lock.release()


def check_derived_output(path, *, replace=False):
    """Reject collisions before reports or inference start expensive computation."""
    path = Path(path).resolve()
    if path.exists():
        protected = (path / "run_manifest.json").exists() or any(
            any(path.rglob(pattern)) for pattern in ("*.keras", "history.csv", "*.h5", "*.pt", "*.joblib", "test_predictions.npz"))
        if protected:
            raise ValueError("Cannot replace a directory containing training artifacts.")
        if any(path.iterdir()) and not replace:
            raise FileExistsError(f"Derived output exists: {path}; use --output-dir or --replace-derived.")
    return path


@contextmanager
def derived_output(path, *, replace=False):
    """Render transactionally; existing models/histories are never replacement targets."""
    path = check_derived_output(path, replace=replace)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=path.name + ".partial-", dir=path.parent))
    backup = None
    try:
        yield temporary
        if path.exists():
            backup = Path(tempfile.mkdtemp(prefix=path.name + ".previous-", dir=path.parent))
            backup.rmdir()
            path.rename(backup)
        try:
            temporary.rename(path)
        except BaseException:
            if backup is not None:
                backup.rename(path)
            raise
        if backup is not None:
            assert backup.parent == path.parent and not backup.is_symlink()
            shutil.rmtree(backup)
    finally:
        if temporary.exists():
            assert temporary.parent == path.parent and not temporary.is_symlink()
            shutil.rmtree(temporary)
