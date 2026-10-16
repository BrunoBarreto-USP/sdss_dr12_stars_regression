from pathlib import Path
import hashlib
import json
import tempfile
import unittest


class ArtifactBundleTests(unittest.TestCase):
    def test_bundle_detects_tampered_artifact(self):
        from stellar_workflow.artifacts import verify_bundle
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            content = b"checkpoint"
            (root / "model.bin").write_bytes(content)
            manifest = {"schema_version": 1, "files": {"model.bin": {
                "sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content)}}}
            (root / "artifact_manifest.json").write_text(json.dumps(manifest))
            verify_bundle(root)
            (root / "model.bin").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "checksum|size"):
                verify_bundle(root)

    def test_bundle_rejects_paths_outside_release(self):
        from stellar_workflow.artifacts import verify_bundle
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "artifact_manifest.json").write_text(json.dumps({
                "schema_version": 1, "files": {"../outside.bin": {"sha256": "0"*64, "bytes": 0}}}))
            with self.assertRaisesRegex(ValueError, "path|outside"):
                verify_bundle(root)
