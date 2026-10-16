"""Execute the cleared guide without saving cell outputs into tracked source."""
from pathlib import Path
import sys


def main():
    import nbformat
    from nbclient import NotebookClient
    from jupyter_client.kernelspec import KernelSpecManager
    import tempfile
    import json
    root = Path(__file__).resolve().parents[1]
    notebook = nbformat.read(root / "orchestration.ipynb", as_version=4)
    # Use this Python environment even if a global python3 kernel points elsewhere.
    with tempfile.TemporaryDirectory(prefix="stellar-kernel-") as temporary:
        folder = Path(temporary) / "stellar-check"
        folder.mkdir()
        (folder / "kernel.json").write_text(json.dumps({
            "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
            "display_name": "Stellar check", "language": "python"}), encoding="utf-8")
        manager = KernelSpecManager(kernel_dirs=[temporary])
        client = NotebookClient(notebook, timeout=240, kernel_name="stellar-check",
                                resources={"metadata": {"path": str(root)}})
        client.km = client.create_kernel_manager()
        client.km.kernel_spec_manager = manager
        client.execute()
    print(f"Notebook executed: {len(notebook.cells)} cells; tracked source stays cleared.")


if __name__ == "__main__":
    main()
