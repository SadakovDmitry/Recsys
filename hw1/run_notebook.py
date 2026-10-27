"""Execute the entire notebook and export a convenient HTML view."""
import os
from pathlib import Path
import sys
import time
import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter


def main():
    root = Path(__file__).resolve().parent
    os.chdir(root)
    os.environ.setdefault("MPLCONFIGDIR", str(root / ".cache" / "matplotlib"))
    os.environ.setdefault("JUPYTER_PLATFORM_DIRS", "1")
    os.environ.setdefault("JUPYTER_RUNTIME_DIR", str(root / ".cache" / "jupyter"))
    os.environ.setdefault("IPYTHONDIR", str(root / ".cache" / "ipython"))
    path = root / "experiments.ipynb"
    notebook = nbformat.read(path, as_version=4)
    started = time.perf_counter()
    def save_progress(cell, cell_index, **kwargs):
        nbformat.write(notebook, path)
        print(f"Cell {cell_index + 1}/{len(notebook.cells)} complete; "
              f"elapsed {(time.perf_counter()-started)/60:.1f} min", flush=True)
    client = NotebookClient(notebook, timeout=3600, kernel_name="python3",
                            resources={"metadata": {"path": str(root)}},
                            on_cell_executed=save_progress)
    # Use the interpreter whose dependencies were installed, not an unrelated kernel.
    from jupyter_client import KernelManager
    client.km = KernelManager(kernel_name="python3")
    client.km.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
    try:
        client.execute()
    finally:
        nbformat.write(notebook, path)
    html, _ = HTMLExporter().from_notebook_node(notebook)
    path.with_suffix(".html").write_text(html, encoding="utf-8")
    print(f"Done in {(time.perf_counter()-started)/60:.1f} min: {path.name}, experiments.html", flush=True)


if __name__ == "__main__":
    main()
