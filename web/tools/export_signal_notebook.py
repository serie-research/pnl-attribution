# /// script
# requires-python = ">=3.11"
# dependencies = ["nbformat>=5", "jupytext>=1.16", "nbclient>=0.10", "ipykernel>=6.29"]
# ///
"""Export the executed signal notebook using the shared nb2mdx renderers.

Run after nbconvert execution: uv run web/tools/export_signal_notebook.py
"""
from pathlib import Path
import re
import shutil

import nbformat
from nb2mdx import fenced, output_mdx


def main():
    web = Path(__file__).resolve().parents[1]
    source = web / "notebooks/signal-attribution.ipynb"
    notebook = nbformat.read(source, as_version=4)
    groups = {"## 1.": "Optimiser", "## 2.": "Durin",
              "## 3.": "Other methods", "## 4.": "Reproduction",
              "## 5.": "Approximate Shapley"}
    body = ["{/* Generated from signal-attribution.ipynb using nb2mdx.py rendering helpers. */}\n\n",
            f"export const cells = {len(notebook.cells)};\n\n",
            "<details>\n<summary>Optimiser</summary>\n\n"]
    assets = {}
    for cell in notebook.cells:
        if cell.cell_type == "markdown":
            group = next((name for prefix, name in groups.items()
                          if cell.source.startswith(prefix)), None)
            if group and group != "Optimiser":
                body.append(f"</details>\n\n<details>\n<summary>{group}</summary>\n\n")
            body.append(re.sub(r"^#{1,3} ", "#### ", cell.source, flags=re.M) + "\n\n")
        elif cell.cell_type == "code":
            if cell.execution_count is None:
                raise ValueError("Execute all code cells before exporting")
            body.extend([fenced(cell.source, "python"),
                         output_mdx(cell.outputs, "solution", assets)])
        else:
            raise ValueError(f"Unsupported cell: {cell.cell_type}")
    body.append("</details>\n")
    if assets:
        raise ValueError("This text-only notebook unexpectedly contains image output")
    (web / "src/content/exercises/signal-attribution/solution.mdx").write_text("".join(body))
    shutil.copyfile(source, web / "public/notebooks/signal-attribution.ipynb")


if __name__ == "__main__":
    main()
