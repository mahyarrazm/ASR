#!/usr/bin/env python3
"""
Build ASR_pipeline.ipynb from asr_publication_pipeline.py.

The script is the single source of truth. Cells are split at the top-level
banner comments so the notebook can be run section by section in Colab while
staying byte-identical to the script in content.
"""

import json
import re
from pathlib import Path

SOURCE = Path("asr_publication_pipeline.py")
NOTEBOOK = Path("ASR_pipeline.ipynb")

INTRO = """# Machine-learning prediction of ASR mortar-bar expansion

Model comparison, input reduction, validation, and interpretation for
alkali-silica reaction expansion of mortar bars.

**Before running**

1. Select a GPU runtime: *Runtime -> Change runtime type -> GPU*.
2. TabPFN weights are gated. Accept the licence at
   [huggingface.co/Prior-Labs](https://huggingface.co/Prior-Labs) and add your
   token as a Colab secret named `TABPFN_TOKEN` (key icon, notebook access on).
   To run everything else without TabPFN, set `REQUIRE_TABPFN = False` in the
   configuration cell.
3. Upload the dataset CSV to `/content/` and set `CSV_PATH`.
4. *Runtime -> Run all*.

Every figure is rendered inline and written to `ASR_publication_outputs/` as
600-dpi PNG, vector PDF, and editable SVG, alongside CSV tables, a run
manifest, and a checksummed artefact inventory. A verified ZIP of everything is
offered for download at the end.

All cross-validation folds and the locked holdout are **mixture-disjoint**: no
mixture contributes rows to both a training set and the set used to score it.
The conventional random-row protocol is also run and reported so the optimism
it introduces is visible.
"""

BANNER = re.compile(r"^# =+\n(# .*\n)+# =+\n", re.MULTILINE)


def build():
    text = SOURCE.read_text()

    # Separate the module docstring so it becomes the first code cell intact.
    match = re.search(r'^(#!.*\n(?:# -\*-.*\n)?)("""(?:.|\n)*?""")\n', text)
    header = match.group(0) if match else ""
    body = text[len(header):]

    boundaries = [m.start() for m in BANNER.finditer(body)]
    boundaries = [0] + boundaries + [len(body)]
    chunks = [body[start:end].strip("\n")
              for start, end in zip(boundaries[:-1], boundaries[1:])]
    chunks = [chunk for chunk in chunks if chunk.strip()]
    if header.strip():
        chunks.insert(0, header.strip("\n"))

    cells = [{
        "cell_type": "markdown", "metadata": {},
        "source": INTRO.splitlines(keepends=True),
    }]
    for chunk in chunks:
        cells.append({
            "cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": chunk.splitlines(keepends=True),
        })

    notebook = {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": [], "toc_visible": True},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4, "nbformat_minor": 0,
    }
    NOTEBOOK.write_text(json.dumps(notebook, indent=1))
    print(f"wrote {NOTEBOOK} with {len(cells)} cells "
          f"({sum(c['cell_type'] == 'code' for c in cells)} code)")


if __name__ == "__main__":
    build()
