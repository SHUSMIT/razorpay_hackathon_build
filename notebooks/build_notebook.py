#!/usr/bin/env python
"""Convert notebooks/01_eda.py (percent-format) into 01_eda.ipynb.

Avoids a jupytext dependency -- the mapping we need is small.
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def parse(src: str) -> list[dict]:
    cells, kind, buf = [], "code", []

    def flush():
        text = "\n".join(buf).strip("\n")
        if not text.strip():
            return
        if kind == "markdown":
            body = "\n".join(
                ln[2:] if ln.startswith("# ") else ln[1:] if ln.startswith("#") else ln
                for ln in text.splitlines()
            )
            cells.append({"cell_type": "markdown", "metadata": {},
                          "source": body.splitlines(keepends=True)})
        else:
            cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                          "outputs": [], "source": text.splitlines(keepends=True)})

    for line in src.splitlines():
        if line.startswith("# %% [markdown]"):
            flush()
            kind, buf = "markdown", []
        elif line.startswith("# %%"):
            flush()
            kind, buf = "code", []
        else:
            buf.append(line)
    flush()
    return cells


def main() -> None:
    src = (HERE / "01_eda.py").read_text(encoding="utf-8")
    # Drop the module docstring; it is scaffolding, not notebook content.
    body = src.split('"""', 2)[-1] if src.startswith('"""') else src
    nb = {
        "cells": parse(body),
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    out = HERE / "01_eda.ipynb"
    out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print(f"wrote {out} ({len(nb['cells'])} cells)")


if __name__ == "__main__":
    main()
