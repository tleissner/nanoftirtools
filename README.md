[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)

**Author:** Till Leissner — SDU (University of Southern Denmark)  
**License:** GPL-3.0-or-later


# nanoftirtools (uv project)

Reproducible dev environment using **uv** for the `nanoftirtools` library.

## Quick start

> Install uv (once): https://docs.astral.sh/uv/getting-started/

```bash
# 1) Choose and install a Python (once)
uv python install 3.12

# 2) Create & sync the virtual env from pyproject
uv venv
uv sync

# 3) Make this project editable (so changes reflect immediately)
uv pip install -e .

# 4) Optional extras (GWY reader)
uv pip install -e .[gwy]

# 5) Jupyter kernel (optional)
uv run python -m ipykernel install --user --name nanoftirtools --display-name "Python (nanoftirtools)"
```

