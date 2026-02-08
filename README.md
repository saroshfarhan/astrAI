ad-astrAI
======

A multi agent AI tool for detecting elements and molecules from spectral and image data from galaxies far far away.


Folder layout
-------------
- `experiments/` — notebooks, prototypes, and other exploratory work combining data and agents.
- `agent/` — production-ready or reusable agent code lives here.
- `data/` — datasets and related assets reside here (keep raw data out of version control as needed).

Setup
-----
1) Install Python 3.11 (e.g., `pyenv install 3.11.0`), matching `.python-version`.
2) Install `uv` for dependency management:
   - macOS/Linux: `curl -LsSf https://astral.sh/uv/install.sh | sh`
   - Windows (PowerShell): `iwr https://astral.sh/uv/install.ps1 | iex`
3) Clone and install deps:
   - `git clone <repo-url> && cd astrAI`
   - `uv sync`  # creates a virtualenv and installs from `uv.lock`

Running
-------
- `uv run python main.py` — entry point placeholder for now.
- Add datasets under `data/` and keep large/raw files out of Git (use `.gitignore`/DVC as needed).
