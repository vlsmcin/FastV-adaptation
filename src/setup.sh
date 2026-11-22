#!/usr/bin/env bash
set -euo pipefail

# Modern FastV setup (compatible with Python 3.10-3.13, PyTorch >=2.4, Transformers >=4.45 / 5.x)
# Does NOT overwrite upstream transformers with legacy 4.31/4.39 forks.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

python -m pip install -U pip
python -m pip install -r "${REPO_ROOT}/requirements.txt"
