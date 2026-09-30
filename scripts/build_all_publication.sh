#!/usr/bin/env bash
set -euo pipefail

python scripts/run_publication_ablations.py
python scripts/build_publication_package.py

echo
echo "Publication package complete: outputs/publication/"
