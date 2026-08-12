#!/usr/bin/env bash
# Run any part of the analysis without remembering the setup steps.
#
#   ./run.sh                 scope comparison (default)
#   ./run.sh pipeline        full figure and table package
#   ./run.sh search          model and hyperparameter search
#   ./run.sh structure       problem-formulation comparison
#
# Works from any directory, creates the virtual environment on first use, and
# installs dependencies if they are missing. Point it at a different dataset
# with ASR_CSV_PATH, and choose a subset of the data with ASR_SCOPE:
#
#   ASR_SCOPE=c1260 ./run.sh pipeline
#   ASR_CSV_PATH=/path/to/data.csv ./run.sh

set -euo pipefail
cd "$(dirname "$0")"

if [ ! -d .venv ]; then
    echo "creating virtual environment..."
    python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

if ! python -c "import numpy, pandas, sklearn" 2>/dev/null; then
    echo "installing dependencies..."
    python -m pip install --quiet --upgrade pip
    python -m pip install --quiet -r requirements.txt
fi

DEFAULT_CSV="$HOME/Smart Lab/Lab/ML/Final Dataset/ASR_FinalE-ComCo.csv"
export ASR_CSV_PATH="${ASR_CSV_PATH:-$DEFAULT_CSV}"
if [ ! -f "$ASR_CSV_PATH" ]; then
    echo "dataset not found at: $ASR_CSV_PATH"
    echo "set ASR_CSV_PATH to its location, for example:"
    echo "  ASR_CSV_PATH=/path/to/ASR_FinalE-ComCo.csv $0 ${1:-}"
    exit 1
fi
echo "dataset: $ASR_CSV_PATH"
if [ "${1:-local}" = "pipeline" ]; then echo "scope  : ${ASR_SCOPE:-all}"; fi
echo

case "${1:-local}" in
    local)     shift || true; python asr_local.py --csv "$ASR_CSV_PATH" "$@" ;;
    pipeline)  python asr_publication_pipeline.py ;;
    search)    python asr_model_search.py ;;
    structure) python asr_structure.py ;;
    *) echo "usage: $0 [local|pipeline|search|structure]"; exit 1 ;;
esac
