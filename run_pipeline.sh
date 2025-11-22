#!/usr/bin/env bash
set -e

# run_pipeline.sh
# This wrapper script activates the project’s virtual environment and runs the
# pipeline with the provided arguments. It makes it convenient to execute
# the signal discovery pipeline with a single command once you have your
# operational data and specify the relevant tickers and dates.

# Check for the virtual environment
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "Virtual environment not found. Please run init_project.sh first."
    exit 1
fi

# Activate the virtual environment
source "$VENV_DIR/bin/activate"

# Pass all arguments through to the pipeline module
python -m signal_discovery.src.pipeline "$@"
