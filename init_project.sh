#!/usr/bin/env bash
set -e

# init_project.sh
#
# This script bootstraps the development environment for the signal discovery pipeline.  It
# creates a Python virtual environment (if one does not already exist), installs
# dependencies from requirements.txt, and runs a basic diagnostics script to verify
# the setup.  Running this script repeatedly is idempotent and safe.

# Determine the root directory of the project (the location of this script)
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Create virtual environment if it doesn't already exist
if [ ! -d "$ROOT_DIR/venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv "$ROOT_DIR/venv"
fi

# Activate the virtual environment
source "$ROOT_DIR/venv/bin/activate"

# Upgrade pip itself for reliability
python -m pip install --upgrade pip

# Install required packages
echo "Installing required Python packages..."
set +e  # allow pip install to fail gracefully if offline
python -m pip install -r "$ROOT_DIR/requirements.txt"
INSTALL_EXIT_CODE=$?
set -e
if [ $INSTALL_EXIT_CODE -ne 0 ]; then
    echo "Warning: Failed to install one or more packages.  "\
         "You may need to install dependencies manually in an online environment."
fi

# Run diagnostics to verify environment setup
echo "Running diagnostics..."
python "$ROOT_DIR/diagnostics.py"

echo "Environment setup complete.  You are ready to proceed."