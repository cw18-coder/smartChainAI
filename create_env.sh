#!/bin/bash
# Script to create a Python 3.12 virtual environment using uv

set -e

# Name of the virtual environment directory
ENV_DIR=".venv"

# Create the virtual environment with Python 3.12 using uv
uv venv --python=3.12 "$ENV_DIR"

echo "Virtual environment created at $ENV_DIR using Python 3.12."
echo "To activate, run: source $ENV_DIR/bin/activate"

# install necessary packages using the requirements.txt file
if [ -f "requirements.txt" ]; then
    source "$ENV_DIR/bin/activate"
    uv pip install -r requirements.txt
    echo "Installed packages from requirements.txt."
else
    echo "No requirements.txt file found. Skipping package installation."
fi