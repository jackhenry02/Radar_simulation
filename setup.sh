#!/bin/bash

# Exit if any command fails
#set -e
#deactivate

# Create virtual environment
python -m venv .Radar_venv

# Activate the virtual environment
source .Radar_venv/bin/activate

# Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Add Jupyter kernel
# python -m ipykernel install --user --name=.Radar_venv --display-name "Python (.Radar_venv)"

#nbstripout --install

# To run, use - source setup.sh - in the command terminal