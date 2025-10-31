#!/bin/bash

# This script downloads the required datasets from Kaggle.
# It requires the Kaggle API to be installed and configured
# (i.e., `kaggle.json` in `~/.kaggle/`).

echo "Installing Kaggle API..."
pip install -q kaggle

echo "Creating data directories..."
mkdir -p data/raw
mkdir -p data/processed
mkdir -p outputs/models
mkdir -p outputs/plots
mkdir -p logs

echo "Downloading Persian WikiText dataset..."
kaggle datasets download miladfa7/persian-wikipedia-dataset -f Persian-WikiText-1.txt -p data/raw/
unzip -q data/raw/Persian-WikiText-1.txt.zip -d data/raw/
rm data/raw/Persian-WikiText-1.txt.zip

echo "Downloading Persian Stop Words..."
kaggle datasets download alioraji/persian-stop-words -f Persian_Stop_Words.txt -p data/raw/

echo "Data download complete."
