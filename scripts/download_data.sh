#!/bin/bash
set -e

DATA_DIR="app/data/training_data"

if [ -d "$DATA_DIR" ] && [ "$(ls -A $DATA_DIR)" ]; then
    echo "Data already exists, skipping download."
    exit 0
fi

if [ ! -f "$HOME/.kaggle/kaggle.json" ]; then
    echo "ERROR: Kaggle API token not found at ~/.kaggle/kaggle.json"
    echo "Please follow the setup steps in the README under 'Dataset Setup'."
    exit 1
fi

echo "Downloading dataset..."
kaggle datasets download -d alistairking/recyclable-and-household-waste-classification -p "$DATA_DIR" --unzip
echo "Done."