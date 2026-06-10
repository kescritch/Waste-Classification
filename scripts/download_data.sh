#!/bin/bash
set -e

DATA_DIR="app/data/training_data"

if [ -d "$DATA_DIR" ] && [ "$(ls -A $DATA_DIR)" ]; then
    echo "Data already exists, skipping download."
    exit 0
fi

echo "Downloading dataset..."
kaggle datasets download -d kylescritchfield/sorted-waste-classification -p "$DATA_DIR" --unzip
echo "Done."