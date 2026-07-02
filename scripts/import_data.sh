#!/bin/bash

set -e
echo "move to root directory"
cd "$(dirname "$0")/.."

if [ -d "physionet.org" ]; then
    echo "physionet.org already exists. Skipping download."
    exit 0
fi

echo "Downloading data from physionet.org"
wget -r -N -c -np https://physionet.org/files/eegmmidb/1.0.0/