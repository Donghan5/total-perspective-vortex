#!/bin/bash

set -e

echo "Import data from physionet.org"

wget -r -N -c -np https://physionet.org/files/eegmmidb/1.0.0/