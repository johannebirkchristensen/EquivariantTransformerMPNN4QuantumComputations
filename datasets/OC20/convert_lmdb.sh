#!/bin/bash
#BSUB -J convert_lmdb
#BSUB -o logs/convert_lmdb%J.out
#BSUB -e logs/convert_lmdb%J.err   
#BSUB -q hpc
#BSUB -W 24:00
#BSUB -R "rusage[mem=16GB]"
#BSUB -n 4
#BSUB -R "span[hosts=1]"

mkdir -p logs

echo "================================================================"
echo "Converting OC20 to LMDB format"
echo "================================================================"
echo "Started: $(date)"

PYTHON="/work3/s203788/miniconda3/bin/python"

# Path to your conversion script
CONVERT_SCRIPT="/work3/s203788/Master_Project_2026/EquivariantTransformerMPNN4QuantumComputations/datasets/OC20/convert_to_lmdb.py"

# Check if script exists
if [ ! -f "$CONVERT_SCRIPT" ]; then
    echo "ERROR: Conversion script not found at $CONVERT_SCRIPT"
    exit 1
fi

echo "Running: $PYTHON $CONVERT_SCRIPT"
$PYTHON "$CONVERT_SCRIPT"

echo "================================================================"
echo "Finished: $(date)"
echo "Exit code: $?"
echo "================================================================"