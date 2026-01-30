#!/bin/bash
#BSUB -J model_run
#BSUB -o model_run%J.out
#BSUB -e model_run%J.err   
#BSUB -q gpua100
#BSUB -W 2
#BSUB -R "rusage[mem=4GB]"
#BSUB -n 4
### -- specify that the cores must be on the same host -- 
#BSUB -R "span[hosts=1]"




module load python  # Load Python if needed (adjust based on your HPC system)
source /work3/s203788/Master_Project_2026/EquivariantTransformerMPNN4QuantumComputations/env/master_env/bin/activate # Activate your virtual environment
python modellen.py

#python DataVisualization.py --- IGNORE ---