#!/bin/bash
#BSUB -J QM9_Model_Run
#BSUB -o trained_models/QM9/QM9_Model_Run%J.out
#BSUB -e trained_models/QM9/QM9_Model_Run%J.err   
#BSUB -q gpua100
#BSUB -W 72:00
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "rusage[mem=8GB]"
#BSUB -n 4
### -- specify that the cores must be on the same host -- 
#BSUB -R "span[hosts=1]"




module load python  # Load Python if needed (adjust based on your HPC system)
source /work3/s203788/Master_Project_2026/EquivariantTransformerMPNN4QuantumComputations/env/master_env/bin/activate # Activate your virtual environment
python train_qm9v2.py

#python DataVisualization.py --- IGNORE ---