#!/bin/bash
#BSUB -J Orginal_EquiformerV2_GATA_all2all_all_l_HTR_phi_continued_ej727i1f
#BSUB -o trained_models/MatPES/V2_out_and_error/Orginal_EquiformerV2_GATA_all2all_all_l_HTR_phi_continued_ej727i1f%J.out
#BSUB -e trained_models/MatPES/V2_out_and_error/Orginal_EquiformerV2_GATA_all2all_all_l_HTR_phi_continued_ej727i1f%J.err   
#BSUB -q gpua100
#BSUB -W 48:00
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "rusage[mem=32GB]"
#BSUB -n 6
#BSUB -R "span[hosts=1]"
echo "================================================================"
echo "Job ID: $LSB_JOBID"
echo "Started at: $(date)"
echo "Running on: $(hostname)"
echo "================================================================"
#4
#<28166947>

# Use the ACTUAL Python from conda base (where PyTorch is installed)
PYTHON_BIN="/work3/s203788/miniconda3/bin/python"

# Set VIRTUAL_ENV to match your setup
export VIRTUAL_ENV="/work3/s203788/Master_Project_2026/master_env"
export PATH="/work3/s203788/miniconda3/bin:$PATH"

# Verify
echo "================================================================"
echo "Python binary: $PYTHON_BIN"
echo "Python version: $($PYTHON_BIN --version)"

$PYTHON_BIN -c "import torch; print(f'PyTorch: {torch.__version__}')" || {
    echo "ERROR: PyTorch import failed!"
    exit 1
}
echo "================================================================"

# Check GPU
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
    echo "================================================================"
fi

# Resume training
echo "Beginning training on MatPES dataset"
# CONTINUED ONLY WITH PATHS... 5 for each type. remember to change the wandb_run_id for each one
# AND put miltuple lines in queue for each script so they run sequentially if CUDA out of memory. 
# < Orginal_EquiformerV2_GATA_all2all_phi_continued>
#$PYTHON_BIN train_model_from_checkpoint_MatPES.py  --model moreAT_gata_all2all_phi --checkpoint /work3/s203788/Master_Project_2026/EquivariantTransformerMPNN4QuantumComputations/models/trained_models/MatPES/GATAall2all/matpes_20260419_030444 --wandb_run_id 8lfwml0d # the orange

#$PYTHON_BIN train_model_from_checkpoint_MatPES.py  --model moreAT_gata_all2all_phi --checkpoint /work3/s203788/Master_Project_2026/EquivariantTransformerMPNN4QuantumComputations/models/trained_models/MatPES/GATAall2all/matpes_20260421_031834 --wandb_run_id 5dr7s0nj # the pale bluish green 
#$PYTHON_BIN train_model_from_checkpoint_MatPES.py  --model moreAT_gata_all2all_phi --checkpoint /work3/s203788/Master_Project_2026/EquivariantTransformerMPNN4QuantumComputations/models/trained_models/MatPES/GATAall2all/matpes_20260423_040027 --wandb_run_id cp0p74ac # the redish purple



# < Orginal_EquiformerV2_GATA_all2all_all_lphi_continued>
#$PYTHON_BIN train_model_from_checkpoint_MatPES.py  --model moreAT_global_phi --checkpoint /work3/s203788/Master_Project_2026/EquivariantTransformerMPNN4QuantumComputations/models/trained_models/MatPES/GATAall2all/matpes_20260419_023710 --wandb_run_id tlnfnviq # the pink
#$PYTHON_BIN train_model_from_checkpoint_MatPES.py  --model moreAT_global_phi --checkpoint /work3/s203788/Master_Project_2026/EquivariantTransformerMPNN4QuantumComputations/models/trained_models/MatPES/GATAall2all/matpes_20260421_033506 --wandb_run_id k5q17ddh # the deeper pink
#$PYTHON_BIN train_model_from_checkpoint_MatPES.py  --model moreAT_global_phi --checkpoint /work3/s203788/Master_Project_2026/EquivariantTransformerMPNN4QuantumComputations/models/trained_models/MatPES/GATAall2all/matpes_20260421_071132 --wandb_run_id 1s8rljza # the blue teal 



# < Orginal_EquiformerV2_GATA_all2all_all_l_HTR_phi_continued>
#$PYTHON_BIN train_model_from_checkpoint_MatPES.py  --model moreAT_global_htr_phi --checkpoint /work3/s203788/Master_Project_2026/EquivariantTransformerMPNN4QuantumComputations/models/trained_models/MatPES/GATAall2all/matpes_20260418_134848 --wandb_run_id ucr1qpcj # the red
#$PYTHON_BIN train_model_from_checkpoint_MatPES.py  --model moreAT_global_htr_phi --checkpoint /work3/s203788/Master_Project_2026/EquivariantTransformerMPNN4QuantumComputations/models/trained_models/MatPES/GATAall2all/matpes_20260419_010919 --wandb_run_id 5gg07sae # the purple
$PYTHON_BIN train_model_from_checkpoint_MatPES.py  --model moreAT_global_htr_phi --checkpoint /work3/s203788/Master_Project_2026/EquivariantTransformerMPNN4QuantumComputations/models/trained_models/MatPES/GATAall2all/matpes_20260421_035224 --wandb_run_id ej727i1f # the yellow
$PYTHON_BIN train_model_from_checkpoint_MatPES.py  --model moreAT_global_htr_phi --checkpoint /work3/s203788/Master_Project_2026/EquivariantTransformerMPNN4QuantumComputations/models/trained_models/MatPES/GATAall2all/matpes_20260421_035224 --wandb_run_id ej727i1f # the yellow
$PYTHON_BIN train_model_from_checkpoint_MatPES.py  --model moreAT_global_htr_phi --checkpoint /work3/s203788/Master_Project_2026/EquivariantTransformerMPNN4QuantumComputations/models/trained_models/MatPES/GATAall2all/matpes_20260421_035224 --wandb_run_id ej727i1f # the yellow





