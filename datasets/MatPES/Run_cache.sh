#!/bin/sh 
### General options 
### -- specify queue -- 
#BSUB -q hpc
### -- set the job Name -- 
#BSUB -J My_Application
### -- ask for number of cores (default: 1) -- 
#BSUB -n 4 
### -- specify that the cores must be on the same host -- 
#BSUB -R "span[hosts=1]"
### -- specify that we need 4GB of memory per core/slot -- 
#BSUB -R "rusage[mem=4GB]"
### -- specify that we want the job to get killed if it exceeds 5 GB per core/slot -- 
#BSUB -M 5GB
### -- set walltime limit: hh:mm -- 
#BSUB -W 24:00 
### -- set the email address -- 
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
##BSUB -u your_email_address
### -- send notification at start -- 
#BSUB -B 
### -- send notification at completion -- 
#BSUB -N 
### -- Specify the output and error file. %J is the job-id -- 
### -- -o and -e mean append, -oo and -eo mean overwrite -- 
#BSUB -o Output_%J.out 
#BSUB -e Output_%J.err 



echo "================================================================"
echo "Job ID: $LSB_JOBID"
echo "Started at: $(date)"
echo "Running on: $(hostname)"
echo "================================================================"


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
$PYTHON_BIN preprocess_cache.py
