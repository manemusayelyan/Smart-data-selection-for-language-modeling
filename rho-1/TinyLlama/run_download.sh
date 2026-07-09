#!/bin/bash

#SBATCH --partition=a100
#SBATCH --job-name=prepare_openwebmath
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=64
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH --output=/nfs/dgx/home/undergrad/mane/tinyllama_ct_math/TinyLlama/prepare_openwebmath_%j.log

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Start: $(date)"


python /nfs/dgx/home/undergrad/mane/tinyllama_ct_math/TinyLlama/scripts/prepare_openwebmath.py

echo "End: $(date)"