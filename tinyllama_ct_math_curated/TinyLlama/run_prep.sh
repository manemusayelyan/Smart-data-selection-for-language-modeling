#!/bin/bash

#SBATCH --partition=research
#SBATCH --job-name=prepare_curated_openwebmath
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=64
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH --output=/home/mmusayelyan/tinyllama_ct_math_curated/TinyLlama/prepare_curated_openwebmath_%j.log

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Start: $(date)"

python /home/mmusayelyan/tinyllama_ct_math_curated/TinyLlama/scripts/prepare_owm_curated.py

echo "End: $(date)"