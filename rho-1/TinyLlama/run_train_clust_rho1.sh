#!/bin/bash
#SBATCH --partition=research
#SBATCH --nodes=1
#SBATCH --job-name=clust_rho_1_long_uniform_rm
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=64
#SBATCH --mem=200G
#SBATCH --time=48:00:00
#SBATCH --output=/home/mmusayelyan/%x-%j.out
#SBATCH --error=/home/mmusayelyan/%x-%j.err

set -euo pipefail

ENVDIR=/mnt/weka/mmusayelyan/conda_envs/tinyllama_ct
ENVPY=$ENVDIR/bin/python
PROJECT_DIR=/home/mmusayelyan/rho-1/TinyLlama

export PATH="$ENVDIR/bin:$PATH"
unset PYTHONPATH
export PYTHONNOUSERSITE=1

export PIP_CACHE_DIR=/mnt/weka/mmusayelyan/.cache/pip
export HF_HOME=/mnt/weka/mmusayelyan/.cache/huggingface
export TRANSFORMERS_CACHE=/mnt/weka/mmusayelyan/.cache/huggingface
export TORCH_HOME=/mnt/weka/mmusayelyan/.cache/torch
export TORCH_EXTENSIONS_DIR=/mnt/weka/mmusayelyan/.cache/torch_extensions
export TMPDIR=/mnt/weka/mmusayelyan/tmp
export WANDB_DIR=/mnt/weka/mmusayelyan/.cache/wandb

export CUDA_HOME=/usr
export LD_LIBRARY_PATH=$ENVDIR/lib:${LD_LIBRARY_PATH:-}
export LD_LIBRARY_PATH=$ENVDIR/lib/python3.10/site-packages/nvidia/cuda_runtime/lib:$LD_LIBRARY_PATH

cd "$PROJECT_DIR"

"$ENVPY" -m torch.distributed.run \
  --standalone \
  --nproc_per_node=8 \
  rho_1_self_ref.py \
  --checkpoint_dir=/mnt/weka/mmusayelyan/tinyllama_checkpoints/TinyLlama-1.1B-intermediate-step-1431k-3T \
  --reference_checkpoint_dir=/mnt/weka/mmusayelyan/tinyllama_ct_uniform_14B_exp/step-13952 \
  --train_data_dir=/mnt/weka/mmusayelyan/data/prep_hkmeans_owm_3.63M \
  --out_dir=/mnt/weka/mmusayelyan/clust_rho_1_long_uniform_rm \
  --selection_method=self_reference_intersection \
  --rm_loss_select_ratio=0.7 \
  --rm_entropy_select_ratio=0.7