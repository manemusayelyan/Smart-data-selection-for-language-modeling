#!/bin/bash
#SBATCH --partition=research
#SBATCH --nodes=1
#SBATCH --job-name=tinyllama_ct_clust_14B
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=64
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH --output=/home/mmusayelyan/tinyllama_ct_math_curated/logs/%x-%j.out
#SBATCH --error=/home/mmusayelyan/tinyllama_ct_math_curated/logs/%x-%j.err

set -euo pipefail

ENVDIR=/mnt/weka/mmusayelyan/conda_envs/tinyllama_ct
ENVPY=$ENVDIR/bin/python
PROJECT_DIR=/home/mmusayelyan/tinyllama_ct_math_curated/TinyLlama

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
  tinyllama_ct.py \
  --out_dir=/mnt/weka/mmusayelyan/tinyllama_ct_clust_14B_exp2 \
  --max_steps 13952 \
  --warmup_steps 250 \
  --stable_steps 10928 \
  --save_interval 6976