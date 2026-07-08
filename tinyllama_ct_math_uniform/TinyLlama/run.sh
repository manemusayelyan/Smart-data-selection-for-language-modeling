#!/bin/bash

#SBATCH --partition=a100
#SBATCH --job-name=download_tinyllama
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=64
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH --output=/nfs/dgx/home/undergrad/mane/tinyllama_ct_math/TinyLlama/download_tinyllama_%j.log


python - <<'PYEOF'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
    local_dir="checkpoints/TinyLlama-1.1B-intermediate-step-1431k-3T",
    ignore_patterns=["*.md"]
)
PYEOF
