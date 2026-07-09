#!/bin/bash
#SBATCH --partition=research
#SBATCH --job-name=count_tokens_curated_openwebmath
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=64
#SBATCH --mem=60G
#SBATCH --time=24:00:00
#SBATCH --output=/home/mmusayelyan/tinyllama_ct_math/TinyLlama/count_tokens_curated_openwebmath_%j.log
    
    
python /home/mmusayelyan/tinyllama_ct_math/TinyLlama/count_tokens.py --data_dir=/mnt/weka/mmusayelyan/data/prep_curated_openwebmath_3.63M \
        --block_size=2048 \
        --micro_batch_size=16 \
        --grad_accum=4 \
        --world_size=8