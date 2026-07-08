#!/bin/bash
#SBATCH --partition=research
#SBATCH --job-name=count_owm_uniform_tokens
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=64
#SBATCH --mem=60G
#SBATCH --time=24:00:00
#SBATCH --output=/home/mmusayelyan/tinyllama_ct_math_uniform/TinyLlama/count_owm_uniform_tokens_%j.log
    
    
python /home/mmusayelyan/tinyllama_ct_math_uniform/TinyLlama/count_tokens.py --data_dir=/mnt/weka/mmusayelyan/data/prep_owm_uniform_3.91M \
        --block_size=2048 \
        --micro_batch_size=16 \
        --grad_accum=4 \
        --world_size=8