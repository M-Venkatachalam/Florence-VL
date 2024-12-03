#!/bin/bash
#SBATCH --job-name=llama
#SBATCH --nodes=4
#SBATCH --gres=gpu:8        # uncomment only if/as needed
#SBATCH --time=2-00:00:00    # run for one day
#SBATCH --cpus-per-task=16    # change as needed
#SBATCH --signal=USR1@140
#SBATCH --open-mode=append
#SBATCH --output=/data/home/jiuhai/llama3-mlp3x/logs/%j.out
#SBATCH --error=/data/home/jiuhai/llama3-mlp3x/logs/%j.err

source /fsx_0/user/jiuhai/florence/bin/activate
export HF_HOME=/fsx_0/user/jiuhai

bash pretrain_llama.sh
bash finetune_llama.sh
