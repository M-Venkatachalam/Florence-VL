#!/bin/bash
#SBATCH --job-name=sft-eval
#SBATCH --nodes=1
#SBATCH --account=ar-ai-research-interns
#SBATCH --gres=gpu:8        # uncomment only if/as needed
#SBATCH --time=2-00:00:00    # run for one day
#SBATCH --cpus-per-task=16    # change as needed
#SBATCH --signal=USR1@140
#SBATCH --open-mode=append
#SBATCH --output=/data/home/jiuhai/llama3-mlp3x/logs/%j.out
#SBATCH --error=/data/home/jiuhai/llama3-mlp3x/logs/%j.err

source /fsx_0/user/jiuhai/llama-full/bin/activate
export HF_HOME=/fsx_0/user/jiuhai
export WANDB_API_KEY='d8075df78a873149bb390d22e6fc2c6de539e365'





# sh pretrain_full.sh 
sh finetune_full.sh







