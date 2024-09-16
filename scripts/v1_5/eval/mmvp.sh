#!/bin/bash

# Check if model path argument is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <model-path>"
  exit 1
fi

MODEL_PATH=$1
cov_model=$2

python -m llava.eval.model_vqa_loader \
    --model-path $MODEL_PATH \
    --question-file /fsx_0/user/jiuhai/data/eval/mmvp/mmvp.jsonl \
    --image-folder /fsx_0/user/jiuhai/hub/datasets--MMVP--MMVP/snapshots/37eafecab8a3940c50c2ade5b36de69dbc99a8cf/ \
    --answers-file playground/data/eval/mmvp/mmvp.jsonl \
    --temperature 0 \
    --conv-mode $cov_model
