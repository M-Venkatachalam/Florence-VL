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
    --question-file /fsx_0/user/jiuhai/data/eval/cv-bench/llava-cv-bench-2D.jsonl \
    --image-folder /fsx_0/user/jiuhai/hub/datasets--nyu-visionx--CV-Bench/snapshots/22409a927ab5cf68e3655023d51694587455fc99/ \
    --answers-file playground/data/eval/cv-bench/test-2D.jsonl \
    --temperature 0 \
    --conv-mode $cov_model
