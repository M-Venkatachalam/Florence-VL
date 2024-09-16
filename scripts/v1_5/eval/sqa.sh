#!/bin/bash

source /fsx_0/user/jiuhai/florence/bin/activate
export HF_HOME=/fsx_0/user/jiuhai
export WANDB_API_KEY='d8075df78a873149bb390d22e6fc2c6de539e365'
export OPENAI_API_KEY=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


bash scripts/v1_5/eval/cv_bench_2D.sh   /fsx_0/user/jiuhai/model/llava-15-sft-llama3-final llama3
bash scripts/v1_5/eval/cv_bench_3D.sh   /fsx_0/user/jiuhai/model/llava-15-sft-llama3-final   llama3
bash scripts/v1_5/eval/mmvp.sh  /fsx_0/user/jiuhai/model/llava-15-sft-llama3-final   llama3

python playground/data/eval/cv-bench/eval.py  /fsx_0/user/jiuhai/data/eval/cv-bench/llava-cv-bench-2D.jsonl   /data/home/jiuhai/llama3-mlp3x/playground/data/eval/cv-bench/test-2D.jsonl
python playground/data/eval/cv-bench/eval.py  /fsx_0/user/jiuhai/data/eval/cv-bench/llava-cv-bench-3D.jsonl   /data/home/jiuhai/llama3-mlp3x/playground/data/eval/cv-bench/test-3D.jsonl
python playground/data/eval/cv-bench/eval.py  /fsx_0/user/jiuhai/data/eval/mmvp/mmvp.jsonl   /data/home/jiuhai/llama3-mlp3x/playground/data/eval/mmvp/mmvp.jsonl



python -m llava.eval.model_vqa_science \
    --model-path  /fsx_0/user/jiuhai/model/llava-15-sft-llama3-final \
    --question-file ./playground/data/eval/scienceqa/llava_test_CQM-A.json \
    --image-folder ./ScienceQA/tools/test \
    --answers-file ./playground/data/eval/scienceqa/answers/florence.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode llama3

python llava/eval/eval_science_qa.py \
    --base-dir ./ScienceQA/data/scienceqa \
    --result-file ./playground/data/eval/scienceqa/answers/florence.jsonl \
    --output-file ./playground/data/eval/scienceqa/answers/florence_output.jsonl \
    --output-result ./playground/data/eval/scienceqa/answers/florence_result.json
