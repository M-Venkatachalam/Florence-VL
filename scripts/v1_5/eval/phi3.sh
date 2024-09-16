#!/bin/bash


python -m llava.eval.phi3_vqa_loader \
    --question-file /fsx_0/user/jiuhai/data/eval/cv-bench/llava-cv-bench-2D.jsonl \
    --image-folder /fsx_0/user/jiuhai/hub/datasets--nyu-visionx--CV-Bench/snapshots/22409a927ab5cf68e3655023d51694587455fc99/ \
    --answers-file playground/data/eval/cv-bench/test-2D.jsonl \
    --temperature 0 



python -m llava.eval.phi3_vqa_loader \
    --question-file /fsx_0/user/jiuhai/data/eval/cv-bench/llava-cv-bench-3D.jsonl \
    --image-folder /fsx_0/user/jiuhai/hub/datasets--nyu-visionx--CV-Bench/snapshots/22409a927ab5cf68e3655023d51694587455fc99/ \
    --answers-file playground/data/eval/cv-bench/test-3D.jsonl \
    --temperature 0 




python -m llava.eval.phi3_vqa_loader \
    --question-file /fsx_0/user/jiuhai/data/eval/mmvp/mmvp.jsonl \
    --image-folder /fsx_0/user/jiuhai/hub/datasets--MMVP--MMVP/snapshots/37eafecab8a3940c50c2ade5b36de69dbc99a8cf/ \
    --answers-file playground/data/eval/mmvp/mmvp.jsonl \
    --temperature 0 



python -m llava.eval.phi3_sqa_loader \
    --question-file ./playground/data/eval/scienceqa/llava_test_CQM-A.json \
    --image-folder ./ScienceQA/tools/test \
    --answers-file ./playground/data/eval/scienceqa/answers/florence.jsonl \
    --temperature 0 

python llava/eval/eval_science_qa.py \
    --base-dir ./ScienceQA/data/scienceqa \
    --result-file ./playground/data/eval/scienceqa/answers/florence.jsonl \
    --output-file ./playground/data/eval/scienceqa/answers/florence_output.jsonl \
    --output-result ./playground/data/eval/scienceqa/answers/florence_result.json
