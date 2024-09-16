export CUDA_DEVICE_MAX_CONNECTIONS=1
export GPUS_PER_NODE=8
export NNODES=16
export MASTER_PORT=29501
export CPUS_PER_TASK=24


# export DATA_PATH=/fsx_0/user/jiuhai/data/ShareGPT4V/cambrian_sharegpt4v_vision_flan_docmatix.json

export DATA_PATH=/fsx_0/user/jiuhai/data/ShareGPT4V/cambrian_sharegpt4v_vision_flan_docmatix

# export DATA_PATH=/fsx_0/user/jiuhai/hub/datasets--Vision-Flan--vision-flan/snapshots/e8c6f09736277ef63b33dea5e9bbe94392dba76c/vision_flan_processed.json



export IMG=/fsx_0/user/jiuhai/data/LLaVA-Instruct-150K

export CKPT_PATH=/fsx_0/user/jiuhai/model/llava-pretrain-pixelprose-Phi3-second
export VIT_PATH=/fsx_0/user/jiuhai/model/llava-pretrain-pixelprose-Phi3-second/vision_tower
export OUTPUT=/fsx_0/user/jiuhai/model/llava-sft-Phi3-second-two-epoch




export SAVE_PATH=phi3-sft


export LEARNIG_RATE=9e-5

export TUNE_ENTIRE_MODEL=true





SRUN_ARGS=${SRUN_ARGS:-""}
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
srun -p q1 \
    --nodes=$NNODES \
    --ntasks-per-node=1 \
    --gres=gpu:$GPUS_PER_NODE \
    --cpus-per-task=$CPUS_PER_TASK \
    --kill-on-bad-exit=1 \
    ${SRUN_ARGS} \
    bash -c 'torchrun --nnodes $NNODES --nproc_per_node $GPUS_PER_NODE --node_rank $SLURM_NODEID --master_addr $(scontrol show hostname $SLURM_NODELIST | head -n1) --master_port ${MASTER_PORT} llava/train/train_mem.py \
    --model_name_or_path ${CKPT_PATH} \
    --deepspeed ./scripts/zero2.json \
    --version phi3 \
    --data_path ${DATA_PATH} \
    --image_folder ${IMG} \
    --vision_tower ${VIT_PATH} \
    --mm_projector_type mlp3x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ${OUTPUT} \
    --num_train_epochs 2 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate ${LEARNIG_RATE} \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 1 \
    --lazy_preprocess True \
    --report_to wandb \
    --run_name ${SAVE_PATH}'




python -m accelerate.commands.launch \
    --num_processes=8 \
    -m lmms_eval \
    --model llava \
    --model_args pretrained="/fsx_0/user/jiuhai/model/llava-sft-Phi3-second-two-epoch,conv_template=phi3" \
    --tasks  textvqa_val,gqa,realworldqa,vizwiz_vqa_val,scienceqa_img,pope,mmvet,mme,seedbench,hallusion_bench_image,llava_in_the_wild,mathvista_testmini,docvqa_val,ocrbench,chartqa,ai2d,mmmu_val,mmbench_en_dev,infovqa_val,mmbench_cn_dev,mmstar \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix llava-llama-3-pretrain-gpt-full-20m \
    --output_path ./logs/



# bash scripts/v1_5/eval/cv_bench_2D.sh   /fsx_0/user/jiuhai/model/llava-llama-3-sft-phi3
# bash scripts/v1_5/eval/cv_bench_3D.sh  /fsx_0/user/jiuhai/model/llava-llama-3-sft-phi3
# bash scripts/v1_5/eval/mmvp.sh   /fsx_0/user/jiuhai/model/llava-llama-3-sft-phi3

# python playground/data/eval/cv-bench/eval.py  /fsx_0/user/jiuhai/data/eval/cv-bench/llava-cv-bench-3D.jsonl   /data/home/jiuhai/llama3-mlp3x/playground/data/eval/cv-bench/test-3D.jsonl
# python playground/data/eval/cv-bench/eval.py  /fsx_0/user/jiuhai/data/eval/cv-bench/llava-cv-bench-2D.jsonl   /data/home/jiuhai/llama3-mlp3x/playground/data/eval/cv-bench/test-2D.jsonl
# python playground/data/eval/cv-bench/eval.py  /fsx_0/user/jiuhai/data/eval/mmvp/mmvp.jsonl   /data/home/jiuhai/llama3-mlp3x/playground/data/eval/mmvp/mmvp.jsonl
