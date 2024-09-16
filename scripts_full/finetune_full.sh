export CUDA_DEVICE_MAX_CONNECTIONS=1
export GPUS_PER_NODE=8
export NNODES=1
export MASTER_PORT=29501
export CPUS_PER_TASK=16


export DATA_PATH=/fsx_0/user/jiuhai/data/LLaVA-Instruct-150K/llava_v1_5_mix665k.json
export IMG=/fsx_0/user/jiuhai/data/LLaVA-Instruct-150K 

export CKPT_PATH=/fsx_0/user/jiuhai/model/llava-llama-3-pretrain-full
export VIT_PATH=/fsx_0/user/jiuhai/model/llava-llama-3-pretrain-full/vision_tower
export OUTPUT=/fsx_0/user/jiuhai/model/llava-llama-3-sft-full-v
export TUNE_ENTIRE_MODEL=true


export SAVE_PATH=llama3-sft
export LEARNIG_RATE=1e-5


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
    --version llama3 \
    --data_path ${DATA_PATH} \
    --image_folder ${IMG} \
    --vision_tower ${VIT_PATH} \
    --mm_projector_type mlp3x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --tune_entire_model ${TUNE_ENTIRE_MODEL} \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ${OUTPUT} \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
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
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --run_name ${SAVE_PATH}'




