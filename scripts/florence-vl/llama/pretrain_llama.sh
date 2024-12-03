export CUDA_DEVICE_MAX_CONNECTIONS=1
export GPUS_PER_NODE=8
export NNODES=16
export MASTER_PORT=29502
export CPUS_PER_TASK=16




export DATA_PATH=/fsx_0/user/jiuhai/data/ShareGPT4V/pixelprose-all-sharegpt4v-caption.json
export IMG=/fsx_0/user/jiuhai/data/LLaVA-Instruct-150K

export OUTPUT=/fsx_0/user/jiuhai/model/llava-llama-3-pretrain-pixelprose-all-final-test

export SAVE_PATH=llama3-pretrained-pixelprose
export TUNE_ENTIRE_MODEL=true
export BASE_LR=1e-5
export GRADIENT_ACCU_STEPS=1








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
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path /fsx_0/user/jiuhai/hub/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/4281e96c7cf5ab6b312ef0cb78373efa3976a9dd  \
    --version plain \
    --data_path ${DATA_PATH} \
    --image_folder ${IMG} \
    --vision_tower /fsx_0/user/jiuhai/hub/models--microsoft--Florence-2-large-ft/snapshots/c669c6b8bfbd7f0193fcb31f997879045a3612f3  \
    --mm_projector_type mlp3x_gelu \
    --tune_entire_model ${TUNE_ENTIRE_MODEL} \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir ${OUTPUT} \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps ${GRADIENT_ACCU_STEPS} \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 3000 \
    --save_total_limit 1 \
    --learning_rate ${BASE_LR} \
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




cp  /fsx_0/user/jiuhai/model/modeling_florence2.py  ${OUTPUT}/vision_tower/

