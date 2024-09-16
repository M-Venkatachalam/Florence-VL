#!/bin/bash
#SBATCH --job-name=eval
#SBATCH --nodes=1
#SBATCH --gres=gpu:8        # uncomment only if/as needed
#SBATCH --time=2-00:00:00    # run for one day
#SBATCH --cpus-per-task=16    # change as needed
#SBATCH --signal=USR1@140
#SBATCH --open-mode=append
#SBATCH --output=/data/home/jiuhai/llama3-mlp3x/logs/%j.out
#SBATCH --error=/data/home/jiuhai/llama3-mlp3x/logs/%j.err


source /fsx_0/user/jiuhai/llama3-v/bin/activate
export HF_HOME=/fsx_0/user/jiuhai
export WANDB_API_KEY='d8075df78a873149bb390d22e6fc2c6de539e365'
export OPENAI_API_KEY=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


deepspeed llava/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path /fsx_0/user/jiuhai/hub/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/e1945c40cd546c78e41f1151f4db032b271faeaa  \
    --version plain \
    --data_path /fsx_0/user/jiuhai/data/ShareGPT4V/share-captioner_coco_lcs_sam_1246k_1107.json \
    --image_folder /fsx_0/user/jiuhai/data/LLaVA-Instruct-150K \
    --vision_tower /fsx_0/user/jiuhai/hub/models--microsoft--Florence-2-large-ft/snapshots/c669c6b8bfbd7f0193fcb31f997879045a3612f3  \
    --mm_projector_type mlp3x_gelu \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir /fsx_0/user/jiuhai/model/llava-llama-3-florence-pretrain-sharegpt \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 24000 \
    --save_total_limit 1 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb




deepspeed llava/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path /fsx_0/user/jiuhai/hub/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/e1945c40cd546c78e41f1151f4db032b271faeaa  \
    --version llama3 \
    --data_path /fsx_0/user/jiuhai/data/ShareGPT4V/sharegpt4v_mix665k_cap23k_coco-ap9k_lcs3k_sam9k_div2k.json \
    --image_folder /fsx_0/user/jiuhai/data/LLaVA-Instruct-150K \
    --vision_tower /fsx_0/user/jiuhai/hub/models--microsoft--Florence-2-large-ft/snapshots/c669c6b8bfbd7f0193fcb31f997879045a3612f3  \
    --pretrain_mm_mlp_adapter /fsx_0/user/jiuhai/model/llava-llama-3-florence-pretrain-sharegpt/mm_projector.bin \
    --mm_projector_type mlp3x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir /fsx_0/user/jiuhai/model/llava-llama-3-florence-sharegpt \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 5e-6 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb




python -m accelerate.commands.launch \
    --num_processes=8 \
    -m lmms_eval \
    --model llava \
    --model_args pretrained="/fsx_0/user/jiuhai/model/llava-llama-3-florence-sharegpt,conv_template=llama3" \
    --tasks  textvqa_val,gqa,vizwiz_vqa_val,scienceqa_img,pope,mmvet,mme,seedbench,hallusion_bench_image,llava_in_the_wild,mathvista_testmini,docvqa_val,ocrbench,chartqa,ai2d,mmmu_val,mmbench_en_dev \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix llava-lama-3-florence-sharegpt \
    --output_path ./logs/
