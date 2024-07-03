#!/bin/bash
# only for 4090
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export HF_ENDPOINT=https://hf-mirror.com

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True 

MODEL_TYPE=phi-2

# PRETRAIN_DIR=bunny-$MODEL_TYPE-9B-alignment-pretrained-tune-vision
# --pretrain_mm_mlp_adapter checkpoints/checkpoints-pretrain/$PRETRAIN_DIR/mm_projector.bin \
# --vision_tower_pretrained_local_path ./checkpoints/checkpoints-llama/bunny-llama-8B-caption-model-tune-vision-only \
# --use_s2 \
OUTPUT_DIR=bunny-lora-$MODEL_TYPE-3B-dpo

mkdir -p checkpoints/checkpoints-$MODEL_TYPE/$OUTPUT_DIR
deepspeed --include=localhost:0 --master_port 25679 bunny/train/train_dpo.py \
    --lora_enable True --lora_r 64 --lora_alpha 128 --mm_projector_lr 2e-5 \
    --deepspeed ./script/deepspeed/zero3.json \
    --model_name_or_path microsoft/phi-2 \
    --model_type $MODEL_TYPE \
    --version bunny \
    --data_path ./data/VLFeedBack.json \
    --image_folder None \
    --vision_tower google/siglip-so400m-patch14-384 \
    --freeze_vision_tower True \
    --tune_vision_tower False \
    --mm_projector_type mlp2x_gelu \
    --image_aspect_ratio pad \
    --group_by_modality_length False \
    --bf16 True \
    --output_dir checkpoints/checkpoints-$MODEL_TYPE/$OUTPUT_DIR \
    --num_train_epochs 2 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --max_length 2048 \
    --max_prompt_length 1024 \
    --max_target_length 1024 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to none | tee 2>&1 checkpoints/checkpoints-$MODEL_TYPE/$OUTPUT_DIR/log.txt