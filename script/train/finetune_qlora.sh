#!/bin/bash
export HF_ENDPOINT=https://hf-mirror.com
MODEL_TYPE=mixtral
# export NCCL_DEBUG=INFO
# PRETRAIN_DIR=bunny-$MODEL_TYPE-70B
OUTPUT_DIR=bunny-lora-$MODEL_TYPE-8x7B

mkdir -p checkpoints/checkpoints-$MODEL_TYPE/$OUTPUT_DIR
#    --vision_tower_pretrained_local_path checkpoints/checkpoints-qwen2/bunny-qwen2-caption-model-pgdp \
#    --pretrain_mm_mlp_adapter checkpoints/checkpoints-pretrain/$PRETRAIN_DIR/mm_projector.bin \
# --lora_r 128 --lora_alpha 256

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True deepspeed --include=localhost:4,5,6,7 --master_port 25679 bunny/train/train.py \
    --lora_enable True --lora_r 64 --lora_alpha 128 --mm_projector_lr 2e-5 \
    --deepspeed ./script/deepspeed/zero2.json \
    --model_name_or_path mistralai/Mixtral-8x7B-Instruct-v0.1 \
    --model_type $MODEL_TYPE \
    --version mistral \
    --data_path ./data/qa_tuning.json \
    --image_folder ./data/images \
    --vision_tower google/siglip-so400m-patch14-384 \
    --freeze_vision_tower True \
    --mm_projector_type mlp2x_gelu \
    --image_aspect_ratio pad \
    --group_by_modality_length False \
    --bit 4 \
    --quant_type nf4 \
    --double_quant \
    --bf16 True \
    --output_dir checkpoints/checkpoints-$MODEL_TYPE/$OUTPUT_DIR \
    --num_train_epochs 2 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 1 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to none | tee 2>&1 checkpoints/checkpoints-$MODEL_TYPE/$OUTPUT_DIR/log.txt