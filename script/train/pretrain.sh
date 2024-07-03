#!/bin/bash

export HF_ENDPOINT=https://hf-mirror.com
MODEL_TYPE=yi1.5
OUTPUT_DIR=bunny-$MODEL_TYPE-9B-Chat-VisionPretrained-ArXivAlignment
mkdir -p checkpoints/checkpoints-pretrain/$OUTPUT_DIR
# --use_s2 True \
# --vision_tower_pretrained_local_path checkpoints/checkpoints-qwen2/bunny-qwen2-caption-formalgeo-construction-cdl-and-image_cdl-0.5B-merged \
#   checkpoints/checkpoints-qwen2/bunny-qwen2-caption-sythv3-0.4+0.5B \
deepspeed --include=localhost:0,1,2,3,4,5,6,7 --master_port 25678 bunny/train/train.py \
    --deepspeed ./script/deepspeed/zero3.json \
    --model_name_or_path 01-ai/Yi-1.5-9B-Chat \
    --model_type $MODEL_TYPE \
    --version yi-chat \
    --data_path data/arXiv/processed_data/caption_data.json  \
    --image_folder data/arXiv/processed_data \
    --vision_tower google/siglip-so400m-patch14-384 \
    --vision_tower_pretrained_local_path checkpoints/checkpoints-qwen2/bunny-lora-qwen2-qa-FormalGeoV2Aug10Times_structure_only-sft2/merged \
    --freeze_vision_tower True \
    --tune_vision_tower False \
    --tune_mm_mlp_adapter True \
    --mm_projector_type mlp2x_gelu \
    --image_aspect_ratio pad \
    --bf16 True \
    --output_dir checkpoints/checkpoints-pretrain/$OUTPUT_DIR \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 1 \
    --learning_rate 5e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --logging_dir checkpoints/checkpoints-pretrain/$OUTPUT_DIR/ \
    --tf32 True \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to "tensorboard" | tee 2>&1 checkpoints/checkpoints-pretrain/$OUTPUT_DIR/log.txt
