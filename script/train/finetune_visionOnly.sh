#!/bin/bash
# only for 4090
# export NCCL_P2P_DISABLE=1
# export NCCL_IB_DISABLE=1
export HF_ENDPOINT=https://hf-mirror.com

# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True 

MODEL_TYPE=yi1.5

# PRETRAIN_DIR=bunny-$MODEL_TYPE-9B-Chat
# --use_s2 \
# --pretrain_mm_mlp_adapter checkpoints/checkpoints-pretrain/$PRETRAIN_DIR/mm_projector.bin \
# --vision_tower_pretrained_local_path checkpoints/checkpoints-qwen2/bunny-qwen2-caption-formalgeo-construction-cdl-and-image_cdl-0.5B-merged \

OUTPUT_DIR=bunny-$MODEL_TYPE-9B-Chat-TuneVisionOnly-sft2e

mkdir -p checkpoints/checkpoints-$MODEL_TYPE/$OUTPUT_DIR
deepspeed --include=localhost:0,1,2,3 --master_port 25679 bunny/train/train.py \
    --deepspeed ./script/deepspeed/zero3.json \
    --model_name_or_path 01-ai/Yi-1.5-9B-Chat \
    --model_type $MODEL_TYPE \
    --version yi-chat \
    --data_path data/gllava-data/qa_tuning.json \
    --image_folder data/gllava-data/images \
    --vision_tower google/siglip-so400m-patch14-384 \
    --freeze_vision_tower False \
    --tune_vision_tower True \
    --mm_projector_type mlp2x_gelu \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir checkpoints/checkpoints-$MODEL_TYPE/$OUTPUT_DIR \
    --num_train_epochs 2 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --logging_dir  checkpoints/checkpoints-$MODEL_TYPE/$OUTPUT_DIR/ \
    --tf32 True \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to "tensorboard" | tee 2>&1 checkpoints/checkpoints-$MODEL_TYPE/$OUTPUT_DIR/log.txt