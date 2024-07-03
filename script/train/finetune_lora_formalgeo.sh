#!/bin/bash

# only for 4090
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export HF_ENDPOINT=https://hf-mirror.com
MODEL_TYPE=qwen2

# PRETRAIN_DIR=bunny-$MODEL_TYPE-9B-vision-pretrained-formalgeo-qwen0.5B
OUTPUT_DIR=bunny-lora-$MODEL_TYPE-0.5B-formalgeov2-structure_and_paring_problem-sft4epoch

mkdir -p checkpoints/checkpoints-$MODEL_TYPE/$OUTPUT_DIR
#    --vision_tower_pretrained_local_path checkpoints/checkpoints-qwen2/bunny-qwen2-caption-model-pgdp \
#     --pretrain_mm_mlp_adapter checkpoints/checkpoints-pretrain/$PRETRAIN_DIR/mm_projector.bin \
# 
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True 
deepspeed --include=localhost:0,1,2,3,4,5,6,7 --master_port 25679 bunny/train/train.py \
    --lora_enable True --lora_r 16 --lora_alpha 32 --mm_projector_lr 2e-5 \
    --deepspeed ./script/deepspeed/zero3.json \
    --model_name_or_path 01-ai/Yi-1.5-9B \
    --model_type $MODEL_TYPE \
    --version yi \
    --data_path ./data/formalgeo7k/formalgeo7k_v2/qa_structure_and_parsing_problem_train.json \
    --val_data_path ./data/formalgeo7k/formalgeo7k_v2/qa_structure_and_parsing_problem_minitrain.json \
    --image_folder ./data/formalgeo7k/formalgeo7k_v2 \
    --vision_tower google/siglip-so400m-patch14-384 \
    --freeze_vision_tower False \
    --tune_vision_tower False \
    --mm_projector_type mlp2x_gelu \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir checkpoints/checkpoints-$MODEL_TYPE/$OUTPUT_DIR \
    --num_train_epochs 4 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "steps" \
    --eval_steps 100 \
    --metric_for_best_model "exact_match" \
    --greater_is_better True \
    --load_best_model_at_end True \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 1 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_dir checkpoints/checkpoints-$MODEL_TYPE/$OUTPUT_DIR/ \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to "tensorboard" | tee 2>&1 checkpoints/checkpoints-$MODEL_TYPE/$OUTPUT_DIR/log.txt