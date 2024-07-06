#!/bin/bash

# only for 4090
# export NCCL_P2P_DISABLE=1
# export NCCL_IB_DISABLE=1
export HF_ENDPOINT=https://hf-mirror.com
MODEL_TYPE=yi1.5

# PRETRAIN_DIR=bunny-$MODEL_TYPE-34B-Chat
OUTPUT_DIR=bunny-lora-$MODEL_TYPE-9B-Chat-FormalGeoCoT-VisionPretrained-Sft1e-mixv5
mkdir -p checkpoints/checkpoints-$MODEL_TYPE/$OUTPUT_DIR
#    --vision_tower_pretrained_local_path checkpoints/checkpoints-qwen2/bunny-qwen2-caption-model-pgdp \
#     --pretrain_mm_mlp_adapter checkpoints/checkpoints-pretrain/$PRETRAIN_DIR/mm_projector.bin \
#  --vision_tower_pretrained_local_path checkpoints/checkpoints-qwen2/bunny-qwen2-caption-formalgeo-construction-cdl-and-image_cdl-0.5B-merged \
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True 
deepspeed --include=localhost:0,1,2,3,4,5,6,7 --master_port 25679 bunny/train/train.py \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 1e-5 \
    --deepspeed ./script/deepspeed/zero3.json \
    --model_name_or_path 01-ai/Yi-1.5-9B-Chat \
    --model_type $MODEL_TYPE \
    --version yi-chat \
    --data_path data/formalgeo7k/formalgeo7k_v2/custom_json/qa_mixTask/qa_mix_train_v5.json \
    --image_folder data \
    --customized_aug True \
    --vision_tower google/siglip-so400m-patch14-384 \
    --vision_tower_pretrained_local_path checkpoints/checkpoints-qwen2/bunny-lora-qwen2-qa-FormalGeoV2Aug3Times_calibrate_v1v2_structure_only-sft2/merged \
    --freeze_vision_tower True \
    --tune_vision_tower False \
    --mm_projector_type mlp2x_gelu \
    --image_aspect_ratio pad \
    --group_by_modality_length False \
    --bf16 True \
    --output_dir checkpoints/checkpoints-$MODEL_TYPE/$OUTPUT_DIR \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 1 \
    --learning_rate 1e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --logging_dir checkpoints/checkpoints-$MODEL_TYPE/$OUTPUT_DIR/ \
    --tf32 True \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to "tensorboard" | tee 2>&1 checkpoints/checkpoints-$MODEL_TYPE/$OUTPUT_DIR/log.txt


sh zk.sh
