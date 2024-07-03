#!/bin/bash

# only for 4090
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export HF_ENDPOINT=https://hf-mirror.com
# 
#     --vision_tower_pretrained_local_path checkpoints/checkpoints-qwen2/bunny-qwen2-formalgeov2-qa-structure_only-tuneVisionOnly-sft10 \
MODEL_TYPE=qwen2

OUTPUT_DIR=bunny-$MODEL_TYPE-qa-FormalGeoV2Aug10Times_structure_only-sft2
mkdir -p checkpoints/checkpoints-$MODEL_TYPE/$OUTPUT_DIR

deepspeed --include=localhost:0,1,2,3,4,5,6,7 --master_port 25683 bunny/train/train.py \
    --deepspeed ./script/deepspeed/zero3.json \
    --model_name_or_path Qwen/Qwen2-0.5B-Instruct \
    --model_type $MODEL_TYPE \
    --use_formalgeo_vocab_only False \
    --add_formal_tokens False \
    --force_tune_embedding False \
    --version qwen-chat \
    --data_path data/formalgeo7k/formalgeo7k_v2/custom_json/qa_structure_only/qa_structure_only_train_aug.json \
    --val_data_path data/formalgeo7k/formalgeo7k_v2/custom_json/qa_structure_only/qa_structure_only_minival.json \
    --image_folder data/formalgeo7k/formalgeo7k_v2 \
    --customized_aug True \
    --vision_tower google/siglip-so400m-patch14-384 \
    --freeze_vision_tower False \
    --tune_vision_tower False \
    --mm_projector_type mlp2x_gelu \
    --image_aspect_ratio pad \
    --group_by_modality_length False \
    --bf16 True \
    --output_dir checkpoints/checkpoints-$MODEL_TYPE/$OUTPUT_DIR \
    --num_train_epochs 2 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "steps" \
    --eval_steps 500 \
    --metric_for_best_model "eval_loss" \
    --greater_is_better False \
    --load_best_model_at_end False \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_dir checkpoints/checkpoints-$MODEL_TYPE/$OUTPUT_DIR/ \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to "tensorboard" | tee 2>&1 checkpoints/checkpoints-$MODEL_TYPE/$OUTPUT_DIR/log.txt

sh zk.sh