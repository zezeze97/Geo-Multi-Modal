#!/bin/bash
#SBATCH -o job.%j.out
#SBATCH --partition=GPU80G
#SBATCH --qos=low
#SBATCH -J siglip0.4B-yi1.5-9B-Chat-visionPretrained-alignment-sft2-V14
#SBATCH --nodes=1    
#SBATCH --ntasks=4     
#SBATCH --cpus-per-task=16 
#SBATCH --gres=gpu:4  
#SBATCH --time=5-00:00:00

export HF_ENDPOINT=https://hf-mirror.com

# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True 

MODEL_TYPE=yi1.5

PRETRAIN_DIR=bunny-$MODEL_TYPE-9B-Chat-VisionPretrained-v7-1epoch
# --use_s2 \
# --vision_tower_pretrained_local_path checkpoints/checkpoints-qwen2/bunny-qwen2-caption-formalgeo-construction-cdl-and-image_cdl-0.5B-merged \
# 
# --vision_tower_pretrained_local_path checkpoints/checkpoints-qwen2/bunny-lora-qwen2-qa-FormalGeoV2Aug10Times_structure_only-sft2/merged \

OUTPUT_DIR=bunny-$MODEL_TYPE-9B-Chat-FormalGeoCoT-VisionPretrained-sft2e-v14

mkdir -p checkpoints/checkpoints-$MODEL_TYPE/$OUTPUT_DIR
deepspeed --include=localhost:0,1,2,3 --master_port 25679 bunny/train/train.py \
    --deepspeed ./script/deepspeed/zero3.json \
    --model_name_or_path 01-ai/Yi-1.5-9B-Chat \
    --model_type $MODEL_TYPE \
    --use_formalgeo_vocab_only False \
    --add_formal_tokens False \
    --force_tune_embedding False \
    --version yi-chat \
    --data_path data/formalgeo7k/formalgeo7k_v2/custom_json/qa_mixTask/qa_mix_train_v14.json \
    --image_folder data/formalgeo7k/formalgeo7k_v2 \
    --customized_aug True \
    --vision_tower google/siglip-so400m-patch14-384 \
    --vision_tower_pretrained_local_path checkpoints/checkpoints-qwen2/bunny-lora-qwen2-qa-FormalGeoV2Aug10Times_calibrate_structure_only-sft4-add05/merged \
    --pretrain_mm_mlp_adapter checkpoints/checkpoints-pretrain/$PRETRAIN_DIR/mm_projector.bin \
    --freeze_vision_tower True \
    --tune_vision_tower False \
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

# sh zk.sh