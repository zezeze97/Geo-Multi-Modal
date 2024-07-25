#!/bin/bash
#SBATCH -o job.%j.out
#SBATCH --partition=GPU80G
#SBATCH --qos=low
#SBATCH -J siglip0.4B-yi1.5-9B-Chat-visionPretrained-alignment-sft2-V14-MultiNode
#SBATCH --nodes=2    
#SBATCH --ntasks=8    
#SBATCH --cpus-per-task=16 
#SBATCH --gres=gpu:4  
#SBATCH --time=5-00:00:00



export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True 
export NCCL_IB_DISABLE=1
export GPUS_PER_NODE=4
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=9901


#replaces the content of hostfile every time
function makehostfile() {
perl -e '$slots=split /,/, $ENV{"SLURM_STEP_GPUS"};
$slots=4 if $slots==0; # workaround 8 gpu machines
@nodes = split /\n/, qx[scontrol show hostnames $ENV{"SLURM_JOB_NODELIST"}];
print map { "$b$_ slots=$slots\n" } @nodes'
}
makehostfile > hostfile



MODEL_TYPE=yi1.5
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True 
PRETRAIN_DIR=bunny-$MODEL_TYPE-9B-Chat-VisionPretrained-v7-1epoch
# --use_s2 \
# --vision_tower_pretrained_local_path checkpoints/checkpoints-qwen2/bunny-qwen2-caption-formalgeo-construction-cdl-and-image_cdl-0.5B-merged \
# 
# --vision_tower_pretrained_local_path checkpoints/checkpoints-qwen2/bunny-lora-qwen2-qa-FormalGeoV2Aug10Times_structure_only-sft2/merged \

OUTPUT_DIR=bunny-$MODEL_TYPE-9B-Chat-FormalGeoCoT-VisionPretrained-sft2e-v14-2Node

mkdir -p checkpoints/checkpoints-$MODEL_TYPE/$OUTPUT_DIR
deepspeed --num_gpus 4 --num_nodes 2 --hostfile ./hostfile --launcher SLURM --master_addr $MASTER_ADDR --master_port $MASTER_PORT bunny/train/train_multiNode.py \
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
    --gradient_accumulation_steps 2 \
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