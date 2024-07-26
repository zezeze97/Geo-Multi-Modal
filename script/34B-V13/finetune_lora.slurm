#!/bin/bash
#SBATCH -o job.%j.out
#SBATCH --partition=GPU80G
#SBATCH --qos=low
#SBATCH -J 34B-V13
#SBATCH --nodes=1    
#SBATCH --ntasks=4    
#SBATCH --cpus-per-task=16 
#SBATCH --gres=gpu:4  
#SBATCH --time=5-00:00:00
module load cuda/11.8
module load gcc/12.2.0
module load openmpi/4.1.5-gcc_12.2.0
export TMPDIR=/tmp
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True 
export NCCL_IB_DISABLE=1
export GPUS_PER_NODE=4
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=9905



#replaces the content of hostfile every time
function makehostfile() {
perl -e '$slots=split /,/, $ENV{"SLURM_STEP_GPUS"};
$slots=4 if $slots==0; # workaround 8 gpu machines
@nodes = split /\n/, qx[scontrol show hostnames $ENV{"SLURM_JOB_NODELIST"}];
print map { "$b$_ slots=$slots\n" } @nodes'
}
makehostfile > hostfile


MODEL_TYPE=yi1.5

PRETRAIN_DIR=bunny-$MODEL_TYPE-34B-Chat-VisionPretrained-v13-1epoch
OUTPUT_DIR=bunny-lora-128-$MODEL_TYPE-34B-Chat-FormalGeoCoT-VisionPretrained-sft2e-v13
mkdir -p checkpoints/checkpoints-$MODEL_TYPE/$OUTPUT_DIR
#    --vision_tower_pretrained_local_path checkpoints/checkpoints-qwen2/bunny-qwen2-caption-model-pgdp \
#     
#  --vision_tower_pretrained_local_path checkpoints/checkpoints-qwen2/bunny-qwen2-caption-formalgeo-construction-cdl-and-image_cdl-0.5B-merged \
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True 
deepspeed --num_gpus 4 --num_nodes 1 --hostfile ./hostfile --launcher SLURM --master_addr $MASTER_ADDR --master_port $MASTER_PORT bunny/train/train_multiNode.py \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 1e-5 \
    --deepspeed ./script/deepspeed/zero3.json \
    --model_name_or_path 01-ai/Yi-1.5-34B-Chat \
    --model_type $MODEL_TYPE \
    --use_formalgeo_vocab_only False \
    --add_formal_tokens False \
    --force_tune_embedding False \
    --version yi-chat \
    --data_path data/formalgeo7k/formalgeo7k_v2/custom_json/qa_resoning/formalgeov2_aug_train_less_rewrite.json \
    --image_folder data/formalgeo7k/formalgeo7k_v2 \
    --customized_aug True \
    --vision_tower google/siglip-so400m-patch14-384 \
    --vision_tower_pretrained_local_path checkpoints/checkpoints-qwen2/bunny-lora-qwen2-qa-FormalGeoV2Aug10Times_calibrate_structure_only-sft4-add05/merged \
    --pretrain_mm_mlp_adapter checkpoints/checkpoints-pretrain/$PRETRAIN_DIR/mm_projector.bin \
    --freeze_vision_tower True \
    --tune_vision_tower False \
    --mm_projector_type mlp2x_gelu \
    --image_aspect_ratio pad \
    --group_by_modality_length False \
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


python script/merge_lora_weights.py --model-type yi1.5 \
                                    --model-base 01-ai/Yi-1.5-34B-Chat \
                                    --model-path checkpoints/checkpoints-$MODEL_TYPE/$OUTPUT_DIR \
                                    --vision-encoder-path checkpoints/checkpoints-qwen2/bunny-lora-qwen2-qa-FormalGeoV2Aug10Times_calibrate_structure_only-sft4-add05/merged \
                                    --save-model-path checkpoints/checkpoints-$MODEL_TYPE/$OUTPUT_DIR/merged



sh script/34B-V13/eval_reason_all.sh