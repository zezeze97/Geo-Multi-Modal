#!/bin/bash
#SBATCH -o job.%j.out
#SBATCH --partition=GPU80G
#SBATCH --qos=low
#SBATCH -J siglip0.4B-qwen2-0.5B-lora-dataAug10Times-calibrate-v1v2-sft4e
#SBATCH --nodes=2    
#SBATCH --ntasks=8      
#SBATCH --cpus-per-task=16 
#SBATCH --ntasks-per-node=4 
#SBATCH --gres=gpu:4  
#SBATCH --time=5-00:00:00

echo "Allocated nodes:"
scontrol show hostname $SLURM_JOB_NODELIST
echo "GPUs per node: $SLURM_GPUS_ON_NODE"

source activate bunny
export HF_ENDPOINT=https://hf-mirror.com
export MASTER_ADDR=$(hostname -s)
export MASTER_PORT=$(comm -23 <(seq 49152 65535 | sort) <(ss -tan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)
echo $MASTER_ADDR
echo $MASTER_PORT
#replaces the content of hostfile every time
function makehostfile() {
perl -e '$slots=split /,/, $ENV{"SLURM_STEP_GPUS"};
$slots=4 if $slots==0; # workaround 8 gpu machines
@nodes = split /\n/, qx[scontrol show hostnames $ENV{"SLURM_JOB_NODELIST"}];
print map { "$b$_ slots=$slots\n" } @nodes'
}
makehostfile > hostfile

MODEL_TYPE=qwen2
OUTPUT_DIR=bunny-lora-$MODEL_TYPE-qa-FormalGeoV2Aug10Times_calibrate_v1v2_structure_only-sft4
mkdir -p checkpoints/checkpoints-$MODEL_TYPE/$OUTPUT_DIR

deepspeed --num_nodes 2 --num_gpus 4 --launcher slurm --hostfile hostfile \
    bunny/train/train.py --lora_enable True --lora_r 16 --lora_alpha 32 --mm_projector_lr 2e-5 \
    --deepspeed ./script/deepspeed/zero3.json \
    --model_name_or_path Qwen/Qwen2-0.5B-Instruct \
    --model_type $MODEL_TYPE \
    --use_formalgeo_vocab_only False \
    --add_formal_tokens False \
    --force_tune_embedding False \
    --version qwen-chat \
    --data_path data/formalgeo7k/formalgeo7k_v2/custom_json/qa_structure_only/qa_structure_only_train_aug10times_calibrate_mixV1V2_en.json \
    --image_folder data \
    --customized_aug True \
    --vision_tower google/siglip-so400m-patch14-384 \
    --freeze_vision_tower False \
    --tune_vision_tower False \
    --mm_projector_type mlp2x_gelu \
    --image_aspect_ratio pad \
    --group_by_modality_length False \
    --bf16 True \
    --output_dir checkpoints/checkpoints-$MODEL_TYPE/$OUTPUT_DIR \
    --num_train_epochs 4 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 1 \
    --learning_rate 2e-4 \
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