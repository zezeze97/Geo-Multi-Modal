#!/bin/bash
#SBATCH -o job.%j.out
#SBATCH --partition=GPU80G
#SBATCH --qos=low
#SBATCH -J 34B-S2-Alignmnet
#SBATCH --nodes=2    
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=64           # number of cores per tasks
#SBATCH --gres=gpu:4  
#SBATCH --time=5-00:00:00
export GPUS_PER_NODE=4
export OMP_NUM_THREADS=64
module load cuda/11.8
module load gcc/12.2.0
module load openmpi
export SLURM_MPI_TYPE=pmi2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True 
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=eno
export NCCL_P2P_LEVEL=NVL
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=9905

# export NCCL_DEBUG=INFO
# export DEEPSPEED_LOG_LEVEL=DEBUG

echo $NCCL_SOCKET_IFNAME



export MODEL_TYPE=yi1.5
export OUTPUT_DIR=bunny-$MODEL_TYPE-34B-Chat-VisionPretrained-S2
mkdir -p checkpoints/checkpoints-pretrain/$OUTPUT_DIR

srun --jobid $SLURM_JOBID bash -c 'python -m torch.distributed.run \
    --nproc_per_node $GPUS_PER_NODE --nnodes $SLURM_NNODES --node_rank $SLURM_PROCID \
    --master_addr $MASTER_ADDR --master_port $MASTER_PORT bunny/train/train.py \
    --deepspeed ./script/deepspeed/zero3.json \
    --model_name_or_path 01-ai/Yi-1.5-34B-Chat \
    --model_type $MODEL_TYPE \
    --use_formalgeo_vocab_only False \
    --add_formal_tokens False \
    --force_tune_embedding False \
    --version yi-chat \
    --data_path data/formalgeo7k/formalgeo7k_v2/custom_json/qa_structure_only/qa_structure_only_train_aug10times_calibrate_en.json \
    --image_folder data/formalgeo7k/formalgeo7k_v2 \
    --customized_aug True \
    --vision_tower google/siglip-so400m-patch14-384 \
    --vision_tower_pretrained_local_path checkpoints/checkpoints-qwen2/bunny-lora-qwen2-qa-FormalGeoV2Aug10Times_calibrate_structure_only-sft4-add05/merged \
    --use_s2 True \
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
    --report_to "tensorboard" | tee 2>&1 checkpoints/checkpoints-pretrain/$OUTPUT_DIR/log.txt'
    
