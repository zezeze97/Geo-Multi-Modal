#!/bin/bash
#SBATCH -o job.%j.out
#SBATCH --partition=GPU80G
#SBATCH --qos=low
#SBATCH -J 9B-DPO
#SBATCH --nodes=1    
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
export MASTER_PORT=9909

# export NCCL_DEBUG=INFO
# export DEEPSPEED_LOG_LEVEL=DEBUG

echo $NCCL_SOCKET_IFNAME


export MODEL_TYPE=yi1.5
export OUTPUT_DIR=bunny-$MODEL_TYPE9B-dpo
mkdir -p checkpoints/checkpoints-$MODEL_TYPE/$OUTPUT_DIR


srun --jobid $SLURM_JOBID bash -c 'python -m torch.distributed.run \
--nproc_per_node $GPUS_PER_NODE --nnodes $SLURM_NNODES --node_rank $SLURM_PROCID \
--master_addr $MASTER_ADDR --master_port $MASTER_PORT \
bunny/train/train_dpo.py \
    --deepspeed ./script/deepspeed/zero3.json \
    --model_name_or_path 01-ai/Yi-1.5-9B-Chat \
    --model_type $MODEL_TYPE \
    --version yi \
    --data_path data_backup/vl_dpo_data/temp.json \
    --image_folder data_backup/vl_dpo_data/merged_images \
    --vision_tower google/siglip-so400m-patch14-384 \
    --freeze_vision_tower True \
    --tune_vision_tower False \
    --mm_projector_type mlp2x_gelu \
    --image_aspect_ratio pad \
    --group_by_modality_length False \
    --bf16 True \
    --output_dir checkpoints/checkpoints-$MODEL_TYPE/$OUTPUT_DIR \
    --num_train_epochs 10 \
    --per_device_train_batch_size 4 \
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
    --tf32 True \
    --model_max_length 4096 \
    --max_length 4096 \
    --max_prompt_length 2048 \
    --max_target_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to "tensorboard" | tee 2>&1 checkpoints/checkpoints-$MODEL_TYPE/$OUTPUT_DIR/log.txt'