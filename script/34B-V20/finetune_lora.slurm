#!/bin/bash
#SBATCH -o job.%j.out
#SBATCH --partition=GPU80G
#SBATCH --qos=normal
#SBATCH -J 34B-V20
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
export MASTER_PORT=9909

# export NCCL_DEBUG=INFO
# export DEEPSPEED_LOG_LEVEL=DEBUG

echo $NCCL_SOCKET_IFNAME


export MODEL_TYPE=yi1.5

export PRETRAIN_DIR=bunny-$MODEL_TYPE-34B-Chat-VisionPretrained-v13-1epoch
export OUTPUT_DIR=bunny-lora-128-$MODEL_TYPE-34B-Chat-FormalGeoCoT-VisionPretrained-sft1e-v20
mkdir -p checkpoints/checkpoints-$MODEL_TYPE/$OUTPUT_DIR


srun --jobid $SLURM_JOBID bash -c 'python -m torch.distributed.run \
--nproc_per_node $GPUS_PER_NODE --nnodes $SLURM_NNODES --node_rank $SLURM_PROCID \
--master_addr $MASTER_ADDR --master_port $MASTER_PORT \
bunny/train/train.py \
--lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 1e-5 \
--model_name_or_path 01-ai/Yi-1.5-34B-Chat \
--model_type $MODEL_TYPE \
--use_formalgeo_vocab_only False \
--add_formal_tokens False \
--force_tune_embedding False \
--version yi-chat \
--data_path data_backup/formalgeo7k/formalgeo7k_v2/custom_json/qa_mixTask/qa_mix_train_v20.json \
--image_folder data_backup/formalgeo7k/formalgeo7k_v2 \
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
--report_to "tensorboard" --deepspeed ./script/deepspeed/zero3.json | tee 2>&1 checkpoints/checkpoints-$MODEL_TYPE/$OUTPUT_DIR/log.txt'