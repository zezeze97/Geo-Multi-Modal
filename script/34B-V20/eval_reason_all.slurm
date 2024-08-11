#!/bin/bash
#SBATCH -o job.%j.out
#SBATCH --partition=GPU80G
#SBATCH --qos=low
#SBATCH -J Eval-34B-V20
#SBATCH --nodes=1    
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=8           # number of cores per tasks
#SBATCH --gres=gpu:2  
#SBATCH --time=5-00:00:00

module load cuda/11.8
module load gcc/12.2.0
module load openmpi


CUDA_VISIBLE_DEVICES=0 python script/merge_lora_weights.py --model-path checkpoints/checkpoints-yi1.5/bunny-lora-128-yi1.5-34B-Chat-FormalGeoCoT-VisionPretrained-sft1e-v20 \
                                    --model-base 01-ai/Yi-1.5-34B-Chat \
                                    --model-type yi1.5 \
                                    --vision-encoder-path checkpoints/checkpoints-qwen2/bunny-lora-qwen2-qa-FormalGeoV2Aug10Times_calibrate_structure_only-sft4-add05/merged \
                                    --save-model-path checkpoints/checkpoints-yi1.5/bunny-lora-128-yi1.5-34B-Chat-FormalGeoCoT-VisionPretrained-sft1e-v20/merged

sh script/34B-V20/eval_geo_multi.sh formalgeo_test_qs > eval_logs/34B-V20/q2ans.log 2>&1
sh script/34B-V20/eval_geo_multi.sh formalgeo_test_qs_q2cdl_and_ans > eval_logs/34B-V20/q2cdl_and_ans.log 2>&1
sh script/34B-V20/eval_geo_multi.sh formalgeo_test_qs_q_and_cdl2ans > eval_logs/34B-V20/q_and_cdl2ans.log 2>&1
sh script/34B-V20/eval_geo_multi.sh formalgeo_test_qs_choice > eval_logs/34B-V20/q2ans_choice.log 2>&1
sh script/34B-V20/eval_geo_multi.sh formalgeo_test_qs_q2cdl_and_ans_choice > eval_logs/34B-V20/q2cdl_and_ans_choice.log 2>&1 
sh script/34B-V20/eval_geo_multi.sh formalgeo_test_qs_q_and_cdl2ans_choice > eval_logs/34B-V20/q_and_cdl2ans_choice.log 2>&1 
sh script/34B-V20/eval_geo_multi_cascade.sh formalgeo_test_qs_meta Q+PredCDL2Ans > eval_logs/34B-V20/cascade_q_and_Predcdl2ans.log 2>&1
sh script/34B-V20/eval_geo_multi_cascade.sh formalgeo_test_qs_meta_choice Q+PredCDL2Ans > eval_logs/34B-V20//cascade_q_and_Predcdl2ans_choice.log 2>&1
sh script/34B-V20/eval_geo_multi_cascade.sh formalgeo_test_qs_meta Q+PredCDL2CalibrateCDLandAns > eval_logs/34B-V20/cascade_q_and_Predcdl2cdl_and_ans.log 2>&1
sh script/34B-V20/eval_geo_multi_cascade.sh formalgeo_test_qs_meta_choice Q+PredCDL2CalibrateCDLandAns > eval_logs/34B-V20/cascade_q_and_Predcdl2cdl_and_ans_choice.log 2>&1