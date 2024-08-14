#!/bin/bash
#SBATCH -o job.%j.out
#SBATCH --partition=GPU80G
#SBATCH --qos=normal
#SBATCH -J Eval-9B-V20-withoutDiagram
#SBATCH --nodes=1    
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=64          # number of cores per tasks
#SBATCH --gres=gpu:4  
#SBATCH --time=5-00:00:00

module load cuda/11.8
module load gcc/12.2.0
module load openmpi

sh script/9B-V20/eval_geo_multi_withoutDiagram.sh formalgeo_test_qs > eval_logs/9B-V20-withoutDiagram/q2ans_withoutDiagram.log 2>&1
sh script/9B-V20/eval_geo_multi_withoutDiagram.sh formalgeo_test_qs_q2cdl_and_ans > eval_logs/9B-V20-withoutDiagram/q2cdl_and_ans_withoutDiagram.log 2>&1
sh script/9B-V20/eval_geo_multi_withoutDiagram.sh formalgeo_test_qs_q_and_cdl2ans > eval_logs/9B-V20-withoutDiagram/q_and_cdl2ans_withoutDiagram.log 2>&1
sh script/9B-V20/eval_geo_multi_withoutDiagram.sh formalgeo_test_qs_choice > eval_logs/9B-V20-withoutDiagram/q2ans_choice_withoutDiagram.log 2>&1
sh script/9B-V20/eval_geo_multi_withoutDiagram.sh formalgeo_test_qs_q2cdl_and_ans_choice > eval_logs/9B-V20-withoutDiagram/q2cdl_and_ans_choice_withoutDiagram.log 2>&1 
sh script/9B-V20/eval_geo_multi_withoutDiagram.sh formalgeo_test_qs_q_and_cdl2ans_choice > eval_logs/9B-V20-withoutDiagram/q_and_cdl2ans_choice_withoutDiagram.log 2>&1 
sh script/9B-V20/eval_geo_multi_cascade_withoutDiagram.sh formalgeo_test_qs_meta Q+PredCDL2Ans > eval_logs/9B-V20-withoutDiagram/cascade_q_and_Predcdl2ans_withoutDiagram.log 2>&1
sh script/9B-V20/eval_geo_multi_cascade_withoutDiagram.sh formalgeo_test_qs_meta_choice Q+PredCDL2Ans > eval_logs/9B-V20-withoutDiagram/cascade_q_and_Predcdl2ans_choice_withoutDiagram.log 2>&1
sh script/9B-V20/eval_geo_multi_cascade_withoutDiagram.sh formalgeo_test_qs_meta Q+PredCDL2CalibrateCDLandAns > eval_logs/9B-V20-withoutDiagram/cascade_q_and_Predcdl2cdl_and_ans_withoutDiagram.log 2>&1
sh script/9B-V20/eval_geo_multi_cascade_withoutDiagram.sh formalgeo_test_qs_meta_choice Q+PredCDL2CalibrateCDLandAns > eval_logs/9B-V20-withoutDiagram/cascade_q_and_Predcdl2cdl_and_ans_choice_withoutDiagram.log 2>&1