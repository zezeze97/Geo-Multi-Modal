#!/bin/bash
sh script/9B-V14/eval_geo_multi.sh formalgeo_test_geoqa_qs > eval_logs/9B-V14/q2ans.log 2>&1
sh script/9B-V14/eval_geo_multi.sh formalgeo_test_geoqa_qs_q2cdl_and_ans > eval_logs/9B-V14/q2cdl_and_ans.log 2>&1
sh script/9B-V14/eval_geo_multi.sh formalgeo_test_geoqa_qs_q_and_cdl2ans > eval_logs/9B-V14/q_and_cdl2ans.log 2>&1
sh script/9B-V14/eval_geo_multi.sh formalgeo_test_geoqa_qs_choice > eval_logs/9B-V14/q2ans_choice.log 2>&1
sh script/9B-V14/eval_geo_multi.sh formalgeo_test_geoqa_qs_q2cdl_and_ans_choice > eval_logs/9B-V14/q2cdl_and_ans_choice.log 2>&1 
sh script/9B-V14/eval_geo_multi.sh formalgeo_test_geoqa_qs_q_and_cdl2ans_choice > eval_logs/9B-V14/q_and_cdl2ans_choice.log 2>&1 
sh script/9B-V14/eval_geo_multi_cascade.sh formalgeo_test_geoqa_qs_meta Q+PredCDL2Ans > eval_logs/9B-V14/cascade_q_and_Predcdl2ans.log 2>&1
sh script/9B-V14/eval_geo_multi_cascade.sh formalgeo_test_geoqa_qs_meta_choice Q+PredCDL2Ans > eval_logs/9B-V14//cascade_q_and_Predcdl2ans_choice.log 2>&1
sh script/9B-V14/eval_geo_multi_cascade.sh formalgeo_test_geoqa_qs_meta Q+PredCDL2CalibrateCDLandAns > eval_logs/9B-V14/cascade_q_and_Predcdl2cdl_and_ans.log 2>&1
sh script/9B-V14/eval_geo_multi_cascade.sh formalgeo_test_geoqa_qs_meta_choice Q+PredCDL2CalibrateCDLandAns > eval_logs/9B-V14/cascade_q_and_Predcdl2cdl_and_ans_choice.log 2>&1
sh script/9B-V14/eval_geo_translate_multi.sh > eval_logs/9B-V14/translate.log 2>&1