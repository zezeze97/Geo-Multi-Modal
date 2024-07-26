#!/bin/bash
sh script/34B-V10/eval_geo_multi.sh formalgeo_test_geoqa_qs > eval_logs/34B-V10/q2ans.log 2>&1
sh script/34B-V10/eval_geo_multi.sh formalgeo_test_geoqa_qs_q2cdl_and_ans > eval_logs/34B-V10/q2cdl_and_ans.log 2>&1
sh script/34B-V10/eval_geo_multi.sh formalgeo_test_geoqa_qs_q_and_cdl2ans > eval_logs/34B-V10/q_and_cdl2ans.log 2>&1
sh script/34B-V10/eval_geo_multi.sh formalgeo_test_geoqa_qs_choice > eval_logs/34B-V10/q2ans_choice.log 2>&1
sh script/34B-V10/eval_geo_multi.sh formalgeo_test_geoqa_qs_q2cdl_and_ans_choice > eval_logs/34B-V10/q2cdl_and_ans_choice.log 2>&1 
sh script/34B-V10/eval_geo_multi.sh formalgeo_test_geoqa_qs_q_and_cdl2ans_choice > eval_logs/34B-V10/q_and_cdl2ans_choice.log 2>&1 
sh script/34B-V10/eval_geo_multi_cascade.sh formalgeo_test_geoqa_qs_meta Q+PredCDL2Ans > eval_logs/34B-V10/cascade_q_and_Predcdl2ans.log 2>&1
sh script/34B-V10/eval_geo_multi_cascade.sh formalgeo_test_geoqa_qs_meta_choice Q+PredCDL2Ans > eval_logs/34B-V10//cascade_q_and_Predcdl2ans_choice.log 2>&1
sh script/34B-V10/eval_geo_multi_cascade.sh formalgeo_test_geoqa_qs_meta Q+PredCDL2CalibrateCDLandAns > eval_logs/34B-V10/cascade_q_and_Predcdl2cdl_and_ans.log 2>&1
sh script/34B-V10/eval_geo_multi_cascade.sh formalgeo_test_geoqa_qs_meta_choice Q+PredCDL2CalibrateCDLandAns > eval_logs/34B-V10/cascade_q_and_Predcdl2cdl_and_ans_choice.log 2>&1
sh script/34B-V10/eval_geo_translate_multi.sh > eval_logs/34B-V10/translate.log 2>&1