sh script/eval/full/eval_geo_multi.sh formalgeo_test_geoqa_qs > logs/q2ans.log 2>&1
sh script/eval/full/eval_geo_multi.sh formalgeo_test_geoqa_qs_q2cdl_and_ans > logs/q2cdl_and_ans.log 2>&1
sh script/eval/full/eval_geo_multi.sh formalgeo_test_geoqa_qs_q_and_cdl2ans > logs/q_and_cdl2ans.log 2>&1
sh script/eval/full/eval_geo_multi.sh formalgeo_test_geoqa_qs_q_and_Predcdl2ans > logs/q_and_Predcdl2ans.log 2>&1

sh script/eval/full/eval_geo_multi.sh formalgeo_test_geoqa_qs_q_and_Predcdl2cdl_and_ans > logs/q_and_Predcdl2cdl_and_ans.log 2>&1
sh script/eval/full/eval_geo_multi.sh formalgeo_test_geoqa_qs_choice > logs/q2ans_choice.log 2>&1
sh script/eval/full/eval_geo_multi.sh formalgeo_test_geoqa_qs_q2cdl_and_ans_choice > logs/q2cdl_and_ans_choice.log 2>&1
sh script/eval/full/eval_geo_multi.sh formalgeo_test_geoqa_qs_q_and_cdl2ans_choice > logs/q_and_cdl2ans_choice.log 2>&1


sh script/eval/full/eval_geo_multi.sh formalgeo_test_geoqa_qs_q_and_Predcdl2ans_choice > logs/q_and_Predcdl2ans_choice.log 2>&1
sh script/eval/full/eval_geo_multi.sh formalgeo_test_geoqa_qs_q_and_Predcdl2cdl_and_ans_choice > logs/q_and_Predcdl2cdl_and_ans_choice.log 2>&1
sh script/eval/full/eval_geo_translate_multi.sh > logs/translate.log 2>&1

sh script/eval/full/eval_geo_multi_cascade.sh formalgeo_test_geoqa_qs_meta Q+PredCDL2Ans > logs/cascade_q_and_Predcdl2ans.log 2>&1 
sh script/eval/full/eval_geo_multi_cascade.sh formalgeo_test_geoqa_qs_meta_choice Q+PredCDL2Ans > logs/cascade_q_and_Predcdl2ans_choice.log 2>&1 
sh script/eval/full/eval_geo_multi_cascade.sh formalgeo_test_geoqa_qs_meta Q+PredCDL2CalibrateCDLandAns > logs/cascade_q_and_Predcdl2cdl_and_ans.log 2>&1
sh script/eval/full/eval_geo_multi_cascade.sh formalgeo_test_geoqa_qs_meta_choice Q+PredCDL2CalibrateCDLandAns > logs/cascade_q_and_Predcdl2cdl_and_ans_choice.log 2>&1
sh zk.sh