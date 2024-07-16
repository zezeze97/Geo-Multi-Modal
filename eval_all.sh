sh script/eval/full/eval_geo_multi.sh formalgeo_test_geoqa_qs > q2ans.log 2>&1 
sh script/eval/full/eval_geo_multi.sh formalgeo_test_geoqa_qs_q2cdl_and_ans > q2cdl_and_ans.log 2>&1 
sh script/eval/full/eval_geo_multi.sh formalgeo_test_geoqa_qs_q_and_cdl2ans > q_and_cdl2ans.log 2>&1 
sh script/eval/full/eval_geo_multi.sh formalgeo_test_geoqa_qs_q_and_Predcdl2ans > q_and_Predcdl2ans.log 2>&1 
sh script/eval/full/eval_geo_multi.sh formalgeo_test_geoqa_qs_q_and_Predcdl2cdl_and_ans > q_and_Predcdl2cdl_and_ans.log 2>&1 

sh script/eval/full/eval_geo_multi.sh formalgeo_test_geoqa_qs_choice > q2ans_choice.log 2>&1 
sh script/eval/full/eval_geo_multi.sh formalgeo_test_geoqa_qs_q2cdl_and_ans_choice > q2cdl_and_ans_choice.log 2>&1 
sh script/eval/full/eval_geo_multi.sh formalgeo_test_geoqa_qs_q_and_cdl2ans_choice > q_and_cdl2ans_choice.log 2>&1 
sh script/eval/full/eval_geo_multi.sh formalgeo_test_geoqa_qs_q_and_Predcdl2ans_choice > q_and_Predcdl2ans_choice.log 2>&1 
sh script/eval/full/eval_geo_multi.sh formalgeo_test_geoqa_qs_q_and_Predcdl2cdl_and_ans_choice > q_and_Predcdl2cdl_and_ans_choice.log 2>&1 
# sh zk.sh