nohup sh script/formalizer-addV4/eval_geo_caption_multi.sh consCDL_and_imgCDL_calibrate > structure_eval_log/consCDL_and_imgCDL_calibrate.log 2>&1 &
nohup sh script/formalizer-addV4/eval_geo_caption_multi.sh consCDL_and_imgCDL_natrualAug_calibrate > structure_eval_log/consCDL_and_imgCDL_natrualAug_calibrate.log 2>&1 &
nohup sh script/formalizer-addV4/eval_geo_caption_multi.sh consCDL_and_imgCDL_natrualAug > structure_eval_log/consCDL_and_imgCDL_natrualAug.log 2>&1 &
nohup sh script/formalizer-addV4/eval_geo_caption_multi.sh consCDL_and_imgCDL > structure_eval_log/consCDL_and_imgCDL.log 2>&1 &


nohup sh script/formalizer-addV4/eval_geo_caption_multi.sh only_consCDL_calibrate > structure_eval_log/only_consCDL_calibrate.log 2>&1 &
nohup sh script/formalizer-addV4/eval_geo_caption_multi.sh only_consCDL_natrualAug_calibrate > structure_eval_log/only_consCDL_natrualAug_calibrate.log 2>&1 &
nohup sh script/formalizer-addV4/eval_geo_caption_multi.sh only_consCDL_natrualAug > structure_eval_log/only_consCDL_natrualAug.log 2>&1 &
nohup sh script/formalizer-addV4/eval_geo_caption_multi.sh only_consCDL > structure_eval_log/only_consCDL.log 2>&1 &

nohup sh script/formalizer-addV4/eval_geo_caption_multi.sh only_imgCDL_calibrate > structure_eval_log/only_imgCDL_calibrate.log 2>&1 &
nohup sh script/formalizer-addV4/eval_geo_caption_multi.sh only_imgCDL_natrualAug_calibrate > structure_eval_log/only_imgCDL_natrualAug_calibrate.log 2>&1 &
nohup sh script/formalizer-addV4/eval_geo_caption_multi.sh only_imgCDL_natrualAug > structure_eval_log/only_imgCDL_natrualAug.log 2>&1 &
nohup sh script/formalizer-addV4/eval_geo_caption_multi.sh only_imgCDL > structure_eval_log/only_imgCDL.log 2>&1 &

nohup sh script/formalizer-addV4/eval_geo_caption_multi.sh train_qs_naturalAug_calibrate > structure_eval_log/train_qs_naturalAug_calibrate.log 2>&1 &
nohup sh script/formalizer-addV4/eval_geo_caption_multi.sh formalgeo_qs_naturalAug_calibrate > structure_eval_log/formalgeo_qs_naturalAug_calibrate.log 2>&1 &