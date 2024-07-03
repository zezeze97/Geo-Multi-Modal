CUDA_VISIBLE_DEVICES=0,1 python bunny/eval/model_vqa.py --model-type yi1.5 \
                                                    --model-path checkpoints/checkpoints-yi1.5/bunny-lora-yi1.5-34B-Chat-128_rank/merged \
                                                    --vision_encoder_path checkpoints/checkpoints-yi1.5/bunny-lora-yi1.5-34B-Chat-128_rank/merged \
                                                    --question-file data/gllava-data/test_questions.jsonl \
                                                    --answers-file outputs/bunny-lora-yi1.5-34B-Chat-128_rank/result.jsonl \
                                                    --num-chunks 1 \
                                                    --chunk-idx 0 \
                                                    --image-folder data/gllava-data/images \
                                                    --temperature 0 \
                                                    --conv-mode yi-chat
python eval/geo/geo_acc_calculate.py --ground_truth_file data/gllava-data/test_answers.jsonl \
                                    --predictions_file outputs/bunny-lora-yi1.5-34B-Chat-128_rank/result.jsonl \
                                    --question_file data/gllava-data/test_questions.jsonl \
                                    --save_correct_wrong