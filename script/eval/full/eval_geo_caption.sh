CUDA_VISIBLE_DEVICES=0 python bunny/eval/model_vqa.py --model-type qwen2 \
                                                    --model-path checkpoints/checkpoints-qwen2/bunny-lora-qwen2-caption-formalgeov2-construction-cdl-and-image-cdl-0.5B-sft50/merged \
                                                    --vision_encoder_path checkpoints/checkpoints-qwen2/bunny-lora-qwen2-caption-formalgeov2-construction-cdl-and-image-cdl-0.5B-sft50/merged \
                                                    --question-file data/formalgeo7k/formalgeo7k_v2/structure_only_test_qs.jsonl \
                                                    --answers-file ./outputs/bunny-lora-qwen2-caption-formalgeov2-construction-cdl-and-image-cdl-0.5B-sft50/result.jsonl \
                                                    --num-chunks 1 \
                                                    --chunk-idx 0 \
                                                    --image-folder /research/zhangzr/Bunny/data/formalgeo7k/formalgeo7k_v2 \
                                                    --temperature 0 \
                                                    --conv-mode bunny