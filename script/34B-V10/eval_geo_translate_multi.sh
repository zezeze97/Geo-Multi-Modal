#!/bin/bash


# Assign the command line arguments to variables
N=4
base_answer_path='./outputs/bunny-lora-128-yi1.5-34B-Chat-FormalGeoCoT-VisionPretrained-sft1e-v10/test_geoqa_translate'
gpus=(0, 1, 2, 3)  # Define the GPU IDs array


# Loop over each chunk/process
for (( chunk_id=0; chunk_id<N; chunk_id++ ))
do
    # Define the answer path for each chunk
    answer_path="${base_answer_path}_${chunk_id}.jsonl"
    if [ -f "$answer_path" ]; then
        rm "$answer_path"
    fi  
    # Run the Python program in the background
    # --vision_encoder_path checkpoints/checkpoints-yi1.5/bunny-yi1.5-9B-rerun \
    # 
    CUDA_VISIBLE_DEVICES="${gpus[chunk_id]}" python bunny/eval/model_vqa.py --model-type yi1.5 \
                                                                     --model-path checkpoints/checkpoints-yi1.5/bunny-lora-128-yi1.5-34B-Chat-FormalGeoCoT-VisionPretrained-sft1e-v10/merged \
                                                                     --vision_encoder_path checkpoints/checkpoints-yi1.5/bunny-lora-128-yi1.5-34B-Chat-FormalGeoCoT-VisionPretrained-sft1e-v10/merged \
                                                                     --question-file data/formalgeo7k/formalgeo7k_v2/custom_json/qa_translate/formalgeo_test_geoqa_translate_qs.jsonl \
                                                                     --answers-file "$answer_path" \
                                                                     --num-chunks "$N" \
                                                                     --chunk-idx "$chunk_id" \
                                                                     --image-folder /research/zhangzr/Bunny/data/formalgeo7k/formalgeo7k_v2 \
                                                                     --temperature 0 \
                                                                     --crop \
                                                                     --conv-mode yi-chat &

    # Uncomment below if you need a slight delay between starting each process
    # sleep 0.1
done

# Wait for all background processes to finish
wait

merged_file="${base_answer_path}_merged.jsonl"
if [ -f "$merged_file" ]; then
    rm "$merged_file"
fi
# Merge all the JSONL files into one
#cat "${base_answer_path}"_*.jsonl > "${base_answer_path}_merged.jsonl"
for ((i=0; i<N; i++)); do
  input_file="${base_answer_path}_${i}.jsonl"
  cat "$input_file" >> "${base_answer_path}_merged.jsonl"
done
# remove the unmerged files
for (( chunk_id=0; chunk_id<N; chunk_id++ ))
do
    # Define the answer path for each chunk
    answer_path="${base_answer_path}_${chunk_id}.jsonl"
    if [ -f "$answer_path" ]; then
        rm "$answer_path"
    fi
done
python eval/evaluation_formalgeo/eval_construction_cdl_and_image_cdl.py --prediction "$merged_file"
                                    