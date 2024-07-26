#!/bin/bash
question_file=$1
qa_mode=$2

# Assign the command line arguments to variables
N=8
base_answer_path="./outputs/report//bunny-lora-128-yi1.5-34B-Chat-FormalGeoCoT-VisionPretrained-sft1e-v14/${question_file}_${qa_mode}"
gpus=(0, 1, 2, 3, 4, 5, 6, 7)  # Define the GPU IDs array


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
                                                                     --model-path checkpoints/checkpoints-yi1.5/bunny-lora-128-yi1.5-34B-Chat-FormalGeoCoT-VisionPretrained-sft1e-v14/merged \
                                                                     --vision_encoder_path checkpoints/checkpoints-yi1.5/bunny-lora-128-yi1.5-34B-Chat-FormalGeoCoT-VisionPretrained-sft1e-v14/merged \
                                                                     --question-file data/formalgeo7k/formalgeo7k_v2/custom_json/qa_resoning/geoqa_test/${question_file}.jsonl \
                                                                     --answers-file "$answer_path" \
                                                                     --num-chunks "$N" \
                                                                     --chunk-idx "$chunk_id" \
                                                                     --image-folder data/formalgeo7k/formalgeo7k_v2 \
                                                                     --temperature 0 \
                                                                     --num-beams 1 \
                                                                     --crop \
                                                                     --conv-mode yi-chat \
                                                                     --cascade-mode \
                                                                     --recong-model-type qwen2 \
                                                                     --recong-model-path checkpoints/checkpoints-qwen2/bunny-lora-qwen2-qa-FormalGeoV2Aug10Times_calibrate_structure_only-sft4-add05/merged \
                                                                     --recong-model-vision_encoder_path checkpoints/checkpoints-qwen2/bunny-lora-qwen2-qa-FormalGeoV2Aug10Times_calibrate_structure_only-sft4-add05/merged \
                                                                     --recong-conv-mode qwen-chat \
                                                                     --process-meta-qs-mode "$qa_mode" & # Q+PredCDL2Ans/Q+PredCDL2CalibrateCDLandAns

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
# 判断 question_file 变量是否包含 "choice"
if [[ "$question_file" == *"choice"* ]]; then
    # 如果包含 "choice"，使用带 --choice_mode 参数的命令
    python eval/geo/geo_formalgeo_acc_calculate.py --predictions_file "$merged_file" --save_correct_wrong --choice_mode
else
    # 如果不包含 "choice"，使用不带 --choice_mode 参数的命令
    python eval/geo/geo_formalgeo_acc_calculate.py --predictions_file "$merged_file" --save_correct_wrong
fi