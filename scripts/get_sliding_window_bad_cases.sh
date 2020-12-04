MODEL_NAME_OR_PATH="./output/albert-xxlarge-v2_task1_128_sliding_window/checkpoint-40500"


CUDA_VISIBLE_DEVICES=1 python get_bad_cases.py \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --max_seq_length 128 \
    --data_dir "./dataset/task1" \
    --sliding_window  