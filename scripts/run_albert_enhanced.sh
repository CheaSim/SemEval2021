model='albert-xxlarge-v2'

MODEL_NAME_OR_PATH="/home/xx/pretrained_model/"${model}


SEMEVAL_DIR_TASK1="./dataset/enhanced_albert_task1"
SEMEVAL_DIR_TASK2="./dataset/task2"
# TRAIN_1_DEV_2="./dataset/train_1_dev_2"

OUTPUT_DIR=./output/${model}_task2_256_ab_2021_enhanced
DATA_DIR=${SEMEVAL_DIR_TASK1}
# -m torch.distributed.launch --nproc_per_node=1  --nnodes=1
# lr = 1e-5 get the result

CUDA_VISIBLE_DEVICES=2 python \
        run_roberta.py \
        --task_name semevalenhanced \
        --model_name_or_path ${MODEL_NAME_OR_PATH} \
        --do_train \
        --do_eval \
        --eval_all_checkpoints \
        --data_dir $DATA_DIR \
        --learning_rate 5e-6 \
        --num_train_epochs 10 \
        --max_seq_length 128 \
        --output_dir  $OUTPUT_DIR \
        --logging_dir $OUTPUT_DIR \
        --save_steps 500 \
        --eval_steps 500 \
        --per_device_eval_batch_size=8 \
        --per_device_train_batch_size=1 \
        --gradient_accumulation_steps 16 \
        --evaluate_during_training  \
        --overwrite_output  \
        --overwrite_cache











# baseline
# CUDA_VISIBLE_DEVICES=2 python ./Baselines/Run_GAReader.py
