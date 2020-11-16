
model='xlnet-base-cased'



MODEL_NAME_OR_PATH="/home/xx/pretrained_model/"${model}


SEMEVAL_DIR="./dataset/training_data"
SEMEVAL_DIR_TASK2="./dataset/task2"
# -m torch.distributed.launch --nproc_per_node=1  --nnodes=1

CUDA_VISIBLE_DEVICES=2 python \
        run_roberta.py \
        --task_name semeval \
        --model_name_or_path ${MODEL_NAME_OR_PATH} \
        --do_train \
        --do_eval \
        --data_dir $SEMEVAL_DIR \
        --learning_rate 1e-5 \
        --num_train_epochs 10 \
        --max_seq_length 480 \
        --output_dir ./output/${model}_normal \
        --save_steps 2000 \
        --per_device_eval_batch_size=8 \
        --per_device_train_batch_size=1 \
        --gradient_accumulation_steps 1 \
        --overwrite_output  \
        --evaluate_during_training   








