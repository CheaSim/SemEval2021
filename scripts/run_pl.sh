export TASK="semeval"
export DATA_DIR=./dataset/task1/
export MAX_LENGTH=512
export LEARNING_RATE=1e-5
export BERT_MODEL="/home/xx/pretrained_model/roberta-base"
export BATCH_SIZE=1
export NUM_EPOCHS=8
export SEED=7
export OUTPUT_DIR_NAME=semeval-pl-roberta
export CURRENT_DIR=${PWD}
export OUTPUT_DIR=${CURRENT_DIR}/${OUTPUT_DIR_NAME}

# Make output directory if it doesn't exist
mkdir -p $OUTPUT_DIR
# Add parent directory to python path to access lightning_base.py
export PYTHONPATH="../":"${PYTHONPATH}"

echo $OUTPUT_DIR

CUDA_VISIBLE_DEVICES=0 python3 main.py --gpus 1 --data_dir $DATA_DIR \
--task $TASK \
--model_name_or_path $BERT_MODEL \
--output_dir $OUTPUT_DIR \
--max_seq_length  $MAX_LENGTH \
--learning_rate $LEARNING_RATE \
--num_train_epochs $NUM_EPOCHS \
--train_batch_size $BATCH_SIZE \
--seed $SEED \
--do_train \
--do_predict
