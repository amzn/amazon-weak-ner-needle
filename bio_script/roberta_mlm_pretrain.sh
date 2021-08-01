#!/bin/bash

if [[ $# -ne 1  ]]; then
      GPUID=0,1,2,3,4,5,6,7
  else
      GPUID=$1
fi

echo "Run on GPU $GPUID"

# data
PROJECT_ROOT=$(dirname "$(readlink -f "$0")")/..
DATA_DIR=${PROJECT_ROOT}/bio_script/tasks/unlabeled/all_text.txt

# model
BERT_MODEL=roberta-large

# param
MAX_LENGTH=256
BATCH_SIZE=32
NUM_EPOCHS=3
LR=5e-5
WARMUP=0
SAVE_STEPS=30000
SEED=1

# output
OUTPUT_DIR=${PROJECT_ROOT}/outputs/${BERT_MODEL}_mlm_EPOCH_${NUM_EPOCHS}/

[ -e $OUTPUT_DIR/script   ] || mkdir -p $OUTPUT_DIR/script
cp -f $(readlink -f "$0") $OUTPUT_DIR/script

CUDA_VISIBLE_DEVICES=$GPUID python ${PROJECT_ROOT}/bert-ner/run_language_modeling.py \
    --output_dir $OUTPUT_DIR \
    --overwrite_output_dir \
    --logging_dir $OUTPUT_DIR/log \
    --model_name_or_path=$BERT_MODEL \
    --do_train \
    --block_size  $MAX_LENGTH \
    --num_train_epochs $NUM_EPOCHS \
    --train_data_file=$DATA_DIR \
    --learning_rate $LR \
    --warmup_steps $WARMUP \
    --num_train_epochs $NUM_EPOCHS \
    --per_device_train_batch_size $BATCH_SIZE \
    --per_device_eval_batch_size $BATCH_SIZE \
    --save_steps $SAVE_STEPS \
    --seed $SEED \
    --mlm \
    --line_by_line \
    --dataloader_drop_last
