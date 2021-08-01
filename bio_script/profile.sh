#!/bin/bash

GPUID=0,1,2,3,4,5,6,7
if [[ $# -ne 1  ]]; then
      GPUID=$GPUID
  else
      GPUID=$1
fi
MASTER_PORT=12${GPUID//,/}99
MASTER_PORT=${MASTER_PORT:0:4}
echo "Run on GPU $GPUID MASTER PORT $MASTER_PORT"

NUM_GPU="${GPUID//[,]}"
NUM_GPU="${#NUM_GPU}"
DISTRIBUTE_GPU=true

# data
PROJECT_ROOT=$(dirname "$(readlink -f "$0")")/..
# TASK="BC5CDR-disease" 
# TASK="NCBI-disease" 
TASK="BC5CDR-chem" 
DATA_DIR=${PROJECT_ROOT}/bio_script/tasks/${TASK}

# model
BERT_MODEL="dmis-lab/biobert-v1.1"
BERT_CKP=${TASK}/crf-dmis-lab-biobert-v1.1_EPOCH_5_gpu_1
BERT_MODEL_PATH=${PROJECT_ROOT}/outputs/bio-ner/${BERT_CKP}

# PROFILE_FILE=train.txt
# PROFILE_FILE=test.txt
PROFILE_FILE=dev.txt
PROFILE_FILE=weak.txt
PROFILE_FILE=$DATA_DIR/$PROFILE_FILE

# param
MAX_LENGTH=256
BATCH_SIZE=256
SAVE_STEPS=200000
SEED=1

# output
OUTPUT_DIR=${BERT_MODEL_PATH}/predict/profile

[ -e $OUTPUT_DIR/script   ] || mkdir -p $OUTPUT_DIR/script
cp -f $(readlink -f "$0") $OUTPUT_DIR/script

CUDA_VISIBLE_DEVICES=$GPUID python \
    $($DISTRIBUTE_GPU && echo "-m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $MASTER_PORT") \
    ${PROJECT_ROOT}/bert-ner/run_ner.py \
    --do_profile \
    --profile_file $PROFILE_FILE \
    --no_eval \
    --tokenizer_name_or_path $BERT_MODEL \
    --model_name_or_path $BERT_MODEL \
    --config_name_or_path $BERT_MODEL_PATH \
    --lm_model_name_or_path $BERT_MODEL_PATH \
    --output_dir $OUTPUT_DIR \
    --logging_dir $OUTPUT_DIR/log \
    --max_seq_length  $MAX_LENGTH \
    --per_device_train_batch_size $BATCH_SIZE \
    --per_device_eval_batch_size $BATCH_SIZE \
    --save_steps $SAVE_STEPS \
    --seed $SEED \
    --data_dir $DATA_DIR \
    --labels ./labels.txt \
    --logging_steps 100 \
    --evaluate_during_training \
    --eval_steps 200
