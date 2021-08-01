#!/bin/bash

GPUID=0,1,2,3,4,5,6,7
if [[ $# -ne 1  ]]; then
      echo "using default GPUID"
  else
      GPUID=$1
fi

MASTER_PORT=12${GPUID//,/}99
MASTER_PORT=${MASTER_PORT:0:4}
echo "Run on GPU $GPUID MASTER PORT $MASTER_PORT"

NUM_GPU="${GPUID//[,]}"
NUM_GPU="${#NUM_GPU}"
DISTRIBUTE_GPU=false

# data
PROJECT_ROOT=$(dirname "$(readlink -f "$0")")/..
# TASK="NCBI-disease"
# TASK="BC5CDR-disease"
TASK="BC5CDR-chem"
DATA_DIR=${PROJECT_ROOT}/bio_script/tasks/${TASK}

USE_DA=false

# model
# BERT_MODEL=bert-base-cased
BERT_MODEL="dmis-lab/biobert-v1.1"

LOSSFUNC=corrected_nll
LOSSFUNC=nll

# param
MAX_WEIGHT=1
MAX_LENGTH=256
BATCH_SIZE=72
NUM_EPOCHS=5
LR=5e-5
WARMUP=0
SAVE_STEPS=200000
EVAL_STEPS=150
SEED=1

# output
OUTPUT_DIR=${PROJECT_ROOT}/outputs/bio-ner/${TASK}/crf-${BERT_MODEL//\//-}_EPOCH_${NUM_EPOCHS}
if [ $LR != 5e-5 ]; then
    OUTPUT_DIR=${OUTPUT_DIR}_LR_${LR}
fi
if [ $MAX_WEIGHT != 1 ]; then
    OUTPUT_DIR=${OUTPUT_DIR}_MAXWEI_${MAX_WEIGHT}
fi
if [ $WARMUP != 0 ]; then
    OUTPUT_DIR=${OUTPUT_DIR}_WUP_${WARMUP}
fi
if [ $BATCH_SIZE != 72 ]; then
    OUTPUT_DIR=${OUTPUT_DIR}_BSZ_${BATCH_SIZE}
fi
if [ $LOSSFUNC != 'nll' ]; then
    OUTPUT_DIR=${OUTPUT_DIR}_LOSS_${LOSSFUNC}
fi
if  $USE_DA ; then
    OUTPUT_DIR=${OUTPUT_DIR}_DA
fi
if [ $NUM_GPU != 2 ]; then
    OUTPUT_DIR=${OUTPUT_DIR}_gpu_${NUM_GPU}
fi
if $DISTRIBUTE_GPU ; then
  OUTPUT_DIR=${OUTPUT_DIR}_distributed
fi
if [ $SEED != 1 ]; then
    OUTPUT_DIR=${OUTPUT_DIR}_SEED_${SEED}
fi
echo $OUTPUT_DIR

[ -e $OUTPUT_DIR/script   ] || mkdir -p $OUTPUT_DIR/script
cp -f $(readlink -f "$0") $OUTPUT_DIR/script

CUDA_VISIBLE_DEVICES=$GPUID python \
    $($DISTRIBUTE_GPU && echo "-m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $MASTER_PORT") \
    ${PROJECT_ROOT}/bert-ner/run_ner.py \
    $($USE_DA && echo '--use_da') \
    --max_weight $MAX_WEIGHT \
    --crf_loss_func $LOSSFUNC \
    --use_crf \
    --model_name_or_path $BERT_MODEL \
    --output_dir $OUTPUT_DIR \
    --logging_dir $OUTPUT_DIR/log \
    --max_seq_length  $MAX_LENGTH \
    --learning_rate $LR \
    --warmup_steps $WARMUP \
    --num_train_epochs $NUM_EPOCHS \
    --per_device_train_batch_size $BATCH_SIZE \
    --per_device_eval_batch_size $BATCH_SIZE \
    --save_steps $SAVE_STEPS \
    --seed $SEED \
    --do_train \
    --data_dir $DATA_DIR \
    --labels ./labels.txt \
    --logging_steps 100 \
    --evaluate_during_training \
    --eval_steps $EVAL_STEPS
