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
DISTRIBUTE_GPU=true

# data
PROJECT_ROOT=$(dirname "$(readlink -f "$0")")/..
TASK="BC5CDR-disease"
TASK="NCBI-disease"
TASK="BC5CDR-chem"
DATA_DIR=${PROJECT_ROOT}/bio_script/tasks/${TASK}

# WEAK_RULE=weak_non_O_overwrite_over_accu_30-WEI_avgaccu
# WEAK_RULE="weak_non_O_overwrite-WEI_wei_accu_pairs-0.5"
# WEAK_RULE="weak_non_O_overwrite-WEI_wei_accu_pairs-0.2_30-1_90-0.2"
# WEAK_RULE="weak_non_O_overwrite-WEI_wei_accu_pairs-0.2"
# WEAK_RULE=weak_all_overwrite-WEI_corrected
# WEAK_RULE=weak_non_O_overwrite-WEI_corrected_weak_non_O_promote
# WEAK_RULE=weak_non_O_overwrite-WEI_avgaccu
# WEAK_RULE=weak_non_O_overwrite_all_overwrite_over_accu_95-WEI_avgaccu_weak_non_O_promote
# WEAK_RULE=weak_all_overwrite-WEI_uni
# WEAK_RULE=weak_no-WEI_uni
# WEAK_RULE=weak_non_O_overwrite-WEI_uni
WEAK_RULE=weak_non_O_overwrite-WEI_avgaccu_weak_non_O_promote
WEAK_ONLY=false
WEAK_DROPO=false

USE_DA=true
USE_DA=false

LOSSFUNC=nll
LOSSFUNC=corrected_nll

# model
BERT_MODEL="dmis-lab/biobert-v1.1"
BERT_CKP=${TASK}/crf-dmis-lab-biobert-v1.1_EPOCH_5_gpu_1
BERT_MODEL_PATH=${PROJECT_ROOT}/outputs/bio-ner/${BERT_CKP}

# param
MAX_WEIGHT=1
MAX_WEIGHT=0.95
MAX_LENGTH=256
BATCH_SIZE=72
NUM_EPOCHS=1
LR=5e-5
WARMUP=0
SAVE_STEPS=20000
EVAL_STEPS=5000
SEED=1

# output
OUTPUT_DIR=${BERT_MODEL_PATH}/selftrain/${WEAK_RULE}_EPOCH_${NUM_EPOCHS}
if  $WEAK_ONLY ; then
    OUTPUT_DIR=${OUTPUT_DIR}_WEAKONLY
fi
if  $WEAK_DROPO ; then
    OUTPUT_DIR=${OUTPUT_DIR}_WEAKDROPO
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
    OUTPUT_DIR=${OUTPUT_DIR}_APPENDA
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

echo "OUTPUT Dir: $OUTPUT_DIR"

[ -e $OUTPUT_DIR/script   ] || mkdir -p $OUTPUT_DIR/script
cp -f $(readlink -f "$0") $OUTPUT_DIR/script 

CUDA_VISIBLE_DEVICES=$GPUID python \
    $($DISTRIBUTE_GPU && echo "-m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $MASTER_PORT") \
    ${PROJECT_ROOT}/bert-ner/run_ner.py \
    $($WEAK_ONLY && echo '--weak_only') \
    $($WEAK_DROPO && echo '--weak_dropo') \
    $($USE_DA && echo '--use_da') \
    --max_weight $MAX_WEIGHT \
    --crf_loss_func $LOSSFUNC \
    --weak_file $BERT_MODEL_PATH/predict/$WEAK_RULE/weak.txt \
    --weak_wei_file $BERT_MODEL_PATH/predict/$WEAK_RULE/weak_wei.npy \
    --tokenizer_name_or_path $BERT_MODEL \
    --model_name_or_path $BERT_MODEL \
    --config_name_or_path $BERT_MODEL_PATH \
    --lm_model_name_or_path $BERT_MODEL_PATH \
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
