#!/bin/bash

PROJECT_ROOT=$(dirname "$(readlink -f "$0")")/..
# TASK="BC5CDR-disease"
# TASK="NCBI-disease"
TASK="BC5CDR-chem"
BERT_CKP=${TASK}/crf-dmis-lab-biobert-v1.1_EPOCH_5_gpu_1
BERT_MODEL_PATH=${PROJECT_ROOT}/outputs/bio-ner/${BERT_CKP}
# WEI_RULE="wei_accu_pairs-0.2"
# WEI_RULE=corrected_weak_non_O_promote
# WEI_RULE=avgaccu
# WEI_RULE=uni
WEI_RULE=avgaccu_weak_non_O_promote
# PRED_RULE=non_O_overwrite_over_accu_30
# PRED_RULE=non_O_overwrite_all_overwrite_over_accu_95
# PRED_RULE=no
# PRED_RULE=all_overwrite
PRED_RULE=non_O_overwrite
WEAK_FILE=weak

python $PROJECT_ROOT/bert-ner/profile2refinedweakdata.py \
  --bertckp $BERT_MODEL_PATH \
  --wei_rule $WEI_RULE \
  --pred_rule $PRED_RULE \
  --weak_file $WEAK_FILE
