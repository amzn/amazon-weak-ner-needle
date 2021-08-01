## Named Entity Recognition with Small Strongly Labeled and Large Weakly Labeled Data

[arXiv](https://arxiv.org/abs/2106.08977)

This is the code base for weakly supervised NER.

We provide a three stage framework:
- Stage I: Domain continual pre-training;
- Stage II: Noise-aware weakly supervised pre-training;
- Stage III: Fine-tuning.

In this code base, we actually provide basic building blocks which allow arbitrary combination of different stages. We also provide examples scripts for reproducing our results in BioMedical NER.

See details in [arXiv](https://arxiv.org/abs/2106.08977).


- [Weakly Supervised NER](#weakly-supervised-ner)
  - [Performance Benchmark](#performance-benchmark)
  - [Dependency](#dependency)
  - [File Structure:](#file-structure)
  - [Data](#data)
    - [Data Format](#data-format)
    - [Pre-processed Data](#pre-processed-data)
  - [Usage](#usage)
    - [Hyperparameter Explaination](#hyperparameter-explaination)
    - [More Rounds of Training, Try Different Combination](#more-rounds-of-training-try-different-combination)
    - [Automate Experiments](#automate-experiments)
    - [Others](#others)



## Performance Benchmark

**BioMedical NER**

|Method (F1) | BC5CDR-chem | BC5CDR-disease | NCBI-disease |
|-------|-------------|----------------|--------------|
|BERT	          |89.99	|79.92	|85.87|
|bioBERT        |92.85	|84.70 	|89.13|
|PubMedBERT	    |93.33	|85.62	|87.82|
|**Ours**|**94.17**	|**90.69**	|**92.28**|

See more in [bio_script/README.md](./bio_script/README.md#performance-benchmark)


## Dependency

```
pytorch==1.6.0
transformers==3.3.1
allennlp==1.1.0
flashtool==0.0.10
ray==0.8.7
```

Install requirements
```
pip install -r requirements.txt
```

(If the `allennlp` and `transformers` are incompatible, install `allennlp` first and then update `transformers`. Since we only use some small functions of `allennlp`, it should works fine. )

## File Structure:

```.
├── bert-ner          #  Python Code for Training NER models
│   └── ...
└── bio_script        #  Shell Scripts for Training BioMedical NER models
    └── ...
```

## Usage
See examples in `bio_script`

### Hyperparameter Explaination

Here we explain hyperparameters used the scripts in `./bio_script`.

#### Training Scripts:
**Scripts**
- `roberta_mlm_pretrain.sh`
- `weak_weighted_selftrain.sh`
- `finetune.sh`

**Hyperparameter**
- `GPUID`: Choose the GPU for training. It can also be specified by `xxx.sh 0,1,2,3`.
- `MASTER_PORT`: automatically constructed (avoid conflicts) for distributed training.
- `DISTRIBUTE_GPU`: use distributed training or not
- `PROJECT_ROOT`: automatically detected, the root path of the project folder.
- `DATA_DIR`: Directory of the training data, where it contains `train.txt` `test.txt` `dev.txt` `labels.txt` `weak_train.txt` (weak data) `aug_train.txt` (optional).
- `USE_DA`: if augment training data by augmentation, i.e., combine `train.txt` + `aug_train.txt` in `DATA_DIR` for training.
- `BERT_MODEL`: the model backbone, e.g., `roberta-large`. See transformers for details.
- `BERT_CKP`: see `BERT_MODEL_PATH`.
- `BERT_MODEL_PATH`: the path of the model checkpoint that you want to load as the initialization. Usually used with `BERT_CKP`.
- `LOSSFUNC`: `nll` the normal loss function, `corrected_nll` noise-aware risk (i.e., add weighted log-unlikelihood regularization: wei*nll + (1-wei)*null ).
- `MAX_WEIGHT`: The maximum weight of a sample in the loss.
- `MAX_LENGTH`: max sentence length.
- `BATCH_SIZE`: batch size per GPU.
- `NUM_EPOCHS`: number of training epoches.
- `LR`: learning rate.
- `WARMUP`: learning rate warmup steps.
- `SAVE_STEPS`: the frequency of saving models.
- `EVAL_STEPS`: the frequency of testing on validation.
- `SEED`: radnom seed.
- `OUTPUT_DIR`: the directory for saving model and code. Some parameters will be automatically appended to the path.
  - `roberta_mlm_pretrain.sh`: It's better to manually check where you want to save the model.]
  - `finetune.sh`: It will be save in `${BERT_MODEL_PATH}/finetune_xxxx`.
  - `weak_weighted_selftrain.sh`: It will be save in `${BERT_MODEL_PATH}/selftrain/${FBA_RULE}_xxxx` (see `FBA_RULE` below)

There are some addition parameters need to be set for weakly supervised learning (`weak_weighted_selftrain.sh`).
- `WEAK_RULE`: what kind of weakly supervised data to use. See [Weakly Supervised Data Refinement Script](#weakly-supervised-data-refinement-script) for details.

#### Profiling Script

**Scripts**
- `profile.sh`

Profiling scripts also use the same entry as the training script: `bert-ner/run_ner.py` but only do evaluation.

**Hyperparameter**
Basically the same as training script.
- `PROFILE_FILE`: can be `train,dev,test` or a specific path to a `txt` data. E.g.,  using Weak by
  > `PROFILE_FILE=weak_train_100.txt`
  > `PROFILE_FILE=$DATA_DIR/$PROFILE_FILE`

- `OUTPUT_DIR`: It will be saved in `OUTPUT_DIR=${BERT_MODEL_PATH}/predict/profile`

#### Weakly Supervised Data Refinement Script

**Scripts**
- `profile2refinedweakdata.sh`

**Hyperparameter**
- `BERT_CKP`: see `BERT_MODEL_PATH`.
- `BERT_MODEL_PATH`: the path of the model checkpoint that you want to load as the initialization. Usually used with `BERT_CKP`.
- `WEI_RULE`: rule for generating weight for each weak sample.
  - `uni`: all are 1
  - `avgaccu`: confidence estimate for new labels generated by `all_overwrite`
  - `avgaccu_weak_non_O_promote`: confidence estimate for new labels generated by `non_O_overwrite`
- `PRED_RULE`: rule for generating new weak labels.
  - `non_O_overwrite`: non-entity ('O') is overwrited by prediction
  - `all_overwrite`: all use prediction, i.e., self-training
  - `no`: use original weak labels
  - `non_O_overwrite_all_overwrite_over_accu_xx`: `non_O_overwrite` + if confidence is higher than `xx` all tokens use prediction as new labels

The generated data will be saved in `${BERT_MODEL_PATH}/predict/weak_${PRED_RULE}-WEI_${WEI_RULE}`
`WEAK_RULE` specified in `weak_weighted_selftrain.sh` is essential the name of folder `weak_${PRED_RULE}-WEI_${WEI_RULE}`.


### More Rounds of Training, Try Different Combination

1. To do training with weakly supervised data from any model checkpoint directory:
  - i) Set `BERT_CKP` appropriately;
  - ii) Create profile data, e.g., run `./bio_script/profile.sh` for dev set and weak set
  - iii) Generate data with weak labels from profile data, e.g., run `./bio_script/profile2refinedweakdata.sh`. You can use different rules to generate weights for each sample (`WEI_RULE`) and different rules to refine weak labels (`PRED_RULE`). See more details in `./ber-ner/profile2refinedweakdata.py`
  - iv) Do training with `./bio_script/weak_weighted_selftrain.sh`.

2. To do fine-tuning with human labeled data from any model checkpoint directory:
  - i) Set `BERT_CKP` appropriately;
  - ii) Run `./bio_script/finetune.sh`.

### Automate Experiments

We can use `autoscript.py` to do experiments in an automatic way. E.g., do grid search. See more examples in `autoscript.py` and [flashtool](https://github.com/Gatech-Flash/FlashPythonToolbox/#auto-script-running).



## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This library is licensed under the MIT-0 License. See the LICENSE file.

