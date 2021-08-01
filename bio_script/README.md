# Shell Scripts For BioMedical NER



- [Shell Scripts For BioMedical NER](#shell-scripts-for-biomedical-ner)
	- [Ray](#ray)
	- [Data Pre-Processing](#data-pre-processing)
	- [File Structure](#file-structure)
	- [Biomedical NER Specific Parameters](#biomedical-ner-specific-parameters)
	- [Performance Benchmark](#performance-benchmark)
	- [Reproducing Results](#reproducing-results)
		- [BC5CDR-chem](#bc5cdr-chem)
		- [BC5CDR-disease](#bc5cdr-disease)
		- [NCBI-disease](#ncbi-disease)

## Ray
You may wnat to start ray service, e.g., 
```bash
ray start --head --port=6379
```

## Data Pre-Processing

To reproduce the data processing:
**Human Labeled Data**:
We download the NER data from [BLURB](https://microsoft.github.io/BLURB/submit.html) [zip](https://microsoft.github.io/BLURB/sample_code/data_generation.tar.gz) (`BC5CDR-chem`, `BC5CDR-disease`, `NCBI-disease` in `data_generation/data/`). I think this is exactly the same as this [github repo](https://github.com/cambridgeltl/MTL-Bioinformatics-2016).

Run `sh ./data/download_labeled.sh`

**Weakly Supervised Data**:
```bash
# Put `data/download_pubmed.sh` into the directories you want to save the data before running it. It takes a long time to run and require large disc space.
mkdir tasks/unlabeled
cp data/download_pubmed.sh tasks/unlabeled/

# Unlabeled in-domain data: Dump PubMed using `data/download_pubmed.sh`.
# You may want to change the name & number of files in the scripts due to change of annual baseline of pubmed (i.e., `GZFILE`, `XMLFILE` and `$(seq -f "%04g" 1 1015)`). See current year's data in https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/
cd tasks/unlabeled/
sh download_pubmed.sh

cat *.txt > all_text
rm *.txt
mv all_text all_text.txt

# Put `data/Annotate.ipynb`, `data/chem_dict.txt`, `data/disease_dict.txt` into the directory.
cd ../..
cp data/Annotate.ipynb tasks/unlabeled/
cp data/chem_dict.txt tasks/unlabeled/
cp data/disease_dict.txt tasks/unlabeled/
```
Annotate data using dictionary `Annotate.ipynb`. Change `TGT_ENTITY_TYPE` for generating data for different tasks, see details in the notebook. 

```
# create soft link
cd BC5CDR-chem
ln -s ../unlabeled/chem_weak.txt weak.txt
cd ../BC5CDR-disease
ln -s ../unlabeled/disease_weak.txt weak.txt
cd ../NCBI-disease
ln -s ../unlabeled/disease_weak.txt weak.txt
```


## File Structure
```.
├── data                        # Scripts for pre-processing
│   ├── download_labeled.sh     # Download PubMed Data
│   ├── download_pubmed.sh      # Download PubMed Data
│   ├── Annotate.ipyb           # Get weak annotation
│   ├── disease_dict.sh         # Dictionary Files for Disease
│   └── chem_dict.txt           # Dictionary Files for Chemical
├── weak_weighted_selftrain.sh   # Script for weakly-supervised training (Stage II)
├── finetune.sh                 # Fine-tune a model with human labeled data
├── profile.sh                  # Create profile data
├── profile2refinedweakdata.sh     # Turn profile data into refined weakly supervised data (weak label refinement)
└── supervised.sh               # Fine-tune from a publicly avaliable pre-trained model (e.g., BioBERT)
```

## Biomedical NER Specific Parameters

Before using any script in this folder, you need to set `TASK` from one of the following:
```
TASK="BC5CDR-chem"
TASK="BC5CDR-disease"
TASK="NCBI-disease"
```

The default max length is 256: `MAX_LENGTH=256`

Other Parameters see [Hyperparameter Explaination in ../README.md](../README.md#hyperparameter-explaination)

## Performance Benchmark


|Method | BC5CDR-chem | BC5CDR-disease | NCBI-disease |
|-------|-------------|----------------|--------------|
|Previous SOTA (F1-score) ||||
|BERT	          |89.99	|79.92	|85.87|
|bioBERT        |92.85	|84.70 	|89.13|
|SciBERT	      |92.51	|84.70  |88.25|
|ClinicalBERT	  |90.80  |83.04	|86.32|
|BlueBERT	      |91.19	|83.69	|88.04|
|PubMedBERT	    |93.33	|85.62	|87.82|
|Reimp (P/R/F1)||||
|BioBERT |92.64/93.28/92.96	|83.73/86.80/85.23	|87.18/91.35/89.22|
|**Ours**|93.21/95.12/**94.17**	|87.99/93.56/**90.69**	|91.76/92.81/**92.28**|



## Reproducing Results

Since BioBERT is already an in-domain BERT, we do not do additional MLM pre-train.
An example script for MLM pre-training is in `roberta_mlm_pretrain.sh`

### Supervised Baseliens

`sh ./bio_script/supervised.sh`

### Example for BC5CDR-chem

**Stage II:**

*Initial NER model*

`./bio_script/supervised.sh 0,1` with
```
TASK="BC5CDR-chem"
NUM_EPOCHS=20
BATCH_SIZE=32
```

*Profiling*

`./bio_script/profile.sh 0,1` with
```
TASK="BC5CDR-chem"
BERT_CKP=${TASK}/crf-dmis-lab-biobert-v1.1_EPOCH_20_BSZ_32
PROFILE_FILE=$DATA_DIR/dev.txt
```
`./bio_script/profile.sh 0,1,2,3,4,5,6,7` with
```
TASK="BC5CDR-chem"
BERT_CKP=${TASK}/crf-dmis-lab-biobert-v1.1_EPOCH_20_BSZ_32
PROFILE_FILE=$DATA_DIR/weak.txt
```

*Refine Weak Labels*

`./bio_script/profile2refinedweakdata.sh` with
```
TASK="BC5CDR-chem"
BERT_CKP=${TASK}/crf-dmis-lab-biobert-v1.1_EPOCH_20_BSZ_32
WEI_RULE=avgaccu_weak_non_O_promote
PRED_RULE=non_O_overwrite
```

*NA-WSL*

`./bio_script/weak_weighted_selftrain.sh 0,1` with
```
TASK="BC5CDR-chem"
BERT_CKP=${TASK}/crf-dmis-lab-biobert-v1.1_EPOCH_20_BSZ_32
FBA_RULE=weak_non_O_overwrite-WEI_avgaccu_weak_non_O_promote
DISTRIBUTE_GPU=true
LOSSFUNC=corrected_nll
MAX_WEIGHT=0.95
USE_DA=false
```

**Stage III: fine-tune**
`./bio_script/finetune.sh 0,1` with
```
TASK="BC5CDR-chem"
BERT_CKP=${TASK}/crf-dmis-lab-biobert-v1.1_EPOCH_20_BSZ_32/selftrain/weak_non_O_overwrite-WEI_avgaccu_weak_non_O_promote_EPOCH_1_MAXWEI_0.95_LOSS_corrected_nll_distributed
NUM_EPOCHS=15
SEED=10
```

