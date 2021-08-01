# BERT NER

## File Structure

```
├── utils.py            # Utilities for Training/Model Arguments
├── crfutils.py         # Utilities for CRF-Layer
├── datautils.py        # Utilities for NER Dataset
├── metricsutils.py     # Utilities for NER Metrics
├── preprocess_Appen.py # Utilities for Processing Appen like NER file
├── loss.py             # Utilities for some additional loss functions (not tested)
├── modeling.py         # Definition of NER Models (a genuine class that add functionality to any AutoModelForTokenClassification, e.g. CRF)
├── run_ner_appen.py    # Main python file for training
├── profile2refinedweakdata.py # Python Script for Turning Profiling Data into Self-training Data
└── run_language_modeling.py  # Main python file for domain continual pre-training with masked language model
```

Here we use the name `appen` for denoting the training data should follows the Appen format. Appen format is essential the same as CoNLL03 format, but for each sentence/query the first line is a category label, e.g., :
```
aps B-category
nike  B-brand
running B-productType
shoes I-productType
```
The category label is ignored by default.
