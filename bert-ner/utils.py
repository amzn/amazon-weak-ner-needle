"""Utilities, e.g., training/model arguments."""
from dataclasses import dataclass, field
from typing import Optional

# This one is for internal data only
featureName2idx = {}

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    tokenizer_name_or_path: str = field(
        default=None, metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name_or_path: str = field(
        default=None, metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    lm_model_name_or_path: str = field(
        default=None, metadata={"help": "Path to lm model fine-tuned on fba data"}
    )
    loss_func: str = field(
        default="CrossEntropyLoss", metadata={"help": "loss function for token classifer"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    use_fast: bool = field(default=False, metadata={"help": "Set this flag to use fast tokenization."})
    # If you want to tweak more attributes on your tokenizer, you should do it in a distinct script,
    # or just modify its tokenizer_config.json.
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    feature_names: Optional[str] = field(
        default=None, metadata={"help": "list of feature names, separated by ,"}
    )
    feature_dim: int = field(
        default=5, metadata={"help": "dimension of feature embedding for the categorical feature embedding lookup"}
    )
    log_soft: bool = field(default=False, metadata={"help": "Set this flag to use fast tokenization."})
    label_X: bool = field(default=False, metadata={"help": "Set this flag to use label X for padding tokens."})
    use_cnn: bool = field(default=False, metadata={"help": "whether to use the cnn for features."})
    cnn_kernels: Optional[str] = field(
        default="3", metadata={"help": "list of cnn kernel size, separated by ,"}
    )
    cnn_out_channels: int = field(
        default=50, metadata={"help": "cnn output channel size"}
    )
    use_crf: bool = field(default=False, metadata={"help": "whether to use the crf for decoding."})
    crf_loss_func: str = field(
        default="nll", metadata={"help": "loss function"}
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    data_dir: str = field(
        metadata={"help": "The input data dir. Should contain the .txt files for a CoNLL-2003-formatted task."}
    )
    labels: Optional[str] = field(
        metadata={"help": "Path to a file containing all labels. If not specified, CoNLL-2003 labels are used."}
    )
    train_split: str = field(
        default="train.txt", metadata={"help": "The file name of training data."}
    )
    dev_split: str = field(
        default="dev.txt", metadata={"help": "The file name of training data."}
    )
    test_split: str = field(
        default="test.txt", metadata={"help": "The file name of training data."}
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    metric_file_prefix: str = field(
        default="metrics_debug", metadata={"help": "path of the output metric file"}
    )
    metric_file_path: str = field(
        default="eval.tsv", metadata={"help": "path of the output metric file"}
    )

    # Data Augmentation
    use_da: bool = field(
        default=False, metadata={"help": "Use additional data augmentation"}
    )

    # Weak Data Control
    weak_file: str = field(
        default=None, metadata={"help": "file name of weak data"}
    )
    weak_wei_file: str = field(
        default=None, metadata={"help": "file name of weighting weak data"}
    )
    weak_dropo: bool = field(
        default=False, metadata={"help": "Turn O into X in weak"}
    )
    weak_only: bool = field(
        default=False, metadata={"help": "Not to mix appen with weak"}
    )

    # Make self-training labels
    pred_file: str = field(
        default=None, metadata={"help": "file name of predict data"}
    )
    save_pred_file: str = field(
        default=None, metadata={"help": "file name of saving predict data"}
    )
    save_pred_rule: str = field(
        default=None, metadata={"help": "rule name of saving predict data"}
    )

    # Do Profile
    do_profile: bool = field(
        default=False, metadata={"help": "if conduct profile"}
    )
    profile_file: str = field(
        default="dev", metadata={"help": "file name of profile data"}
    )

    # Turn off Eval
    no_eval: bool = field(
        default=False, metadata={"help": "turn off evaluation"}
    )

    # Weight Maximum
    max_weight: float = field(
        default=1.0, metadata={"help": "maximum weight of weighted training"}
    )
