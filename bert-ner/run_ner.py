"""Main file for model training."""
import logging
import os
import sys
from typing import Dict, List, Optional, Tuple

import time
import pickle
import tqdm
import numpy as np
from torch import nn
import torch

from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelWithLMHead,
    EvalPrediction,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)
from preprocess import DataProcessor
from datautils import NerDataset
from modeling import NERModel
from metricsutils import (
    compute_accuracy_labels, write_metrics,
    TOKEN_ACCURACY, SPAN_ACCURACY, MEAN_TOKEN_PRECISION,
    MEAN_TOKEN_RECALL, MEAN_SPAN_PRECISION, MEAN_SPAN_RECALL
)
from utils import ModelArguments, DataTrainingArguments
from utils import featureName2idx
from itertools import chain
import ray
try:
    ray.init(ignore_reinit_error=True, address="auto")
except Exception:
    ray.init(ignore_reinit_error=True)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
pad_token_label_id: int = nn.CrossEntropyLoss().ignore_index


class NERTrainer(Trainer):
    """Inherit from Trnasformers Trainer and support NER training."""

    def training_step(self, model, inputs):
        """Training step, capture failure."""
        try:
            return super().training_step(model, inputs)
        except Exception:
            import ipdb
            ipdb.set_trace()
            loss = torch.tensor(0.0).to(self.args.device)
            return loss

    def prediction_step(self, *args, **kwargs):
        """Prediction step, calculate span metrics."""
        loss, logits, labels = super().prediction_step(*args, **kwargs)
        if type(logits) is tuple:
            others = logits[1:]
            logits = logits[0]
        else:
            others = None
        b, l, c = logits.shape
        maxl = self.train_dataset.max_seq_length
        labels = torch.cat([labels, torch.zeros((b, maxl - l)).fill_(pad_token_label_id).to(labels)], dim=1)
        logits = torch.cat([logits, torch.zeros((b, maxl - l, c)).fill_(-20000.0).to(logits)], dim=1)
        if others is not None:
            return loss, (logits,) + others, labels
        else:
            return loss, logits, labels


def main(args):
    """Training/validation/profile's main entry."""
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    path = training_args.output_dir

    try:
        os.mkdir(path)
    except OSError:
        print("Creation of the directory %s failed" % path)
    else:
        print("Successfully created the directory %s " % path)
    logger_name = os.path.join(training_args.output_dir, "log.txt")
    LOG_FORMAT = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
    logging.basicConfig(format=LOG_FORMAT, level=logging.INFO, filename=logger_name, filemode='w')
    if model_args.feature_names is not None:
        nerProcessor = DataProcessor(
            data_args.data_dir,
            features=model_args.feature_names.split(','))
    else:
        nerProcessor = DataProcessor(data_args.data_dir)
    features_dim = nerProcessor.get_features_dim()
    label_list = nerProcessor.get_labels()
    label_map = nerProcessor.get_label_map()
    inversed_label_map = nerProcessor.get_invsered_label_map()
    train_examples = nerProcessor.get_examples(os.path.join(data_args.data_dir, data_args.train_split))
    if data_args.use_da:
        train_examples.extend(nerProcessor.get_examples(os.path.join(data_args.data_dir, "aug_train.txt")))
    train_examples_wei = None
    if data_args.weak_file is not None:
        if os.path.isfile(data_args.weak_file):
            _weak_file = data_args.weak_file
        else:
            _weak_file = os.path.join(data_args.data_dir, data_args.weak_file)
        weak_examples = nerProcessor.get_examples(_weak_file)
        if data_args.weak_dropo:
            nerProcessor.dropO(weak_examples)
        if data_args.weak_only:
            train_examples = []
        if data_args.weak_wei_file is not None:
            train_examples_wei = [1] * len(train_examples)
            train_examples_wei.extend(list(np.load(data_args.weak_wei_file, allow_pickle=True)))
            assert len(train_examples) + len(weak_examples) == len(train_examples_wei)
        train_examples.extend(weak_examples)
    if train_examples_wei is None:
        train_examples_wei = [1] * len(train_examples)
    train_examples_wei = [min(data_args.max_weight, w) for w in train_examples_wei]
    dev_examples = nerProcessor.get_examples(os.path.join(data_args.data_dir, data_args.dev_split))
    test_examples = nerProcessor.get_examples(os.path.join(data_args.data_dir, data_args.test_split))
    set_seed(training_args.seed)
    num_labels = len(label_list)

    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path
        if model_args.config_name_or_path is None
        else model_args.config_name_or_path,
        num_labels=num_labels
    )

    # always overwrite loss function
    config.loss_func = model_args.loss_func

    if not hasattr(config, 'features_dict'):
        config.features_dict = {}
        if model_args.feature_names is not None:
            feature_names = model_args.feature_names.split(',')
            feature_list = [int(featureName2idx[feature]) for feature in feature_names]
            for feature_idx in feature_list:
                config.features_dict[feature_idx] = model_args.feature_dim
    if not hasattr(config, 'features_dim'):
        config.features_dim = features_dim

    if not hasattr(config, 'use_cnn'):
        config.use_cnn = model_args.use_cnn
    if not hasattr(config, 'cnn_kernels'):
        config.cnn_kernels = model_args.cnn_kernels
    if not hasattr(config, 'cnn_out_channels'):
        config.cnn_out_channels = model_args.cnn_out_channels

    if not hasattr(config, 'use_crf'):
        config.use_crf = model_args.use_crf
    if data_args.weak_wei_file is not None:
        assert config.use_crf, "not implemented for non crf model"
    if config.use_crf:
        config.loss_func = model_args.crf_loss_func

    config.inversed_label_map = inversed_label_map

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path
        if model_args.tokenizer_name_or_path is None
        else model_args.tokenizer_name_or_path,
        cache_dir=None,
        use_fast=False,
        config=config,
    )

    load_name_or_path = model_args.model_name_or_path \
        if model_args.lm_model_name_or_path is None \
        else model_args.lm_model_name_or_path

    model = NERModel.from_pretrained(
        load_name_or_path,
        from_tf=bool(".ckpt" in load_name_or_path),
        config=config,
        cache_dir=None,
    )

    label_map['[CLS]'] = pad_token_label_id
    label_map['[SEP]'] = pad_token_label_id
    label_map['X'] = pad_token_label_id

    train_dataset = NerDataset(train_examples, tokenizer, label_map, data_args.max_seq_length, train_examples_wei)
    dev_dataset = NerDataset(dev_examples, tokenizer, label_map, data_args.max_seq_length)
    test_dataset = NerDataset(test_examples, tokenizer, label_map, data_args.max_seq_length)

    def align_predictions(predictions: np.ndarray, label_ids: np.ndarray) -> Tuple[List[int], List[int]]:
        preds = np.argmax(predictions, axis=2)

        batch_size, seq_len = preds.shape
        chunksize = batch_size // 100 + 1

        @ray.remote
        def f(k, label_ids, preds):
            r = []
            for i in range(chunksize * k, min(batch_size, chunksize * (k + 1))):
                preds_list_sub = []
                for j in range(seq_len):
                    if label_ids[i, j] != nn.CrossEntropyLoss().ignore_index:
                        preds_list_sub.append(inversed_label_map[preds[i][j]])
                r.append(preds_list_sub)
            return r
        preds_list = ray.get([f.remote(i, label_ids, preds) for i in range(100)])
        preds_list = list(chain.from_iterable(preds_list))
        assert len(preds_list) == batch_size

        @ray.remote
        def f(k, label_ids, preds):
            r = []
            for i in range(chunksize * k, min(batch_size, chunksize * (k + 1))):
                out_label_list_sub = []
                for j in range(seq_len):
                    if label_ids[i, j] != nn.CrossEntropyLoss().ignore_index:
                        out_label_list_sub.append(inversed_label_map[label_ids[i][j]])
                r.append(out_label_list_sub)
            return r
        out_label_list = ray.get([f.remote(i, label_ids, preds) for i in range(100)])
        out_label_list = list(chain.from_iterable(out_label_list))

        return preds_list, out_label_list

    def compute_metrics(p: EvalPrediction, full: Optional[bool] = False, skip: Optional[bool] = False) -> Dict:
        if skip:
            return {'skip': True}
        if type(p.predictions) is tuple:
            logits = p.predictions[0]
        else:
            logits = p.predictions
        begin_time = time.time()
        preds_list, out_label_list = align_predictions(logits, p.label_ids)
        align_time = time.time() - begin_time
        metrics = compute_accuracy_labels(out_label_list, preds_list, bio_format=True, ignore_labels=set(["category"]))
        result = {
            "TOKEN_ACCURACY": metrics[TOKEN_ACCURACY],
            "SPAN_ACCURACY": metrics[SPAN_ACCURACY],
            "MEAN_TOKEN_PRECISION": metrics[MEAN_TOKEN_PRECISION],
            "MEAN_TOKEN_RECALL": metrics[MEAN_TOKEN_RECALL],
            "MEAN_SPAN_PRECISION": metrics[MEAN_SPAN_PRECISION],
            "MEAN_SPAN_RECALL": metrics[MEAN_SPAN_RECALL],
        }
        if full:
            result["metrics"] = metrics
        total_time = time.time() - begin_time
        result['align_time'] = align_time
        result['total_time'] = total_time
        return result

    # Initialize our Trainer
    trainer = NERTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        compute_metrics=compute_metrics,
        data_collator=NerDataset.dynamic_collator(tokenizer.pad_token_id)
    )

    if training_args.do_train:
        trainer.train(
            model_path=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None
        )
        trainer.save_model()
        # For convenience, we also re-save the tokenizer to the same directory,
        # so that you can share your model easily on huggingface.co/models =)
        if trainer.is_world_process_zero():
            tokenizer.save_pretrained(training_args.output_dir)

    if not data_args.no_eval:
        print("*******************************")
        print("**********Evaluation***********")
        print("*******************************")
        trainer.compute_metrics = lambda x: compute_metrics(x, full=True)
        test_pred = trainer.predict(test_dataset)

        if training_args.do_eval:
            metric_file_path = os.path.join(model_args.model_name_or_path, data_args.metric_file_path)
        else:
            metric_file_path = os.path.join(training_args.output_dir, data_args.metric_file_path)
        with open(metric_file_path, 'w') as fout:
            write_metrics(fout, test_pred.metrics["eval_metrics"])

    if data_args.do_profile:
        print("*******************************")
        print("**********Profiling************")
        print("*******************************")
        trainer.compute_metrics = lambda x: compute_metrics(x, skip=True)
        if data_args.profile_file == "dev":
            profile_examples = dev_examples
            profile_tag = "dev"
        elif data_args.profile_file == "test":
            profile_examples = test_examples
            profile_tag = "test"
        elif data_args.profile_file == "train":
            profile_examples = train_examples
            profile_tag = "train"
        else:
            profile_examples = nerProcessor.get_examples(data_args.profile_file)
            profile_tag = data_args.profile_file.split("/")[-1].split(".")[0]
        chunk_size = 400000
        profile_data = []
        trainer.model.set_returning(['nll'])
        for exid in tqdm.tqdm(range(0, len(profile_examples), chunk_size)):
            chunk_examples = profile_examples[exid:min(len(profile_examples), exid + chunk_size)]
            profile_dataset = NerDataset(chunk_examples, tokenizer, label_map, data_args.max_seq_length)
            profile_predict = trainer.predict(profile_dataset)
            if trainer.is_world_process_zero():
                predictions = profile_predict.predictions
                logits, nll_list = predictions
                if config.use_crf:
                    score = logits.max(-1).max(-1)
                else:
                    score = -nll_list
                label_ids = profile_predict.label_ids
                preds_list, out_label_list = align_predictions(logits, label_ids)
                chunk_profile_data = [(s, s / len(ps), nll, ps == ls, ps, ls, e.words)
                                      for ps, ls, s, nll, e in zip(preds_list, out_label_list, score, nll_list, chunk_examples)]
                decrepted_samples = len([x for x in chunk_profile_data if len(x[-1]) != len(x[-2])])
                if decrepted_samples > 0:
                    print(f"Warning: {decrepted_samples} instances are cutted off but still save all labels, words")
                profile_data.extend(chunk_profile_data)
        if trainer.is_world_process_zero():
            assert len(profile_data) == len(profile_examples)
            pickle.dump(profile_data, open(os.path.join(training_args.output_dir, profile_tag + 'profile_data.pickle'), 'wb'))
            print("Query Level Accuracy:  {:.4f}".format(sum([x[3] for x in profile_data]) / len(profile_data)))
        trainer.model.set_returning([])

    def save_rule(rule, pred, label):
        if "-" in rule:
            rule = rule.split('-')[1]
        if rule is None or rule == 'no':
            return pred
        elif rule == 'non_O_overwrite':
            if label != 'O':
                return label
            else:
                return pred
        elif rule == 'all_overwrite':
            return pred
        else:
            raise NotImplementedError(rule + ' not implemented')

    def screen_rule(rule, ps, ls):
        ori_rule = rule
        if rule is None or rule == 'no' or '-' not in rule:
            return True
        else:
            rule = rule.split("-")[0]
            if rule == 'drop_allmatch':
                for p, l in zip(ps, ls):
                    if l != 'O' and p != l:
                        return True
                return False
            if rule == 'drop_allmatch_error':
                prevp = None
                for p, l in zip(ps, ls):
                    p = save_rule(ori_rule, p, l)
                    if p.startswith('I-'):
                        if prevp != p and prevp != p.replace('I-', 'B-'):
                            return False
                    prevp = p
                for p, l in zip(ps, ls):
                    if l != 'O' and p != l:
                        return True
                return False
            else:
                raise NotImplementedError(rule + " not implemented")

    if data_args.pred_file is not None:
        print("*******************************")
        print("**********Prediction***********")
        print("*******************************")
        trainer.compute_metrics = lambda x: compute_metrics(x, skip=True)
        pred_examples = nerProcessor.get_examples(data_args.pred_file)
        chunk_size = 400000
        preds_list = []
        out_label_list = []
        for exid in tqdm.tqdm(range(0, len(pred_examples), chunk_size)):
            pred_dataset = NerDataset(pred_examples[exid:min(len(pred_examples), exid + chunk_size)],
                                      tokenizer, label_map, data_args.max_seq_length)
            prediction = trainer.predict(pred_dataset)
            if trainer.is_world_process_zero():
                chunk_preds_list, chunk_out_label_list = align_predictions(prediction.predictions, prediction.label_ids)
                preds_list.extend(chunk_preds_list)
                out_label_list.extend(chunk_out_label_list)
        if trainer.is_world_process_zero():
            assert len(preds_list) == len(pred_examples)
            total_error_nums = 0
            total_nomatch_nums = 0
            total_save_nums = 0
            with open(data_args.save_pred_file, 'w') as fout:
                assert len(preds_list) == len(pred_examples)
                for ps, ls, es in tqdm.tqdm(zip(preds_list, out_label_list, pred_examples)):
                    if not screen_rule(data_args.save_pred_rule, ps, ls):
                        continue
                    total_save_nums += 1
                    prevp = 'O'
                    preve = None
                    assert len(ps) == len(ls) == len(es.words)
                    for p, l, e in zip(ps, ls, es.words):
                        if p != l and l != 'O':
                            total_nomatch_nums += 1
                            print('Not Match Weak {} -> Pred {} \t\t {} \t\t {}'.format(l, p, e, " ".join(es.words)))
                        p = save_rule(data_args.save_pred_rule, p, l)
                        if p.startswith('I-'):
                            if prevp != p and prevp != p.replace('I-', 'B-'):
                                print('Error {} -> {} \t\t {} -> {} \t\t {}'.format(prevp, p, preve, e, " ".join(es.words)))
                                total_error_nums += 1
                        prevp = p
                        preve = e
                        fout.write("{}\t{}\n".format(e, p))
                    fout.write("\n")
            print("Total # of Saves ", total_save_nums, " / ", len(preds_list))
            print("Total # of Errors ", total_error_nums)
            print("Total # of Not Match Weak ", total_nomatch_nums)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
