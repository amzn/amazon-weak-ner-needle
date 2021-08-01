"""Utilities for NER data."""
from dataclasses import dataclass
from enum import Enum
from typing import List, Dict

import torch
from torch.utils import data


class InputFeatures(object):
    def __init__(self, input_ids, attention_mask, token_type_ids, label_ids, features, predict_mask, weight=1):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        # self.predict_mask = predict_mask
        # list of list of features
        self.features = features
        self.label_ids = label_ids
        self.predict_mask = predict_mask
        self.weight = weight


class Split(Enum):
    train = "train"
    dev = "dev"
    test = "test"


def get_labels(path: str) -> List[str]:
    if path:
        with open(path, "r") as f:
            labels = f.read().splitlines()
        if "O" not in labels:
            labels = ["O"] + labels
        return labels
    else:
        return ["O", "B-MISC", "I-MISC", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]


def example2feature(example, tokenizer, label_map, max_seq_length, weight=1):
    add_label = 'X'
    # tokenize_count = []
    tokens = ['[CLS]']
    predict_mask = [0]
    label_ids = [label_map['[CLS]']]
    features = []

    for i in range(len(example.features)):
        features.append([0])

    for i, w in enumerate(example.words):
        # use bertTokenizer to split words
        # 1996-08-22 => 1996 - 08 - 22
        # sheepmeat => sheep ##me ##at
        sub_words = tokenizer.tokenize(w)
        if not sub_words:
            sub_words = ['[UNK]']
        tokens.extend(sub_words)

        for j in range(len(sub_words)):
            if j == 0:
                predict_mask.append(1)
                label_ids.append(label_map[example.labels[i]])
            else:
                # '##xxx' -> 'X' (see bert paper)
                predict_mask.append(0)
                label_ids.append(label_map[add_label])
            for idx in range(len(example.features)):
                features[idx].append(example.features[idx][i])

    # truncate
    if len(tokens) > max_seq_length - 1:
        print('Example No.{} is too long, length is {}, truncated to {}!'.format(example.guid, len(tokens), max_seq_length))
        tokens = tokens[0:(max_seq_length - 1)]
        predict_mask = predict_mask[0:(max_seq_length - 1)]
        label_ids = label_ids[0:(max_seq_length - 1)]
        for i in range(len(features)):
            features[i] = features[i][0:(max_seq_length - 1)]

    tokens.append('[SEP]')
    predict_mask.append(0)
    label_ids.append(label_map['[SEP]'])
    for i in range(len(features)):
        features[i] += [0]

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    segment_ids = [0] * len(input_ids)
    input_mask = [1] * len(input_ids)

    padding_length = max_seq_length - len(input_ids)
    mask_padding_with_zero = True
    pad_token = tokenizer.pad_token_id
    pad_token_segment_id = 0
    pad_token_label_id = -100
    input_ids += [pad_token] * padding_length
    input_mask += [0 if mask_padding_with_zero else 1] * padding_length
    segment_ids += [pad_token_segment_id] * padding_length
    predict_mask += [0] * padding_length
    label_ids += [pad_token_label_id] * padding_length
    for i in range(len(features)):
        features[i] += [0] * padding_length
    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(label_ids) == max_seq_length
    for i in range(len(features)):
        assert len(features[i]) == max_seq_length

    feat = InputFeatures(
        input_ids=input_ids,
        attention_mask=input_mask,
        token_type_ids=segment_ids,
        # predict_mask=predict_mask,
        label_ids=label_ids,
        features=features,
        predict_mask=predict_mask,
        weight=weight)

    return feat


class NerDataset(data.Dataset):
    def __init__(self, examples, tokenizer, label_map, max_seq_length, weights=None):
        self.examples = examples
        self.tokenizer = tokenizer
        self.label_map = label_map
        self.max_seq_length = max_seq_length

        if weights is None:
            self.weights = [1] * len(examples)
        else:
            self.weights = weights

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        feat = example2feature(self.examples[idx], self.tokenizer, self.label_map, self.max_seq_length, self.weights[idx])
        return feat

    @classmethod
    def pad(cls, batch):
        input_ids_list = torch.LongTensor([sample.input_ids for sample in batch])
        attention_mask_list = torch.LongTensor([sample.attention_mask for sample in batch])
        token_type_ids_list = torch.LongTensor([sample.token_type_ids for sample in batch])
        label_ids_list = torch.LongTensor([sample.label_ids for sample in batch])
        features_list = torch.LongTensor([sample.features for sample in batch])
        predict_mask_list = torch.LongTensor([sample.predict_mask for sample in batch])
        weights_list = torch.Tensor([sample.weight for sample in batch])

        return input_ids_list, attention_mask_list, token_type_ids_list, label_ids_list, features_list, predict_mask_list, weights_list

    @classmethod
    def dynamic_collator(cls, pad_token_id):
        # return data collator
        @dataclass
        class DyCollator:
            pad_token_id: int

            def __call__(self, batch: List[InputFeatures]) -> Dict[str, torch.Tensor]:
                input_ids = torch.LongTensor([sample.input_ids for sample in batch])
                attention_mask = torch.LongTensor([sample.attention_mask for sample in batch])
                token_type_ids = torch.LongTensor([sample.token_type_ids for sample in batch])
                label_ids = torch.LongTensor([sample.label_ids for sample in batch])
                features = torch.LongTensor([sample.features for sample in batch])
                predict_mask = torch.LongTensor([sample.predict_mask for sample in batch])
                weights = torch.Tensor([sample.weight for sample in batch])

                # exclude broken samples
                valid_samples = (label_ids >= 0).long().sum(1) > 0
                if not valid_samples.all():
                    print("Invalid Samples Appear!!!!!")

                lens = (input_ids[valid_samples] != self.pad_token_id).long().sum(-1)
                maxlen = max(1, max(lens))

                return {"input_ids": input_ids[valid_samples, :maxlen],
                        "attention_mask": attention_mask[valid_samples, :maxlen],
                        "token_type_ids": token_type_ids[valid_samples, :maxlen],
                        "labels": label_ids[valid_samples, :maxlen],
                        "features": features[valid_samples, :maxlen],
                        "predict_mask": predict_mask[valid_samples, :maxlen],
                        "weights": weights[valid_samples],
                        }

        return DyCollator(pad_token_id=pad_token_id)
