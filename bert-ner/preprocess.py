"""Data processor for Appen format NER."""
import os

from collections import defaultdict
from utils import featureName2idx
from tqdm import tqdm


class InputNERExample(object):
    """A single training/test example for NER."""

    def __init__(self, guid, words, labels, features):
        """Initialize Example."""
        self.guid = guid
        self.words = words
        self.labels = labels
        # features is a map of feature column index to array of feature values
        self.features = features


class DataProcessor(object):
    """Data processor for NER."""

    def __init__(self, data_dir, features=[]):
        """Initialize processor."""
        self._label_types = self._read_label(os.path.join(data_dir, "labels.txt"))
        self.data_dir = data_dir
        # map of feature value to its index
        # it is {"brand" : {"None" : 0}, {"O": 0}, {"1":1}}
        self.feature_value_idx_map = defaultdict(lambda: defaultdict(int))
        # list of columns of features
        self.features = []
        for feature in features:
            self.features.append(int(featureName2idx[feature]))
        self.features = sorted(self.features)
        self._num_labels = len(self._label_types)
        self._label_map = {label: i for i, label in enumerate(self._label_types)}
        self._invsered_label_map = {i: label for i, label in enumerate(self._label_types)}
        self.feature_dim_dict = {}
        self.create_mapping()

    def get_features_dim(self):
        """Get number of features."""
        return self.feature_dim_dict

    def _read_label(self, input_file):
        label_types = []
        with open(input_file) as f:
            entries = f.read().strip().split("\n\n")
            for entry in entries:
                for line in entry.splitlines():
                    pieces = line.strip().split()
                    label = pieces[0]
                    label_types.append(label)
        return label_types

    def create_mapping(self):
        """Create feature value to idx mapping for each columns in the dataset."""
        for i in self.features:
            # self.word_level_types = {'alpha': 0, 'digit': 1, 'other': 2, 'alphanum': 3}
            if i != 1:
                self.feature_value_idx_map[i]["None"] = 0
                self.feature_value_idx_map[i]["O"] = 0
            # self.feature_value_idx_map[i]["UNKNOWN"] = 1
        with open(os.path.join(self.data_dir, "train.txt")) as f:
            entries = f.read().strip().split("\n\n")
            for entry in entries:
                lines = entry.splitlines()
                for line in lines:
                    cols = line.rstrip().split('\t')
                    if cols[0].strip() == "":
                        continue
                    for feature_col in self.features:
                        # feature column start from the second column
                        assert feature_col < len(cols)
                        if cols[feature_col] not in self.feature_value_idx_map[feature_col]:
                            self.feature_value_idx_map[feature_col][cols[feature_col]] = max(len(self.feature_value_idx_map[feature_col]) - 1, 0)
        for feature in self.features:
            if feature != 1:
                self.feature_dim_dict[feature] = len(self.feature_value_idx_map[feature]) - 1
            else:
                self.feature_dim_dict[feature] = len(self.feature_value_idx_map[feature])

    def _read_data(self, input_file):
        print("==== Reading Data ====")
        with open(input_file) as f:
            out_lists = []
            entries = f.read().strip().split("\n\n")
            for entry in tqdm(entries):
                words = []
                bad_format = False
                feature_values = {}
                for i in self.features:
                    feature_values[i] = []

                ner_labels = []
                lines = entry.splitlines()
                for line in lines:
                    cols = line.rstrip().split('\t')
                    if len(cols) < 1 or cols[0].strip() == "":
                        bad_format = True
                        break
                    word = cols[0]
                    for idx in self.features:
                        if idx > len(cols):
                            raise RuntimeError("feature idx %d greater than number of columns in line %s " % (idx, line))
                        if cols[idx] in self.feature_value_idx_map[idx]:
                            feature_values[idx].append(self.feature_value_idx_map[idx][cols[idx]])
                        else:
                            feature_values[idx].append(self.feature_value_idx_map[idx]["UNKNOWN"])

                    words.append(word)
                    ner_labels.append(cols[-1])

                if bad_format:
                    continue

                out_lists.append([words, feature_values, ner_labels])
        return out_lists

    def get_examples(self, file_name):
        """Get examples from file."""
        return self._create_examples(
            self._read_data(file_name))

    def dropO(self, examples):
        """Replace O label by X for given examples."""
        for e in examples:
            e.labels = ['X' if l == 'O' else l for l in e.labels]

    def get_labels(self):
        return self._label_types

    def get_num_labels(self):
        return self.get_num_labels

    def get_label_map(self):
        return self._label_map

    def get_invsered_label_map(self):
        return self._invsered_label_map

    def get_start_label_id(self):
        return self._label_map['[CLS]']

    def get_stop_label_id(self):
        return self._label_map['[SEP]']

    def _create_examples(self, all_lists):
        print("==== Create Examples ====")
        examples = []
        for (i, one_lists) in tqdm(enumerate(all_lists)):
            guid = i
            words = one_lists[0]
            features = []
            for key in self.features:
                features.append(one_lists[1][key])
            labels = one_lists[-1]
            examples.append(InputNERExample(
                guid=guid, words=words, features=features, labels=labels))
        return examples
