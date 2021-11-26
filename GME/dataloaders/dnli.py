from torch.utils import data
import torch
import os
from collections import defaultdict
import numpy as np
import random
import json

np.random.seed(0)

label_convert = {
            'positive': 'entailment',
            'negative': 'contradiction',
            'neutral': 'neutral',
        }


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()


class PersonanliProcessor(DataProcessor):
    """Processor for the Personanli data set (GLUE version)."""

    def get_labels(self):
        """See base class."""
        return ["contradiction", "entailment", "neutral"]

    def create_batch(self, turn, persona, labels, set_type="predict"):
        """Creates examples for the training and dev sets."""
        assert len(turn) == len(persona) == len(labels)
        examples = []
        for (i, line) in enumerate(zip(turn, persona)):
            guid = "%s-%s" % (set_type, i)
            text_a = line[0]
            text_b = line[1]
            label = labels[i]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class DNLI(data.Dataset):
    """The DNLI dataset."""

    def __init__(self, mode, tokenizer, max_len):
        self.mode = mode
        self.tokenizer = tokenizer
        assert self.mode in ['test', 'verified_test']
        self.root = os.path.join('../data', 'dnli')
        self.cached_features_file = os.path.join(self.root, 'cached_{}'.format(mode))
        self.max_len = max_len
        self.processor = PersonanliProcessor()
        self.label_list = self.processor.get_labels()

        if os.path.exists(self.cached_features_file):
            print("Loading features from cached file {}".format(self.cached_features_file))
            self.features = torch.load(self.cached_features_file)
        else:
            turn,  personas_items, labels = [], [], []
            with open(os.path.join(self.root, 'dialogue_nli_{}.jsonl'.format(self.mode))) as f:
                s = f.readlines()
                assert len(s) == 1, '{}'.format(len(s))
                train_data = json.loads(s[0].strip())
                print(len(train_data))
                for _data in train_data:
                    turn.append(_data['sentence1'])
                    personas_items.append(_data['sentence2'])
                    labels.append(label_convert[_data['label']])
            examples = self.processor.create_batch(turn, personas_items, labels)
            self.features = convert_examples_to_features(examples, self.label_list, self.max_len, self.tokenizer)

            print("Saving features into cached file {}".format(self.cached_features_file))
            torch.save(self.features, self.cached_features_file)

        self.voc_size = len(self.tokenizer)

        print('DNLI data successfully read.')

    def __getitem__(self, index):
        f = self.features[index]
        input_ids = torch.tensor(f.input_ids, dtype=torch.long)
        input_mask = torch.tensor(f.input_mask, dtype=torch.long)
        segment_ids = torch.tensor(f.segment_ids, dtype=torch.long)
        label_ids = torch.tensor([f.label_id], dtype=torch.long).squeeze(0)

        return input_ids, input_mask, segment_ids, label_ids

    def __len__(self):
        return len(self.features)


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambigiously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        label_id = label_map[example.label]

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_id))
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()
