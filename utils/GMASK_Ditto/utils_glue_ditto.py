from __future__ import absolute_import, division, print_function

import csv
import json
import logging
import os
import sys
from io import open
import jsonlines
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score

from dittolight.dataset import DittoDataset


logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None, evidence=None):
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
        self.evidence = evidence


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, ori_idx_token, evidence):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.ori_idx_token = ori_idx_token
        self.evidence = evidence


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8-sig") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines


class DataProcessor1(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        lines = []
        with open(input_file, "r", encoding="utf-8-sig") as f:
            for line in f:
                a = line.split('\t')
                lines.append(a)
        return lines


class eSnliProcessor(DataProcessor):
    """Processor for the MultiNLI data set (GLUE version)."""

    def get_train_examples(self, data_dir, docs):
        """See base class."""
        return self._create_examples(data_dir, docs, "train")

    def get_dev_examples(self, data_dir, docs):
        """See base class."""
        return self._create_examples(data_dir, docs, "val")

    def get_test_examples(self, data_dir, docs):
        """See base class."""
        return self._create_examples(data_dir, docs, "test")

    def get_labels(self):
        """See base class."""
        return [0, 1, 2]

    def _create_examples(self, path, docs, set_type):
        """Creates examples for the training and dev sets."""
        examples = []

        label_path = os.path.join(path, set_type+'.jsonl')
        label_toi = {"entailment": 0, "contradiction": 1, "neutral": 2}

        reader = jsonlines.Reader(open(label_path))
        i = -1
        for line in reader:
            i += 1
            label = label_toi[line["classification"]]
            evidence_temp = {}
            for evi in line["evidences"][0]:
                if evi["docid"] in evidence_temp.keys():
                    evidence_temp[evi["docid"]].append((evi["start_token"], evi["end_token"]))
                else:
                    evidence_temp[evi["docid"]] = [(evi["start_token"], evi["end_token"])]
            docid = line["annotation_id"]
            text_a = docs[docid+'_premise']
            text_b = docs[docid+'_hypothesis']
            guid = "%s-%s" % (set_type, i)
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label, evidence=evidence_temp))
        return examples


class QuoraProcessor(DataProcessor):
    """Processor for the MultiNLI data set (GLUE version)."""

    def get_train_examples(self, data_dir, docs):
        """See base class."""
        return self._create_examples(data_dir, docs, "train")

    def get_dev_examples(self, data_dir, docs):
        """See base class."""
        return self._create_examples(data_dir, docs, "dev")

    def get_test_examples(self, data_dir, docs):
        """See base class."""
        return self._create_examples(data_dir, docs, "test")

    def get_labels(self):
        """See base class."""
        return [0, 1]

    def _create_examples(self, path, docs, set_type):
        """Creates examples for the training and dev sets."""
        examples = []

        label_path = os.path.join(path, set_type+'.tsv')

        lines = self._read_tsv(label_path)
        i = -1
        for line in lines:
            i += 1
            label = int(line[0])
            evidence_temp = []
            text_a = line[1]
            text_b = line[2]
            guid = "%s-%s" % (set_type, i)
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label, evidence=evidence_temp))
        return examples


class QqpProcessor(DataProcessor1):
    """Processor for the MultiNLI data set (GLUE version)."""

    def get_train_examples(self, data_dir, docs):
        """See base class."""
        return self._create_examples(data_dir, docs, "train")

    def get_dev_examples(self, data_dir, docs):
        """See base class."""
        return self._create_examples(data_dir, docs, "dev")

    def get_test_examples(self, data_dir, docs):
        """See base class."""
        return self._create_examples(data_dir, docs, "test")

    def get_labels(self):
        """See base class."""
        return [0, 1]

    def _create_examples(self, path, docs, set_type):
        """Creates examples for the training and dev sets."""
        if set_type == 'test':
            examples = []
            label_path = os.path.join(path, set_type+'.tsv')
            lines = [] # self._read_tsv(label_path)
            skip_rows = 0
            with codecs.open(label_path, 'r', 'utf-8') as data_fh:
                for _ in range(skip_rows):
                    data_fh.readline()
                for row_idx, row in enumerate(data_fh):
                    try:
                        row = row.strip().split('\t')
                        lines.append(row)
                    except Exception as e:
                        print(e, " file: %s, row: %d" % (label_path, row_idx))
                    continue
            for idx, line in enumerate(lines):
                if idx == 0:
                    continue
                label = 0
                evidence_temp = []
                text_a = line[1]
                if text_a == '':
                    continue
                text_b = line[2]
                if text_b == '':
                    continue
                guid = "%s-%s" % (set_type, line[0])
                idnum = int(line[0])
                examples.append(
                    InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label, evidence=evidence_temp, idnum=idnum))
        else:
            examples = []
            label_path = os.path.join(path, set_type+'.tsv')
            lines = [] # self._read_tsv(label_path)
            skip_rows = 0
            with codecs.open(label_path, 'r', 'utf-8') as data_fh:
                for _ in range(skip_rows):
                    data_fh.readline()
                for row_idx, row in enumerate(data_fh):
                    try:
                        row = row.strip().split('\t')
                        label = int(row[-1])
                        lines.append(row)
                    except Exception as e:
                        print(e, " file: %s, row: %d" % (label_path, row_idx))
                    continue
            for idx, line in enumerate(lines):
                if idx == 0:
                    continue
                if len(line) != 6:
                    continue
                label = int(line[-1])
                evidence_temp = []
                text_a = line[3]
                if text_a == '':
                    continue
                text_b = line[4]
                if text_b == '':
                    continue
                guid = "%s-%s" % (set_type, line[0])
                examples.append(
                    InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label, evidence=evidence_temp))
        return examples


class MrpcProcessor(DataProcessor1):
    """Processor for the MultiNLI data set (GLUE version)."""

    def get_train_examples(self, data_dir, docs):
        """See base class."""
        return self._create_examples(data_dir, docs, "train")

    def get_dev_examples(self, data_dir, docs):
        """See base class."""
        return self._create_examples(data_dir, docs, "dev")

    def get_test_examples(self, data_dir, docs):
        """See base class."""
        return self._create_examples(data_dir, docs, "test")

    def get_labels(self):
        """See base class."""
        return [0, 1]

    def _create_examples(self, path, docs, set_type):
        """Creates examples for the training and dev sets."""
        if set_type == 'test':
            examples = []

            label_path = os.path.join(path, set_type+'.tsv')

            lines = self._read_tsv(label_path)
            for idx, line in enumerate(lines):
                if idx == 0:
                    continue
                label = 0
                text_a = line[-2]
                if text_a == '':
                    continue
                text_b = line[-1]
                if text_b == '':
                    continue
                guid = "%s-%s" % (set_type, idx)
                idnum = int(line[0])
                examples.append(
                    InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label, evidence=[]))
        else:
            examples = []

            label_path = os.path.join(path, set_type+'.tsv')

            lines = self._read_tsv(label_path)
            for idx, line in enumerate(lines):
                if idx == 0:
                    continue
                if len(line) != 5:
                    continue
                label = int(line[0])
                text_a = line[-2]
                if text_a == '':
                    continue
                text_b = line[-1]
                if text_b == '':
                    continue
                guid = "%s-%s" % (set_type, idx)
                examples.append(
                    InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label, evidence=[]))
        return examples


class generalProcessor(DataProcessor1):
    """Processor for the MultiNLI data set (GLUE version)."""

    def get_train_examples(self, data_dir, docs):
        """See base class."""
        return self._create_examples(data_dir, docs, "train")

    def get_dev_examples(self, data_dir, docs):
        """See base class."""
        return self._create_examples(data_dir, docs, "valid")

    def get_test_examples(self, data_dir, docs):
        """See base class."""
        return self._create_examples(data_dir, docs, "test")

    def get_labels(self):
        """See base class."""
        return [0, 1]

    def _create_examples(self, path, docs, set_type):
        """Creates examples for the training and dev sets."""
        examples = []

        val = DittoDataset(path+"/test.txt")

        test_iter = torch.utils.data.DataLoader(val,
                                    batch_size=len(val),
                                    shuffle=False,
                                    num_workers=0,
                                    collate_fn=val.pad)

        app = [x for x in test_iter][0]
        label_path = os.path.join(path, set_type+'.csv')
        data = pd.read_csv(label_path)
        data = data.reset_index()  # make sure indexes pair with number of rows
        listAttr = []
        flag = 0
        for idx, row in data.iterrows():
            qualityLabel = row['label']
            leftWord = ''
            leftID = str(row['left_id'])
            rightWord = ''
            rightID = str(row['right_id'])
            for key in list(row.keys()):
                if 'left' in key and key != 'left_id':
                    leftWord += str(row[key]) + ' ර '
                    if flag == 0:
                        listAttr.append(key)
            for key in list(row.keys()):
                if 'right' in key and key != 'right_id':
                    rightWord += str(row[key]) + ' ර '
                    if flag == 0:
                        listAttr.append(key)
            flag = 1
            guid = "%s-%s" % (set_type, idx)
            examples.append(InputExample(guid=guid, text_a=leftWord, text_b=rightWord, label=qualityLabel, evidence=[]))
        return app, listAttr






def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer, output_mode,
                                 cls_token_at_end=False,
                                 cls_token='[CLS]',
                                 cls_token_segment_id=1,
                                 sep_token='[SEP]',
                                 sep_token_extra=False,
                                 pad_on_left=False,
                                 pad_token=0,
                                 pad_token_segment_id=0,
                                 sequence_a_segment_id=0,
                                 sequence_b_segment_id=1,
                                 mask_padding_with_zero=True):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """

    label_map = {label : i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        ori_idx_token = {}
        evidence_words = []
        for evi in example.evidence:
            if evi.split('_')[1] == 'premise':
                for evi_idx in example.evidence[evi]:
                    txt = example.text_a.split()[evi_idx[0]:evi_idx[1]]
                    evidence_words += txt
            else:
                for evi_idx in example.evidence[evi]:
                    txt = example.text_b.split()[evi_idx[0]:evi_idx[1]]
                    evidence_words += txt

        evidence_string = ''
        for word in evidence_words:
            evidence_string += word
            evidence_string += ' '
        evidence_string = evidence_string[:-1]
        evidence_tokens = tokenizer.tokenize(evidence_string)

        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3". " -4" for RoBERTa.
            special_tokens_count = 4 if sep_token_extra else 3
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - special_tokens_count)
        else:
            # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
            special_tokens_count = 3 if sep_token_extra else 2
            if len(tokens_a) > max_seq_length - special_tokens_count:
                tokens_a = tokens_a[:(max_seq_length - special_tokens_count)]

        for tok in tokens_a:
            tok_id = tokenizer.encoder.get(tok)
            ori_idx_token[tok_id] = tok
        for tok in tokens_b:
            tok_id = tokenizer.encoder.get(tok)
            ori_idx_token[tok_id] = tok
        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = tokens_a + [sep_token]
        if sep_token_extra:
            # roberta uses an extra separator b/w pairs of sentences
            tokens += [sep_token]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        if tokens_b:
            tokens += tokens_b + [sep_token]
            segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)

        if cls_token_at_end:
            tokens = tokens + [cls_token]
            segment_ids = segment_ids + [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        evidence_ids = tokenizer.convert_tokens_to_ids(evidence_tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)
            evidence_ids = evidence_ids + ([pad_token_segment_id] * (max_seq_length - len(evidence_ids)))

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        if output_mode == "classification":
            label_id = label_map[example.label]
        elif output_mode == "regression":
            label_id = float(example.label)
        else:
            raise KeyError(output_mode)

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_id,
                          ori_idx_token=ori_idx_token,
                          evidence = evidence_ids))
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


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds)
    return {
        "acc": acc,
        "f1": f1,
        "acc_and_f1": (acc + f1) / 2,
    }


processors = {
    "esnli": eSnliProcessor,
    "quora": QuoraProcessor,
    "qqp": QqpProcessor,
    "mrpc": MrpcProcessor,
    "general": generalProcessor
}

output_modes = {
    "esnli": "classification",
    "quora": "classification",
    "qqp": "classification",
    "mrpc": "classification",
    "general": "classification"
}

GLUE_TASKS_NUM_LABELS = {
    "esnli": 3,
    "mrpc": 2,
    "quora": 2,
    "qqp": 2,
    "general": 2
}
