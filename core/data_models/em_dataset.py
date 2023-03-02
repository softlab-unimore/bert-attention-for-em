import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import pandas as pd
from utils.bert_utils import tokenize_entity_pair


class EMDataset(Dataset):

    def __init__(self, data: pd.DataFrame, model_name: str,
                 tokenization: str = 'sent_pair',
                 label_col: str = 'label', left_prefix: str = 'left_',
                 right_prefix: str = 'right_', max_len: int = 256,
                 verbose: bool = False, categories: list = None,
                 permute: bool = False, seed: int = 42, typeMask: str = 'off',columnMask: str='', return_offset: bool = False):

        assert isinstance(tokenization, str)
        assert tokenization in ['sent_pair', 'attr', 'attr_pair']

        self.data = data
        self.model_name = model_name
        self.tokenization = tokenization
        self.label_col = label_col
        self.left_prefix = left_prefix
        self.right_prefix = right_prefix
        self.max_len = max_len
        assert (self.max_len % 2) == 0
        self.verbose = verbose
        self.categories = categories
        self.permute = permute
        self.seed = seed
        self.return_offset = return_offset
        self.typeMask = typeMask
        self.columnMask = columnMask

        if label_col not in self.data.columns:
            raise ValueError("Label column not found.")

        # remove labels from feature table
        self.labels = self.data[self.label_col]
        self.X = self.data.drop([self.label_col], axis=1)

        # remove entity identifiers
        ids = ['{}id'.format(self.left_prefix), '{}id'.format(self.right_prefix)]
        for single_id in ids:
            if single_id in self.X.columns:
                self.X = self.X.drop([single_id], axis=1)

        # extract left and right features
        self.left_cols = []
        self.right_cols = []
        remove_cols = []
        for col in self.X.columns:
            if col.startswith(left_prefix):
                self.left_cols.append(col)
            elif col.startswith(right_prefix):
                self.right_cols.append(col)
            else:
                remove_cols.append(col)

        if len(remove_cols) > 0:
            print("Warning: the following columns will be removed from the data: {}".format(remove_cols))
            self.X = self.X.drop(remove_cols, axis=1)

        # check that the dataset contains the same number of left and right features
        assert len(self.left_cols) == len(self.right_cols)

        # check that the left and right feature names are equal
        c1 = [c.replace(self.left_prefix, "") for c in self.left_cols]
        c2 = [c.replace(self.right_prefix, "") for c in self.right_cols]
        assert c1 == c2

        self.complete_data = self.X.copy()
        self.complete_data[self.label_col] = self.labels
        self.columns = c1
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)

    def get_complete_data(self):
        return self.complete_data

    def get_columns(self):
        return self.columns

    def get_params(self):
        params = {'model_name': self.model_name, 'label_col': self.label_col, 'left_prefix': self.left_prefix,
                  'right_prefix': self.right_prefix, 'max_len': self.max_len, 'tokenization': self.tokenization,
                  'return_offset': self.return_offset, 'typeMask':self.typeMask, 'columnMask': self.columnMask}

        return params

    @staticmethod
    def check_features(features: tuple):
        assert isinstance(features, tuple), "Wrong data type for parameter 'features'."
        err_msg = "Wrong features format."
        assert len(features) == 3, err_msg
        l = features[0]
        r = features[1]
        f = features[2]
        assert isinstance(l, pd.Series), err_msg
        assert isinstance(r, pd.Series), err_msg
        assert isinstance(f, dict), err_msg
        params = ['input_ids', 'token_type_ids', 'attention_mask', 'sent1', 'sent2', 'labels']
        assert all([p in f for p in params]), err_msg
        assert isinstance(f['input_ids'], torch.Tensor), err_msg
        assert isinstance(f['token_type_ids'], torch.Tensor), err_msg
        assert isinstance(f['attention_mask'], torch.Tensor), err_msg
        assert isinstance(f['sent1'], str), err_msg
        assert isinstance(f['sent2'], str), err_msg
        assert isinstance(f['labels'], torch.Tensor), err_msg

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.X.iloc[idx]
        label = self.labels.iloc[idx]
        if self.categories is not None:
            category = self.categories[idx]

        left_row = row[self.left_cols]
        left_row.index = self.columns

        right_row = row[self.right_cols]
        right_row.index = self.columns

        unk_token = self.tokenizer.unk_token
        left_row = left_row.fillna(unk_token)
        right_row = right_row.fillna(unk_token)

        if self.permute:
            np.random.seed(self.seed + idx)
            # perm = np.random.permutation(len(self.columns))
            perm = list(reversed(range(len(self.columns))))
            perm_cols = [self.columns[ix] for ix in perm]
            left_row.index = perm_cols
            left_row = left_row.reindex(index=self.columns)
            for attr, val in left_row.copy().iteritems():
                permuted_val = ' '.join(np.random.permutation(str(val).split()))
                left_row[attr] = permuted_val

        tokenized_row = tokenize_entity_pair(left_row, right_row, self.tokenizer, self.tokenization, self.max_len,
                                             return_offset=self.return_offset, typeMask=self.typeMask,
                                             columnMask=self.columnMask)
        tokenized_row['labels'] = torch.tensor(label, dtype=torch.long)
        if self.categories is not None:
            tokenized_row['category'] = category

        if not self.verbose:
            return tokenized_row

        return left_row, right_row, tokenized_row
