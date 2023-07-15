"""
Code taken from
    https://github.com/wbsg-uni-mannheim/contrastive-product-matching/blob/main/src/contrastive/data/datasets.py
"""

import numpy as np

np.random.seed(42)
import random

random.seed(42)

import pandas as pd
import torch
from transformers import AutoTokenizer
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.char as nac


def assign_clusterid(identifier, cluster_id_dict, cluster_id_amount):
    try:
        result = cluster_id_dict[identifier]
    except KeyError:
        result = cluster_id_amount
    return result


def clean_text(text: str) -> str:
    """ Clean the input text. """
    text = text.strip()
    text = " ".join([x.lower() for x in text.split()])
    text = text.replace('"', '')

    return text


def add_seps(x: pd.Series):
    sent = ''
    for attr, val in x.items():
        if pd.isnull(val):
            sent += f'[COL] {attr} [VAL]  '  # Replace None with ''
        else:
            sent += f'[COL] {attr} [VAL] {clean_text(str(val))} '
    sent = sent[:-1]

    return sent


def serialize(data, side=None):
    if side is None:
        feat_cols = [c for c in data.index if c not in ['id', 'labels', 'cluster_id']]
    else:
        feat_cols = [c for c in data.index if
                     c not in [f'id_{side}', 'labels', f'cluster_id_{side}'] and c.endswith(side)]
    entity = data[feat_cols]
    if side is not None:
        entity = entity.rename({c: c.replace(f'_{side}', '') for c in entity.index})
    return add_seps(entity)


class Augmenter:
    def __init__(self, aug):

        stopwords = ['[COL]', '[VAL]', 'Artist_Name', 'name', 'Released', 'CopyRight', 'content', 'Brew_Factory_Name',
                     'Time', 'type', 'Beer_Name', 'category', 'price', 'title', 'authors', 'class', 'description',
                     'Song_Name', 'venue', 'brand', 'Genre', 'year', 'manufacturer', 'Style', 'addr', 'phone',
                     'modelno', 'Price', 'ABV', 'city', 'Album_Name', 'specTableContent']

        aug_typo = nac.KeyboardAug(stopwords=stopwords, aug_char_p=0.1, aug_word_p=0.1)
        aug_swap = naw.RandomWordAug(action="swap", stopwords=stopwords, aug_p=0.1)
        aug_del = naw.RandomWordAug(action="delete", stopwords=stopwords, aug_p=0.1)
        aug_crop = naw.RandomWordAug(action="crop", stopwords=stopwords, aug_p=0.1)
        aug_sub = naw.RandomWordAug(action="substitute", stopwords=stopwords, aug_p=0.1)
        aug_split = naw.SplitAug(stopwords=stopwords, aug_p=0.1)

        aug = aug.strip('-')

        if aug == 'all':
            self.augs = [aug_typo, aug_swap, aug_split, aug_sub, aug_del, aug_crop, None]

        if aug == 'typo':
            self.augs = [aug_typo, None]

        if aug == 'swap':
            self.augs = [aug_swap, None]

        if aug == 'delete':
            self.augs = [aug_del, None]

        if aug == 'crop':
            self.augs = [aug_crop, None]

        if aug == 'substitute':
            self.augs = [aug_sub, None]

        if aug == 'split':
            self.augs = [aug_split, None]

    def apply_aug(self, string):
        aug = random.choice(self.augs)
        if aug is None:
            return string
        else:
            return aug.augment(string)


class ContrastiveClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, path, dataset_type, size=None, tokenizer='huawei-noah/TinyBERT_General_4L_312D', max_length=128,
                 dataset='lspc', aug=False):

        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, additional_special_tokens=('[COL]', '[VAL]'))
        self.dataset_type = dataset_type
        self.dataset = dataset
        self.aug = aug

        if self.aug:
            self.augmenter = Augmenter(self.aug)

        if dataset == 'lspc':
            data = pd.read_pickle(path)
        else:
            data = pd.read_json(path, lines=True)

        data = data.fillna('')

        if self.dataset_type != 'test':
            validation_ids = pd.read_csv(f'../../data/interim/{dataset}/{dataset}-valid.csv')

            if self.dataset_type == 'train':
                data = data[~data['pair_id'].isin(validation_ids['pair_id'])]
            else:
                data = data[data['pair_id'].isin(validation_ids['pair_id'])]

        data = data.reset_index(drop=True)

        data = self._prepare_data(data)

        self.data = data

    def __getitem__(self, idx):
        example = self.data.loc[idx].copy()

        if self.aug:
            example['features_left'] = self.augmenter.apply_aug(example['features_left'])
            example['features_right'] = self.augmenter.apply_aug(example['features_right'])

        return example

    def __len__(self):
        return len(self.data)

    def _prepare_data(self, data):

        data['features_left'] = data.apply(serialize, side='left', axis=1)
        data['features_right'] = data.apply(serialize, side='right', axis=1)

        data = data[['features_left', 'features_right', 'label']]
        data = data.rename(columns={'label': 'labels'})

        return data
