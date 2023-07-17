"""
Code taken from https://github.com/megagonlabs/ditto/blob/master/ditto_light/dataset.py
"""
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from utils.bert_utils import cross_encoder_random_masking, cross_encoder_syntax_masking, \
    cross_encoder_semantic_masking, get_left_right_words_and_ids
from utils.nlp import FastTextModel
import numpy as np

# map lm name to huggingface's pre-trained model names
lm_mp = {'roberta': 'roberta-base',
         'distilbert': 'distilbert-base-uncased'}


def get_tokenizer(lm):
    if lm in lm_mp:
        return AutoTokenizer.from_pretrained(lm_mp[lm])
    else:
        return AutoTokenizer.from_pretrained(lm)


class DittoDataset(Dataset):
    """EM dataset"""

    stopwords = ['COL', 'VAL', '[COL]', '[VAL]', 'PERSON', 'ORG', 'LOC', 'PRODUCT', 'DATE', 'QUANTITY', 'TIME',
                 'Artist_Name', 'name', 'Released', 'CopyRight', 'content', 'Brew_Factory_Name',
                 'Time', 'type', 'Beer_Name', 'category', 'price', 'title', 'authors', 'class', 'description',
                 'Song_Name', 'venue', 'brand', 'Genre', 'year', 'manufacturer', 'Style', 'addr', 'phone',
                 'modelno', 'Price', 'ABV', 'city', 'Album_Name', 'specTableContent']

    def __init__(self,
                 path,
                 max_len=256,
                 size=None,
                 lm='roberta',
                 da=None,
                 typeMask: str = None,
                 topk_mask: int = None,
                 verbose: bool = False,
                 sem_emb_model=None):
        self.tokenizer = get_tokenizer(lm)
        self.pairs = []
        self.labels = []
        self.max_len = max_len
        self.size = size

        if isinstance(path, list):
            lines = path
        else:
            lines = open(path, encoding="utf8")

        for line in lines:
            s1, s2, label = line.strip().split('\t')
            self.pairs.append((s1, s2))
            self.labels.append(int(label))

        self.pairs = self.pairs[:size]
        self.labels = self.labels[:size]
        self.typeMask = typeMask
        self.topk_mask = topk_mask
        self.verbose = verbose
        self.sem_emb_model = sem_emb_model

    def __len__(self):
        """Return the size of the dataset."""
        return len(self.pairs)

    def __getitem__(self, idx):
        """Return a tokenized item of the dataset.

        Args:
            idx (int): the index of the item

        Returns:
            List of int: token ID's of the two entities
            List of int: token ID's of the two entities augmented (if da is set)
            int: the label of the pair (0: unmatch, 1: match)
        """
        left = self.pairs[idx][0]
        right = self.pairs[idx][1]

        # left + right
        features = self.tokenizer(
            text=left,
            text_pair=right,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=self.max_len,
            add_special_tokens=True,
            pad_to_max_length=False,
            return_attention_mask=True,
            return_token_type_ids=True
        )

        if self.typeMask == 'random':
            # features = mask_random(features)
            features = cross_encoder_random_masking(
                features=features, topk=self.topk_mask, tokenizer=self.tokenizer, ignore_tokens=self.stopwords
            )

        elif self.typeMask == 'maskSyn':
            _, _, sent1, sent2 = get_left_right_words_and_ids(self.tokenizer, features)
            features = cross_encoder_syntax_masking(
                sent1=sent1, sent2=sent2, features=features, topk=self.topk_mask, ignore_tokens=self.stopwords
            )

        elif self.typeMask == 'maskSem':
            assert self.sem_emb_model is not None
            _, _, sent1, sent2 = get_left_right_words_and_ids(self.tokenizer, features)
            features = cross_encoder_semantic_masking(
                sent1=sent1, sent2=sent2, features=features, topk=self.topk_mask, sem_emb_model=self.sem_emb_model,
                ignore_tokens=self.stopwords,
            )

        elif self.typeMask is None:
            pass

        else:
            raise ValueError("Wrong masking type!")

        out_features = {}
        for feature in features.data:
            out_features[feature] = features.data[feature].squeeze(0)
        out_features['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)

        if self.verbose:
            out_features['sent1'] = left
            out_features['sent2'] = right
            out_features['word_ids'] = features.word_ids()

        return out_features

    @staticmethod
    def pad(batch):
        """Merge a list of dataset items into a train/test batch
        Args:
            batch (list of tuple): a list of dataset items

        Returns:
            LongTensor: x1 of shape (batch_size, seq_len)
            LongTensor: x2 of shape (batch_size, seq_len).
                        Elements of x1 and x2 are padded to the same length
            LongTensor: a batch of labels, (batch_size,)
        """
        if len(batch[0]) == 3:
            x1, x2, y = zip(*batch)

            maxlen = max([len(x) for x in x1 + x2])
            x1 = [xi + [0] * (maxlen - len(xi)) for xi in x1]
            x2 = [xi + [0] * (maxlen - len(xi)) for xi in x2]
            return torch.LongTensor(x1), \
                torch.LongTensor(x2), \
                torch.LongTensor(y)
        else:
            maxlen = max([len(x['input_ids']) for x in batch])
            input_ids = [torch.cat((x['input_ids'], torch.zeros(maxlen - len(x['input_ids'])))).unsqueeze(0) for x in
                         batch]
            attention_mask = [
                torch.cat((x['attention_mask'], torch.ones(maxlen - len(x['attention_mask'])))).unsqueeze(0) for x in
                batch]
            token_type_ids = [
                torch.cat((x['token_type_ids'], torch.zeros(maxlen - len(x['token_type_ids'])))).unsqueeze(0) for x in
                batch]

            out = {
                'input_ids': torch.cat(input_ids).to(dtype=torch.long),
                'attention_mask': torch.cat(attention_mask).to(dtype=torch.long),
                'token_type_ids': torch.cat(token_type_ids).to(dtype=torch.long),
                'labels': torch.cat([x['labels'].unsqueeze(0) for x in batch]).to(dtype=torch.long)
            }

            if 'sent1' in batch[0]:
                out['sent1'] = np.array([x['sent1'] for x in batch])
            if 'sent2' in batch[0]:
                out['sent2'] = np.array([x['sent2'] for x in batch])
            if 'word_ids' in batch[0]:
                out['word_ids'] = np.array([x['word_ids'] for x in batch])

            return out
