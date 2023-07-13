"""
Code taken from https://github.com/megagonlabs/ditto/blob/master/ditto_light/dataset.py
"""
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from utils.bert_utils import random_masking, syntax_masking, semantic_masking

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

    def __init__(self,
                 path,
                 max_len=256,
                 size=None,
                 lm='roberta',
                 da=None,
                 typeMask: str = None,
                 topk_mask: int = None):
        self.tokenizer = get_tokenizer(lm)
        self.pairs = []
        self.labels = []
        self.max_len = max_len
        self.size = size

        if isinstance(path, list):
            lines = path
        else:
            lines = open(path)

        for line in lines:
            s1, s2, label = line.strip().split('\t')
            self.pairs.append((s1, s2))
            self.labels.append(int(label))

        self.pairs = self.pairs[:size]
        self.labels = self.labels[:size]
        self.typeMask = typeMask
        self.topk_mask = topk_mask

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
            padding='max_length',
            truncation=True,
            return_tensors="pt",
            max_length=self.max_len,
            add_special_tokens=True,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_token_type_ids=True
        )

        if self.typeMask == 'random':
            # features = mask_random(features)
            features = random_masking(features, self.topk_mask)
        elif self.typeMask == 'maskSyn':
            sent1 = left.split()
            sent2 = right.split()
            features = syntax_masking(sent1, sent2, features, self.topk_mask)
        elif self.typeMask == 'maskSem':
            sent1 = left.split()
            sent2 = right.split()
            features = semantic_masking(sent1, sent2, features, self.topk_mask)
        elif self.typeMask is None:
            pass
        else:
            raise ValueError("Wrong masking type!")

        flat_features = {}
        for feature in features.data:
            flat_features[feature] = features.data[feature].squeeze(0)
        flat_features['sent1'] = left
        flat_features['sent2'] = right
        flat_features['word_ids'] = features.word_ids()
        flat_features['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)

        return flat_features

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
            x12, y = zip(*batch)
            maxlen = max([len(x) for x in x12])
            x12 = [xi + [0] * (maxlen - len(xi)) for xi in x12]
            return torch.LongTensor(x12), \
                torch.LongTensor(y)