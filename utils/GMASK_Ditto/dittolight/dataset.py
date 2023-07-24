import torch

from torch.utils import data
from transformers import AutoTokenizer

from .augment import Augmenter

# map lm name to huggingface's pre-trained model names
lm_mp = {'roberta': 'roberta-base',
         'distilbert': 'distilbert-base-uncased'}

def get_tokenizer(lm):
    if lm in lm_mp:
        return AutoTokenizer.from_pretrained(lm_mp[lm])
    else:
        return AutoTokenizer.from_pretrained(lm)


class DittoDataset(data.Dataset):
    """EM dataset"""

    def __init__(self,
                 path,
                 max_len=256,
                 size=None,
                 lm='roberta',
                 da=None):
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
        self.da = da
        if da is not None:
            self.augmenter = Augmenter()
        else:
            self.augmenter = None


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
        # x = self.tokenizer.encode(text=left,
        #                           text_pair=right,
        #                           max_length=self.max_len,
        #                           truncation=True)
        x = self.tokenizer(text=left,
                           text_pair=right,
                           padding=True, truncation=True, max_length=self.max_len,
                           return_tensors='pt'
        )

        # augment if da is set
        if self.da is not None:
            combined = self.augmenter.augment_sent(left + ' [SEP] ' + right, self.da)
            left, right = combined.split(' [SEP] ')
            x_aug = self.tokenizer.encode(text=left,
                                      text_pair=right,
                                      max_length=self.max_len,
                                      truncation=True)
            return x, x_aug, self.labels[idx]
        else:
            return x, self.labels[idx]


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

            maxlen = max([len(x) for x in x1+x2])
            x1 = [xi + [0]*(maxlen - len(xi)) for xi in x1]
            x2 = [xi + [0]*(maxlen - len(xi)) for xi in x2]
            return torch.LongTensor(x1), \
                   torch.LongTensor(x2), \
                   torch.LongTensor(y)
        else:
            x12, y = zip(*batch)
            x12 = [{k: v.squeeze(0) for k, v in x.items()} for x in x12]
            maxlen = max([len(x['input_ids']) for x in x12])
            input_ids = [torch.cat((x['input_ids'], torch.zeros(maxlen - len(x['input_ids'])))).unsqueeze(0) for x in
                         x12]
            attention_mask = [
                torch.cat((x['attention_mask'], torch.ones(maxlen - len(x['attention_mask'])))).unsqueeze(0) for x in
                x12]
            token_type_ids = torch.zeros(len(x12), maxlen)
            sep_ixs = [torch.where(iids[0]==2)[0][-2:] for iids in input_ids]
            for i, ttids in enumerate(token_type_ids):
                ixs = sep_ixs[i] + 1
                ttids[ixs[0]:ixs[1]] = 1
            # token_type_ids = [
            #     torch.cat((x['token_type_ids'], torch.zeros(maxlen - len(x['token_type_ids'])))).unsqueeze(0) for x in
            #     batch]

            out = {
                'input_ids': torch.cat(input_ids).to(dtype=torch.long),
                'attention_mask': torch.cat(attention_mask).to(dtype=torch.long),
                'token_type_ids': token_type_ids.to(dtype=torch.long),
                'labels': torch.LongTensor(list(y)),
                'evidence': torch.zeros(len(batch)).to(dtype=torch.long)
            }
            return out

