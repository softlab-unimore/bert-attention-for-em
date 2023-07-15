"""
Code taken from
    https://github.com/wbsg-uni-mannheim/contrastive-product-matching/blob/main/src/contrastive/data/data_collators.py
"""
import numpy as np
np.random.seed(42)
import random
random.seed(42)
from dataclasses import dataclass
from typing import Optional
import torch
from transformers.tokenization_utils_base import PreTrainedTokenizerBase


@dataclass
class DataCollatorContrastiveClassification:
    tokenizer: PreTrainedTokenizerBase
    max_length: Optional[int] = 128
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, input):
        features_left = [x['features_left'] for x in input]
        features_right = [x['features_right'] for x in input]
        labels = [x['labels'] for x in input]

        batch_left = self.tokenizer(features_left, padding=True, truncation=True, max_length=self.max_length,
                                    return_tensors=self.return_tensors)
        batch_right = self.tokenizer(features_right, padding=True, truncation=True, max_length=self.max_length,
                                     return_tensors=self.return_tensors)

        batch = batch_left
        if 'token_type_ids' in batch.keys():
            del batch['token_type_ids']
        batch['input_ids_right'] = batch_right['input_ids']
        batch['attention_mask_right'] = batch_right['attention_mask']

        batch['labels'] = torch.LongTensor(labels)

        return batch
