"""
Code taken from
    https://github.com/wbsg-uni-mannheim/contrastive-product-matching/blob/main/src/contrastive/data/data_collators.py
"""
import numpy as np

np.random.seed(42)
import random

random.seed(42)
from dataclasses import dataclass
from typing import Optional, Any
import torch
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from utils.bert_utils import bi_encoder_random_masking, bi_encoder_syntax_masking, bi_encoder_semantic_masking, \
    get_words_from_tokens
from utils.nlp import FastTextModel


@dataclass
class DataCollatorContrastiveClassification:
    tokenizer: PreTrainedTokenizerBase
    max_length: Optional[int] = 128
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"
    typeMask: Optional[str] = None
    topk_mask: Optional[int] = None
    sem_emb_model: Optional[Any] = None

    stopwords = ['COL', 'VAL', '[COL]', '[VAL]', 'PERSON', 'ORG', 'LOC', 'PRODUCT', 'DATE', 'QUANTITY', 'TIME',
                 'Artist_Name', 'name', 'Released', 'CopyRight', 'content', 'Brew_Factory_Name', 'Time', 'type',
                 'Beer_Name', 'category', 'price', 'title', 'authors', 'class', 'description',
                 'Song_Name', 'venue', 'brand', 'Genre', 'year', 'manufacturer', 'Style', 'addr', 'phone',
                 'modelno', 'Price', 'ABV', 'city', 'Album_Name', 'specTableContent']

    def __call__(self, input):
        features_left = [x['features_left'] for x in input]
        features_right = [x['features_right'] for x in input]
        labels = [x['labels'] for x in input]

        batch_left = self.tokenizer(features_left, padding=True, truncation=True, max_length=self.max_length,
                                    return_tensors=self.return_tensors)
        batch_right = self.tokenizer(features_right, padding=True, truncation=True, max_length=self.max_length,
                                     return_tensors=self.return_tensors)

        if self.typeMask == 'random':

            features = [bi_encoder_random_masking(
                left_features=batch_left, right_features=batch_right, tokenizer=self.tokenizer,
                topk=self.topk_mask, ignore_tokens=self.stopwords, index=i
            ) for i in range(len(batch_left['input_ids']))]
            batch_left['input_ids'] = torch.cat([x[0].unsqueeze(0) for x in features])
            batch_right['input_ids'] = torch.cat([x[1].unsqueeze(0) for x in features])

        elif self.typeMask == 'maskSyn':
            sent1_list = []
            sent2_list = []
            for i in range(len(batch_left['input_ids'])):
                left_tokens = self.tokenizer.convert_ids_to_tokens(batch_left['input_ids'][i])
                left_words = get_words_from_tokens(batch_left.word_ids(i), left_tokens)

                right_tokens = self.tokenizer.convert_ids_to_tokens(batch_right['input_ids'][i])
                right_words = get_words_from_tokens(batch_right.word_ids(i), right_tokens)

                sent1_list.append(left_words)
                sent2_list.append(right_words)

            features = [bi_encoder_syntax_masking(
                sent1=sent1_list[i], sent2=sent2_list[i], left_features=batch_left,
                right_features=batch_right, topk=self.topk_mask, ignore_tokens=self.stopwords, index=i
            ) for i in range(len(batch_left['input_ids']))]
            batch_left['input_ids'] = torch.cat([x[0].unsqueeze(0) for x in features])
            batch_right['input_ids'] = torch.cat([x[1].unsqueeze(0) for x in features])

        elif self.typeMask == 'maskSem':
            assert self.sem_emb_model is not None
            sent1_list = []
            sent2_list = []
            for i in range(len(batch_left['input_ids'])):
                left_tokens = self.tokenizer.convert_ids_to_tokens(batch_left['input_ids'][i])
                left_words = get_words_from_tokens(batch_left.word_ids(i), left_tokens)

                right_tokens = self.tokenizer.convert_ids_to_tokens(batch_right['input_ids'][i])
                right_words = get_words_from_tokens(batch_right.word_ids(i), right_tokens)

                sent1_list.append(left_words)
                sent2_list.append(right_words)

            features = [bi_encoder_semantic_masking(
                sent1=sent1_list[i], sent2=sent2_list[i], left_features=batch_left,
                right_features=batch_right, topk=self.topk_mask, sem_emb_model=self.sem_emb_model,
                ignore_tokens=self.stopwords, index=i
            ) for i in range(len(batch_left['input_ids']))]
            batch_left['input_ids'] = torch.cat([x[0].unsqueeze(0) for x in features])
            batch_right['input_ids'] = torch.cat([x[1].unsqueeze(0) for x in features])

        elif self.typeMask is None or self.typeMask == 'off':
            pass

        else:
            raise ValueError("Wrong masking type!")

        batch = batch_left
        if 'token_type_ids' in batch.keys():
            del batch['token_type_ids']
        batch['input_ids_right'] = batch_right['input_ids']
        batch['attention_mask_right'] = batch_right['attention_mask']
        batch['word_ids_left'] = np.array([batch_left.word_ids(i) for i in range(len(batch_left['input_ids']))])
        batch['word_ids_right'] = np.array([batch_right.word_ids(i) for i in range(len(batch_left['input_ids']))])
        batch['sent1'] = np.array(features_left)
        batch['sent2'] = np.array(features_right)

        batch['labels'] = torch.LongTensor(labels)

        return batch
