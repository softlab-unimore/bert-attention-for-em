import time

import pandas as pd
from transformers import AutoModelForSequenceClassification
from torch.utils.data import DataLoader
import argparse
import distutils.util
from tqdm import tqdm
import numpy as np
import torch
from sklearn.metrics import f1_score
import os
import pickle
from pathlib import Path

from core.data_models.em_dataset import EMDataset
from utils.general import get_dataset, get_sample
from utils.data_collector import DM_USE_CASES
from utils.data_selection import Sampler
from functools import reduce
from collections import Counter
from itertools import product
from utils.bert_utils import get_left_and_right_word_ids


PROJECT_DIR = Path(__file__).parent.parent.parent
RESULTS_DIR = os.path.join(PROJECT_DIR, 'results', 'models')


def get_most_freq_words(df: pd.DataFrame, topk: int):

    word_counter = Counter()
    for ix, row in df.iterrows():
        row_wo_label = row.copy()
        del row_wo_label['label']
        sent = reduce(lambda a, b: f'{a} {b}', row_wo_label.values)
        word_counter.update(sent.split())

    if topk > 0:
        return word_counter.most_common(topk)
    return word_counter.most_common()[topk:]


def get_most_freq_word_pairs(df: pd.DataFrame, topk: int):
    pair_counter = Counter()
    for ix, row in df.iterrows():
        row_wo_label = row.copy()
        del row_wo_label['label']

        left_cols = [x for x in row.index if x.startswith('left_')]
        left_row = row_wo_label[left_cols]
        left_sent = reduce(lambda a, b: f'{a} {b}', left_row.values)

        right_cols = [x for x in row.index if x.startswith('right_')]
        right_row = row_wo_label[right_cols]
        right_sent = reduce(lambda a, b: f'{a} {b}', right_row.values)

        pairs = product(left_sent.split(), right_sent.split())

        pair_counter.update(pairs)

    if topk > 0:
        return pair_counter.most_common(topk)
    return pair_counter.most_common()[topk:]


def get_most_freq_matching_words(dataset: EMDataset, data_type, term_type, topk):

    sampler = Sampler(dataset)
    dataset_params = sampler.dataset_params.copy()
    dataset_params["verbose"] = False

    if data_type == 'match':
        data = sampler._get_data_by_label(1)
    elif data_type == 'non_match':
        data = sampler._get_data_by_label(0)
    else:
        raise ValueError()

    if term_type == 'singleton':
        out = get_most_freq_words(data, topk)
    elif term_type == 'pair':
        out = get_most_freq_word_pairs(data, topk)
    else:
        raise ValueError()

    return out


def get_random_words(df: pd.DataFrame, size: int):

    word_counter = Counter()
    for ix, row in df.iterrows():
        row_wo_label = row.copy()
        del row_wo_label['label']
        sent = reduce(lambda a, b: f'{a} {b}', row_wo_label.values)
        word_counter.update(sent.split())

    permutation = np.random.permutation(len(word_counter))
    selection = permutation[:size]
    word_counts = word_counter.most_common()
    return [word_counts[i] for i in selection]


def get_random_matching_words(dataset: EMDataset, size):

    data = dataset.get_complete_data()

    return get_random_words(data, size)


def inject_word_into_row(row, word, repeat=5):
    left_cols = [x for x in row.index if x.startswith('left_')]
    right_cols = [x for x in row.index if x.startswith('right_')]

    new_row = row.copy()
    new_row[left_cols[0]] = f"{' '.join([word] * repeat)} {row[left_cols[0]]}"
    new_row[right_cols[0]] = f"{' '.join([word] * repeat)} {row[right_cols[0]]}"

    return new_row


def inject_pair_word_into_row(row, pair, repeat=5):
    left_cols = [x for x in row.index if x.startswith('left_')]
    right_cols = [x for x in row.index if x.startswith('right_')]

    new_row = row.copy()
    new_row[left_cols[0]] = f"{' '.join([pair[0]] * repeat)} {row[left_cols[0]]}"
    new_row[right_cols[0]] = f"{' '.join([pair[1]] * repeat)} {row[right_cols[0]]}"

    return new_row


def inject_into_row(row, word, repeat):
    if isinstance(word, str):
        new_row = inject_word_into_row(row, word, repeat=repeat)
    elif isinstance(word, tuple):
        new_row = inject_pair_word_into_row(row, word, repeat=repeat)
    else:
        raise ValueError()

    return new_row


def inject_words_into_dataset(dataset, data_type, words, repeat=5):
    sampler = Sampler(dataset)
    dataset_params = sampler.dataset_params.copy()
    dataset_params["verbose"] = False

    if data_type == 'match':
        data = sampler._get_data_by_label(1)
    elif data_type == 'non_match':
        data = sampler._get_data_by_label(0)
    else:
        raise ValueError()

    new_datasets = {}
    for word in words:
        word = word[0]

        if word == 'nan':
            continue

        new_dataset = data.apply(lambda x: inject_into_row(x, word, repeat), axis=1)
        # new_dataset = []
        # for ix, row in data.iterrows():
        #     if isinstance(word, str):
        #         new_row = inject_word_into_row(row, word, repeat=repeat)
        #     elif isinstance(word, tuple):
        #         new_row = inject_pair_word_into_row(row, word, repeat=repeat)
        #     else:
        #         raise ValueError()
        #     new_dataset.append(new_row.to_frame().T)
        # new_datasets[word] = pd.concat(new_dataset)
        new_datasets[word] = new_dataset

    out_data = sampler._create_dataset(data, dataset_params)
    out_mod_data = {k: sampler._create_dataset(v, dataset_params) for k, v in new_datasets.items()}

    return out_data, out_mod_data


def get_preds(tuned_model, eval_dataset: EMDataset):
    assert isinstance(eval_dataset, EMDataset), "Wrong data type for parameter 'eval_dataset'."

    tuned_model.to('cpu')
    tuned_model.eval()

    loader = DataLoader(eval_dataset, batch_size=32, num_workers=4, shuffle=False, collate_fn=eval_dataset.pad)

    labels = None
    preds = None
    for features in tqdm(loader):
        input_ids = features['input_ids']
        token_type_ids = features['token_type_ids']
        attention_mask = features['attention_mask']
        batch_labels = features["labels"].numpy()

        outputs = tuned_model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        logits = outputs['logits']
        batch_preds = torch.argmax(logits, axis=1).numpy()

        if labels is None:
            labels = batch_labels
            preds = batch_preds
        else:
            labels = np.concatenate((labels, batch_labels))
            preds = np.concatenate((preds, batch_preds))

    return labels, preds


def get_flipped_preds_stats(df_by_word, model, orig_preds):
    res = {}
    for word, em_df in df_by_word.items():
        _, mod_preds = get_preds(model, em_df)
        flip_idxs = np.where((orig_preds == 0) & (mod_preds == 1))[0]
        flip_perc = (len(flip_idxs) / len(mod_preds)) * 100
        print(word, flip_perc)

        if len(flip_idxs) > 0:
            flip_data = em_df.get_complete_data().iloc[flip_idxs]
            print(flip_data)
            res[word] = {'perc': flip_perc, 'num': len(flip_idxs), 'idxs': list(flip_data.index)}
        else:
            res[word] = {'perc': 0, 'num': 0, 'idxs': []}

    return res


def get_avg_record_length(dataset: EMDataset):
    loader = DataLoader(dataset, batch_size=32, num_workers=4, shuffle=False, collate_fn=dataset.pad)

    record_lengths = []
    for batch in loader:
        batch_word_ids = batch['word_ids']

        for wids in batch_word_ids:
            _, left_word_ids, right_word_ids = get_left_and_right_word_ids(list(wids))
            left_word_ids = np.array(left_word_ids)
            right_word_ids = np.array(right_word_ids)
            num_left_words = left_word_ids[left_word_ids != None].max() + 1
            num_right_words = right_word_ids[right_word_ids != None].max() + 1
            min_num_words = min(num_left_words, num_right_words)
            record_lengths.append(min_num_words)

    return np.mean(record_lengths)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Robustness effectiveness')

    # General parameters
    parser.add_argument('-use_cases', '--use_cases', nargs='+', required=True, choices=DM_USE_CASES + ['all'],
                        help='the names of the datasets')
    parser.add_argument('-bert_model', '--bert_model', default='bert-base-uncased', type=str,
                        help='the version of the BERT model')
    parser.add_argument('-tok', '--tok', default='sent_pair', type=str, choices=['sent_pair', 'attr_pair'],
                        help='the tokenizer for the EM entries')
    parser.add_argument('-label', '--label_col', default='label', type=str,
                        help='the name of the column in the EM dataset that contains the label')
    parser.add_argument('-left', '--left_prefix', default='left_', type=str,
                        help='the prefix used to identify the columns related to the left entity')
    parser.add_argument('-right', '--right_prefix', default='right_', type=str,
                        help='the prefix used to identify the columns related to the right entity')
    parser.add_argument('-max_len', '--max_len', default=128, type=int,
                        help='the maximum BERT sequence length')
    parser.add_argument('-permute', '--permute', default=False, type=lambda x: bool(distutils.util.strtobool(x)),
                        help='boolean flag for permuting dataset attributes')
    parser.add_argument('-v', '--verbose', default=False, type=lambda x: bool(distutils.util.strtobool(x)),
                        help='boolean flag for the dataset verbose modality')
    parser.add_argument('-out_dir', '--output_dir', type=str,
                        help='the directory where to store the results', required=True)
    parser.add_argument('-repeat', '--repeat', default=5, type=int, required=True,
                        help='How many times the injected words are repeated')

    args = parser.parse_args()
    pd.set_option('display.width', None)

    use_cases = args.use_cases
    if use_cases == ['all']:
        use_cases = DM_USE_CASES
    repeat = args.repeat

    results = {}
    for use_case in use_cases:
        conf = {
            'use_case': use_case,
            'model_name': args.bert_model,
            'tok': args.tok,
            'label_col': args.label_col,
            'left_prefix': args.left_prefix,
            'right_prefix': args.right_prefix,
            'max_len': args.max_len,
            'permute': args.permute,
            'verbose': args.verbose,
        }

        train_conf = conf.copy()
        train_conf['data_type'] = 'train'
        train_dataset = get_dataset(train_conf)

        test_conf = conf.copy()
        test_conf['data_type'] = 'test'
        test_dataset = get_dataset(test_conf)

        model_path = os.path.join(RESULTS_DIR, f"{use_case}_{args.tok}_tuned")
        tuned_model = AutoModelForSequenceClassification.from_pretrained(model_path)

        # Find the most frequent words in the matching records of the training set
        # top_train_match = get_most_freq_matching_words(train_dataset, 'match', 'singleton', topk=10)
        #
        # # Find the least frequent words in the matching records of the training set
        # bottom_train_match = get_most_freq_matching_words(train_dataset, 'match', 'singleton', topk=-10)

        random_words = get_random_matching_words(train_dataset, size=10)

        # # Insert the topk words of the train (match) in the test (non-match)
        # test_non_match, top_test_non_match = inject_words_into_dataset(
        #     test_dataset, 'non_match', top_train_match, repeat=5
        # )
        #
        # # Insert the bottomk words of the train (match) in the test (non-match)
        # _, bottom_test_non_match = inject_words_into_dataset(
        #     test_dataset, 'non_match', bottom_train_match, repeat=5
        # )

        # Insert the random words in the test (non-match)
        test_non_match, random_test_non_match = inject_words_into_dataset(
            test_dataset, 'non_match', random_words, repeat=repeat
        )

        # avg_record_length = get_avg_record_length(test_non_match)
        # print(f"UC: {use_case}, AVG_LEN: {avg_record_length}")

        _, orig_preds = get_preds(tuned_model, test_non_match)

        # print("#" * 30)
        # print("TOP")
        # print("#" * 30)
        # top_res = get_flipped_preds_stats(top_test_non_match, model_path, orig_preds)
        #
        # print("#" * 30)
        # print("BOTTOM")
        # print("#" * 30)
        # bottom_res = get_flipped_preds_stats(bottom_test_non_match, model_path, orig_preds)

        print("#" * 30)
        print("RANDOM")
        print("#" * 30)
        random_res = get_flipped_preds_stats(random_test_non_match, tuned_model, orig_preds)

        tot_res = {
            # 'top': top_res,
            # 'bottom': bottom_res,
            'random': random_res
        }

        with open(os.path.join(args.output_dir, f'INJECTION_{use_case}_repeat{repeat}.pickle'), 'wb') as fp:
            pickle.dump(tot_res, fp, protocol=pickle.HIGHEST_PROTOCOL)
