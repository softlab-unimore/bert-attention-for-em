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

from core.data_models.em_dataset import EMDataset
from utils.general import get_dataset, get_sample
from utils.data_collector import DM_USE_CASES
from utils.data_selection import Sampler
from experiments.robustness.perturbation import RelevanceAttributePerturbation
from functools import reduce
from collections import Counter
from itertools import product


def get_most_freq_words(df: pd.DataFrame, topk: int):

    word_counter = Counter()
    for ix, row in df.iterrows():
        row_wo_label = row.copy()
        del row_wo_label['label']
        sent = reduce(lambda a, b: f'{a} {b}', row_wo_label.values)
        word_counter.update(sent.split())

    return word_counter.most_common(topk)


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

    return pair_counter.most_common(topk)


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

        new_dataset = []
        for ix, row in data.iterrows():
            if isinstance(word, str):
                new_row = inject_word_into_row(row, word, repeat=repeat)
            elif isinstance(word, tuple):
                new_row = inject_pair_word_into_row(row, word, repeat=repeat)
            else:
                raise ValueError()
            new_dataset.append(new_row.to_frame().T)
        new_datasets[word] = pd.concat(new_dataset)

    out_data = sampler._create_dataset(data, dataset_params)
    out_mod_data = {k: sampler._create_dataset(v, dataset_params) for k, v in new_datasets.items()}

    return out_data, out_mod_data


def get_preds(model_name, eval_dataset: EMDataset):
    assert isinstance(eval_dataset, EMDataset), "Wrong data type for parameter 'eval_dataset'."

    tuned_model = AutoModelForSequenceClassification.from_pretrained(model_name)

    tuned_model.to('cpu')
    tuned_model.eval()

    loader = DataLoader(eval_dataset, batch_size=32, num_workers=4, shuffle=False)

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


RESULTS_DIR = "C:\\Users\\matte\\PycharmProjects\\bertAttention\\results\\models"


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
    parser.add_argument('-perturbation_type', '--perturbation_type', choices=['attr_rel'],
                        help='type of perturbation to apply to the data')

    args = parser.parse_args()
    pd.set_option('display.width', None)

    use_cases = args.use_cases
    if use_cases == ['all']:
        use_cases = DM_USE_CASES
    pert_type = args.perturbation_type

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

        test_conf = conf.copy()
        test_conf['data_type'] = 'test'
        test_dataset = get_dataset(test_conf)

        model_path = os.path.join(RESULTS_DIR, f"{use_case}_{args.tok}_tuned")

        top_match_single = get_most_freq_matching_words(test_dataset, 'match', 'singleton', topk=10)
        # top_non_match_single = get_most_freq_matching_words(test_dataset, 'non_match', 'singleton', topk=10)
        # top_match_pairs = get_most_freq_matching_words(test_dataset, 'match', 'pair', topk=5)
        # top_non_match_pairs = get_most_freq_matching_words(test_dataset, 'non_match', 'pair', topk=5)

        # mod_match_by_single = inject_words_into_dataset(test_dataset, 'match', top_non_match_single)
        non_match, mod_non_match_by_single = inject_words_into_dataset(
            test_dataset, 'non_match', top_match_single, repeat=5
        )
        # mod_match_by_pair = inject_words_into_dataset(test_dataset, 'match', top_non_match_pairs)
        # mod_non_match_by_pair = inject_words_into_dataset(test_dataset, 'non_match', top_match_pairs)

        # for word, em_df in mod_match_by_single.items():
        #     _, preds = get_preds(model_path, em_df)
        #     flip_idxs = np.where(preds == 0)[0]
        #     print(word, (len(flip_idxs) / len(preds)) * 100)
        #
        #     if len(flip_idxs) > 0:
        #         print(em_df.get_complete_data().iloc[flip_idxs])

        _, orig_preds = get_preds(model_path, non_match)

        res = {}
        for word, em_df in mod_non_match_by_single.items():
            _, mod_preds = get_preds(model_path, em_df)
            flip_idxs = np.where((orig_preds == 0) & (mod_preds == 1))[0]
            flip_perc = (len(flip_idxs) / len(mod_preds)) * 100
            print(word, flip_perc)

            if len(flip_idxs) > 0:
                flip_data = em_df.get_complete_data().iloc[flip_idxs]
                print(flip_data)
                res[word] = {'perc': flip_perc, 'num': len(flip_idxs), 'idxs': list(flip_data.index)}
            else:
                res[word] = {'perc': 0, 'num': 0, 'idxs': []}

        results[use_case] = res

        with open(f'word_occ_hacking_{use_case}_repeat5.pickle', 'wb') as fp:
            pickle.dump(res, fp, protocol=pickle.HIGHEST_PROTOCOL)

    # with open(f'word_occ_hacking_all.pickle', 'wb') as fp:
    #     pickle.dump(results, fp, protocol=pickle.HIGHEST_PROTOCOL)

        # for em_df in mod_match_by_pair.values():
        #     _, preds = get_preds(model_path, em_df)
        #     print((preds == 0).sum() / len(preds) * 100)
        #
        # for em_df in mod_non_match_by_pair.values():
        #     _, preds = get_preds(model_path, em_df)
        #     print((preds == 1).sum() / len(preds) * 100)



        # test_sample = get_sample(test_dataset, {'size': None, 'target_class': 1, 'permute': False, 'seeds': [42, 42]})
        #
        # row = test_sample[0]
        # input_ids = row['input_ids']
        # token_type_ids = row['token_type_ids']
        # attention_mask = row['attention_mask']
        #
        # model_path = os.path.join(RESULTS_DIR, f"{use_case}_{args.tok}_tuned")
        # tuned_model = AutoModelForSequenceClassification.from_pretrained(model_path)
        # outputs = tuned_model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        # logits = outputs['logits']
        # pred = torch.argmax(logits, axis=1).numpy()

    #     # Make perturbation
    #     if pert_type == 'attr_rel':
    #         pert_obj = RelevanceAttributePerturbation(test_dataset)
    #     else:
    #         raise NotImplementedError()
    #
    #     pert_dataset = pert_obj.apply_perturbation()
    #
    #     model_path = os.path.join(RESULTS_DIR, f"{use_case}_{args.tok}_tuned")
    #     f1 = evaluate(model_path, pert_dataset)
    #     results[use_case] = f1
    #
    # print(results)
    # print(":)")
