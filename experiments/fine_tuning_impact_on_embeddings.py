import pandas as pd

from utils.general import get_dataset, get_model, get_sample
from pathlib import Path
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoModel, AutoModelForSequenceClassification
from tqdm import tqdm, trange
import itertools
from collections import Counter
import unicodedata
from multiprocessing import Process, Manager
import pickle
from utils.test_utils import ConfCreator
from sklearn.metrics import mean_absolute_error
import random
import argparse
from utils.data_collector import DM_USE_CASES
import distutils.util
import seaborn as sns

"""
Code adapted from https://github.com/text-machine-lab/dark-secrets-of-BERT/blob/master/visualize_attention.ipynb
"""

PROJECT_DIR = Path(__file__).parent.parent
MODELS_DIR = os.path.join(PROJECT_DIR, 'results', 'models')
RESULTS_DIR = os.path.join(PROJECT_DIR, 'results', 'embs')


def get_use_case_entity_pairs(conf, sampler_conf):
    dataset = get_dataset(conf)

    complete_sampler_conf = sampler_conf.copy()
    complete_sampler_conf['permute'] = conf['permute']
    sample = get_sample(dataset, complete_sampler_conf)

    return sample


def get_most_freq_word_pairs_by_label(entity_pairs, size, seed=42):
    def get_all_word_pairs(left, right):
        # remove short words from the two entities
        min_len = 3
        left = [l for l in left if len(l) > min_len]
        right = [r for r in right if len(r) > min_len]

        # generate all possible pairs of words (x, y), where x belong to the left entity and y to the right one
        word_pairs = list(itertools.product(left, right))

        return word_pairs

    np.random.seed(seed)
    random.seed(seed)
    random_idxs = np.random.permutation(len(entity_pairs))

    print("Getting most freq word pairs by label...")
    # find the most frequent pairs of words from matching/non-matching records and select some random pairs of words

    freq_map = {}
    random_pairs = {}
    # loop over the dataset entity pairs
    for idx, entity in enumerate(entity_pairs):
        left = entity[0].split()
        right = entity[1].split()
        label = entity[2]

        word_pairs = get_all_word_pairs(left, right)

        # remove pairs composed by equal words
        word_pairs = [p for p in word_pairs if p[0] != p[1]]
        # count word pair frequencies
        entity_freq_map = Counter(word_pairs)

        if label not in freq_map:
            freq_map[label] = entity_freq_map
        else:
            # update word pair frequencies
            freq_map[label] += entity_freq_map

        # select a random word pair
        if len(random_pairs) < size:
            if idx in random_idxs:
                random_pair = random.choice(word_pairs)
                it = 1
                while random_pair in random_pairs and it < 10:
                    random_pair = random.choice(word_pairs)
                    it += 1
                random_pairs[random_pair] = 1

    # sample new random pairs if the size has not been reached
    it_count = 1
    while len(random_pairs) < size and it_count < 1000:
        random_idx = random.randint(0, len(entity_pairs) - 1)
        random_entity = entity_pairs[random_idx]
        left = random_entity[0].split()
        right = random_entity[1].split()

        # get all word pairs
        word_pairs = get_all_word_pairs(left, right)

        random_pair = random.choice(word_pairs)
        it = 1
        while random_pair in random_pairs and it < 1000:
            random_pair = random.choice(word_pairs)
            it += 1
        random_pairs[random_pair] = 1
        it_count += 1

    random_pairs = list(random_pairs.keys())

    # get top 'size' matching word pairs that don't appear in non-matching pairs
    most_freq_match_pairs = []
    for pair in [p[0] for p in freq_map[1].most_common(len(freq_map[1]))]:
        if pair not in freq_map[0]:
            most_freq_match_pairs.append(pair)

        if len(most_freq_match_pairs) == size:
            break

    # get top 'size' non-matching word pairs that don't appear in matching pairs
    most_freq_nonmatch_pairs = []
    for pair in [p[0] for p in freq_map[0].most_common(len(freq_map[0]))]:
        if pair not in freq_map[1]:
            most_freq_nonmatch_pairs.append(pair)

        if len(most_freq_nonmatch_pairs) == size:
            break

    assert len(most_freq_match_pairs) == size
    assert len(most_freq_nonmatch_pairs) == size
    assert len(random_pairs) == size

    out_dict = {0: most_freq_nonmatch_pairs, 1: most_freq_match_pairs, 'random': random_pairs}

    return out_dict


def get_sent_word_idxs(offsets: list, sent):
    # aggregate all tokens of the sentence that refer to the same word
    # these tokens can be detected by searching for adjacent offsets from the
    # `offset_mapping` parameter
    tokens_to_sent_offsets = offsets[:]
    tokens_by_word = []  # this list will aggregate the token offsets by word
    prec_token_offsets = None
    tokens_in_word = []  # this list will accumulate all the tokens that refer to a target word
    words_offsets = []  # this list will store for each word the range of token idxs that refer to it
    for ix, token_offsets in enumerate(tokens_to_sent_offsets):

        # special tokens (e.g., [CLS], [SEP]) do not refer to any words
        # their offsets are equal to (0, 0)
        if token_offsets == [0, 0]:

            # save all the tokens that refer to the previous word
            if len(tokens_in_word) > 0:
                l = int(np.sum([len(x) for x in tokens_by_word]))
                words_offsets.append((l, l + len(tokens_in_word)))
                tokens_by_word.append(tokens_in_word)
                prec_token_offsets = None
                tokens_in_word = []

            l = int(np.sum([len(x) for x in tokens_by_word]))
            # words_offsets.append((l, l + 1))
            tokens_by_word.append([token_offsets])
            continue

        if prec_token_offsets is None:
            tokens_in_word.append(token_offsets)
        else:
            # if the offsets of the current and previous tokens are adjacent then they
            # refer to the same word
            if prec_token_offsets[1] == token_offsets[0]:
                tokens_in_word.append(token_offsets)
            else:
                # the current token refers to a new word

                # save all the tokens that refer to the previous word
                l = int(np.sum([len(x) for x in tokens_by_word]))
                words_offsets.append((l, l + len(tokens_in_word)))
                tokens_by_word.append(tokens_in_word)

                tokens_in_word = [token_offsets]

        prec_token_offsets = token_offsets

    # Note that 'words_offsets' contains only real word offsets, i.e. offsets
    # for special tokens (e.g., [CLS], [SEP], [PAD]), except for the [UNK]
    # token, are omitted

    return words_offsets


def get_pair_sent_word_idxs(encoded_pair_sent, sent1, sent2):
    # split the offset mappings at sentence level by exploiting the [SEP] which
    # is identified with the offsets [0, 0] (as any other special tokens)
    offsets = encoded_pair_sent['offset_mapping'].squeeze(0).tolist()
    sep_idx = offsets[1:].index([0, 0])  # ignore the [CLS] token at the index 0
    left_offsets = offsets[:sep_idx + 2]
    right_offsets = offsets[sep_idx + 1:]

    left_word_idxs = get_sent_word_idxs(left_offsets, sent1)
    right_word_idxs = get_sent_word_idxs(right_offsets, sent2)
    right_word_idxs = [(item[0] + sep_idx + 1, item[1] + sep_idx + 1) for item in right_word_idxs]

    return left_word_idxs, right_word_idxs


def get_word_pairs_embedding_idxs(dataset, pairs_map):
    def get_bert_tokenized_word(bert_tokens, idxs):
        word = ''
        for word_piece in bert_tokens[idxs[0]: idxs[1]]:
            while word_piece.startswith('#'):
                word_piece = word_piece[1:]
            word += word_piece

        return word

    def strip_accents(s):
        return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

    def get_word_pair_idxs(dataset, sent1, sent2, encoded_row, pair, left_word_idxs, right_word_idxs):
        # get word indexes from the original sentences
        left_target_idx = sent1.split().index(pair[0])
        right_target_idx = sent2.split().index(pair[1])

        # check if the target word has been truncated
        if left_target_idx >= len(left_word_idxs) or right_target_idx >= len(right_word_idxs):
            return None, 0

        # select BERT's embedding matrix indexes for the target pair of words
        left_emb_idx = left_word_idxs[left_target_idx]
        right_emb_idx = right_word_idxs[right_target_idx]

        # consistency check
        bert_tokens = dataset.tokenizer.convert_ids_to_tokens(encoded_row['input_ids'])
        left = strip_accents(pair[0]).lower()
        pred_left = get_bert_tokenized_word(bert_tokens, left_emb_idx).lower()  # .replace('.0', '').lower()
        right = strip_accents(pair[1]).lower()
        pred_right = get_bert_tokenized_word(bert_tokens, right_emb_idx).lower()  # .replace('.0', '').lower()

        if left != pred_left or right != pred_right:
            # the word pair reconstructed from BERT doesn't coincides with the original word pair
            # print("Skip.")
            # print(f"{left} vs {pred_left}")
            # print(f"{right} vs {pred_right}")
            return None, 1

        return (left_emb_idx, right_emb_idx), 0

    emb_idxs = {}
    bert_features = {}
    skips = 0
    # loop over the records of the dataset and filter-out those that contain no input pairs
    for i in trange(len(dataset)):
        encoded_row = dataset[i][2]
        sent1 = encoded_row['sent1']
        sent2 = encoded_row['sent2']
        sent1 = ' '.join([w for w in sent1.split()])
        sent2 = ' '.join([w for w in sent2.split()])
        label = encoded_row['labels'].item()

        pairs_to_check = {label: [p for p in pairs_map[label] if p[0] in sent1.split() and p[1] in sent2.split()]}
        non_label_based_keys = [k for k in pairs_map if k not in [0, 1]]
        other_pairs_to_check = {k: [p for p in pairs_map[k] if p[0] in sent1.split() and p[1] in sent2.split()] for k in
                                non_label_based_keys}
        pairs_to_check.update(other_pairs_to_check)

        # skip the record if it don't contain any input pair
        if all([len(p_list) == 0 for p_list in pairs_to_check.values()]):
            continue

        bert_features[i] = encoded_row

        # get the BERT's embedding matrix indexes for all the words of the current record
        left_word_idxs, right_word_idxs = get_pair_sent_word_idxs(encoded_row, sent1, sent2)

        # select only the indexes that refer to the word pairs contained in the current record
        all_pair_idxs = {}
        for k in pairs_to_check:
            pair_list = pairs_to_check[k]

            if len(pair_list) > 0:

                pair_idxs_map = {}
                for pair in pair_list:
                    pair_idxs, skip = get_word_pair_idxs(dataset, sent1, sent2, encoded_row, pair, left_word_idxs,
                                                         right_word_idxs)

                    if pair_idxs is not None:
                        pair_idxs_map[pair] = (i, pair_idxs)  # record id, (word 1 idxs, word 2 idxs)
                    else:
                        skips += skip

                all_pair_idxs[k] = pair_idxs_map.copy()

        # save the indexes in a map
        # key -> {
        #   pair: [(record id, (word 1 idxs, word 2 idxs)), ..., (record id, (word 1 idxs, word 2 idxs))]
        # }
        for k in all_pair_idxs:
            pair_idxs = all_pair_idxs[k]

            if k not in emb_idxs:
                emb_idxs[k] = {p: [pair_idxs[p]] for p in pair_idxs}
            else:
                for p in pair_idxs:
                    if p in emb_idxs[k]:
                        emb_idxs[k][p].append(pair_idxs[p])
                    else:
                        emb_idxs[k][p] = [pair_idxs[p]]

    return bert_features, emb_idxs, skips


def get_word_pairs_emb_sim(model, dataset, pairs_map, save_path=None, embs_path=None):
    # Find the indexes of the word pairs in the BERT embedding matrix
    dataset_bert_features, dataset_emb_idxs, skips = get_word_pairs_embedding_idxs(dataset, pairs_map)

    print("Get word pair embeddings...")
    # Get pair word embeddings based on the previous indexes
    # Loop over the dataset tokenized records and apply a forward pass in the BERT architecture
    if embs_path is not None:
        print("Loading pre-computed embeddings...")
        with open(embs_path, 'rb') as f:
            embs = pickle.load(f)
        print("Loaded.")
    else:
        embs = {}
    dataset_pair_embs = {}
    # loop over match, non-match and random word pairs
    for key in dataset_emb_idxs:
        print(f"Extracting embeddings related to the key [{key}]...")
        emb_idxs = dataset_emb_idxs[key]

        record_pair_embs = {}
        # loop over the word pairs
        for pair in tqdm(emb_idxs):

            # loop over the records where the current word pair has occurred
            # [(record id, (word 1 idxs, word 2 idxs)), ..., (record id, (word 1 idxs, word 2 idxs))]
            pair_embs = []
            for pair_features in emb_idxs[pair]:
                # get BERT features for the current record and get bert embeddings
                record_id = pair_features[0]

                if record_id in embs:
                    all_embs = embs[record_id]

                else:
                    record_bert_features = dataset_bert_features[record_id]
                    input_ids = record_bert_features['input_ids'].unsqueeze(0)
                    attention_mask = record_bert_features['attention_mask'].unsqueeze(0)
                    token_type_ids = record_bert_features['token_type_ids'].unsqueeze(0)

                    # BERT forward
                    with torch.no_grad():
                        outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                                        token_type_ids=token_type_ids)

                    # get all hidden states
                    all_embs = outputs['hidden_states'][-1].squeeze(0)
                    embs[record_id] = all_embs

                # extract only the hidden states for the current word pair
                pair_emb_idxs = pair_features[1]
                # if the target word has been split in multiple pieces average the embeddings
                left_target_emb = all_embs[pair_emb_idxs[0][0]:pair_emb_idxs[0][1]].mean(0)
                right_target_emb = all_embs[pair_emb_idxs[1][0]:pair_emb_idxs[1][1]].mean(0)

                pair_embs.append((left_target_emb, right_target_emb))

            # concat the embeddings related to the left word (the same for the right one)
            # compute the cosine similarity between left and right embeddings
            left_embs = torch.cat([p[0].unsqueeze(0) for p in pair_embs])
            right_embs = torch.cat([p[1].unsqueeze(0) for p in pair_embs])
            sim_map = torch.nn.functional.cosine_similarity(left_embs, right_embs, dim=-1).detach().numpy()

            record_pair_embs[pair] = sim_map

        dataset_pair_embs[key] = record_pair_embs

    if save_path is not None:
        print("Saving the embeddings...")
        with open(save_path, 'wb') as f:
            pickle.dump(embs, f)
        print("Saved.")

    return dataset_pair_embs, skips


def get_emb_sim_variation(pt_emb_sim, ft_emb_sim, thr=0.8):
    emb_variation = {}
    # loop over match, non-match and random word pair embeddings
    for key in pt_emb_sim:
        key_pt_emb_sim = pt_emb_sim[key]
        key_ft_emb_sim = ft_emb_sim[key]

        pt_sims = []
        ft_sims = []
        weights = []
        sim_micro_acc_list = []
        sim_macro_acc_list = []
        sim_micro_mae_list = []
        pt_sim_gt_thr_list = []
        ft_sim_gt_thr_list = []
        # loop over the word pairs
        for pair in key_pt_emb_sim:
            pair_pt_emb_sim = key_pt_emb_sim[pair]
            pair_ft_emb_sim = key_ft_emb_sim[pair]
            pt_sims += list(pair_pt_emb_sim)
            ft_sims += list(pair_ft_emb_sim)

            # check if the fine-tuning has increased the similarity between the embeddings associated to match,
            # non-match and random word pairs
            sim_micro_acc = (pair_ft_emb_sim > pair_pt_emb_sim).sum() / len(pair_ft_emb_sim)
            sim_macro_acc = (pair_ft_emb_sim > pair_pt_emb_sim).sum()
            pt_sim_gt_thr = (pair_pt_emb_sim > thr).sum() / len(pair_pt_emb_sim)
            ft_sim_gt_thr = (pair_ft_emb_sim > thr).sum() / len(pair_ft_emb_sim)

            weights.append(len(pair_ft_emb_sim))
            sim_micro_acc_list.append(sim_micro_acc)
            sim_macro_acc_list.append(sim_macro_acc)
            sim_micro_mae_list.append(mean_absolute_error(pair_pt_emb_sim, pair_ft_emb_sim))
            pt_sim_gt_thr_list.append(pt_sim_gt_thr)
            ft_sim_gt_thr_list.append(ft_sim_gt_thr)

        emb_var = {
            'avg_gt_micro_acc': np.average(sim_micro_acc_list, weights=weights / np.sum(weights)),
            'avg_gt_macro_acc': np.sum(sim_macro_acc_list) / np.sum(weights),
            'avg_micro_mae': np.average(sim_micro_mae_list, weights=weights / np.sum(weights)),
            'avg_macro_mae': mean_absolute_error(pt_sims, ft_sims),
            'gt_micro_acc_list': sim_micro_acc_list,
            'gt_macro_acc_list': sim_macro_acc_list,
            'micro_mae_list': sim_micro_mae_list,
            'weights': weights,
            'pt_sims': pt_sims,
            'ft_sims': ft_sims,
            'pt_sim_gt_thr': np.average(pt_sim_gt_thr_list, weights=weights / np.sum(weights)),
            'ft_sim_gt_thr': np.average(ft_sim_gt_thr_list, weights=weights / np.sum(weights)),
        }
        emb_variation[key] = emb_var

    return emb_variation


def get_ft_impact_on_uc_embeddings(uc_conf, sampler_conf, num_pairs, precomputed_embs=False, save=False,
                                   queue=None):
    # Get data
    encoded_dataset = get_use_case_entity_pairs(conf=uc_conf, sampler_conf=sampler_conf)
    entity_pairs = [(row[2]['sent1'], row[2]['sent2'], row[2]['labels'].item()) for row in encoded_dataset]

    # Get some word pairs
    pairs_map = get_most_freq_word_pairs_by_label(entity_pairs, size=num_pairs)

    # Get pre-trained model
    pt_model = AutoModel.from_pretrained(uc_conf['model_name'], output_hidden_states=True)

    # Get fine-tuned model
    ft_model_path = os.path.join(MODELS_DIR, f"{uc_conf['use_case']}_{uc_conf['tok']}_tuned")
    ft_model = AutoModelForSequenceClassification.from_pretrained(ft_model_path, output_hidden_states=True)

    # Get matching and non-matching embedding pairs
    print(f"[{uc_conf['use_case']}] Get pre-trained embeddings...")
    pt_save_path = os.path.join(RESULTS_DIR, uc_conf['use_case'], 'pt_sample_embs.pkl')
    Path(os.path.join(RESULTS_DIR, uc_conf['use_case'])).mkdir(parents=True, exist_ok=True)
    embs_path = None
    save_path = None
    if precomputed_embs is True:
        embs_path = pt_save_path
    if save is True:
        save_path = pt_save_path
    pt_emb_sim, skips = get_word_pairs_emb_sim(pt_model, encoded_dataset, pairs_map, save_path=save_path,
                                               embs_path=embs_path)
    print(f"Skips: {skips}")

    print(f"[{uc_conf['use_case']}] Get fine-tuned embeddings...")
    ft_save_path = os.path.join(RESULTS_DIR, uc_conf['use_case'], 'ft_sample_embs.pkl')
    embs_path = None
    save_path = None
    if precomputed_embs is True:
        embs_path = ft_save_path
    if save is True:
        save_path = ft_save_path
    ft_emb_sim, skips = get_word_pairs_emb_sim(ft_model, encoded_dataset, pairs_map, save_path=save_path,
                                               embs_path=embs_path)
    print(f"Skips: {skips}")

    # Evaluate variation in embedding similarities
    sim_emb_variation = get_emb_sim_variation(pt_emb_sim, ft_emb_sim)

    if queue is not None:
        queue.put((uc_conf['use_case'], sim_emb_variation))
    else:
        return sim_emb_variation


def get_ft_impact_on_embeddings(conf, sampler_conf, num_pairs, precomputed_embs=False, save=False):
    uc_emb_impacts = {}
    for uc in conf['use_case']:
        print("\n\n", uc)
        uc_conf = conf.copy()
        uc_conf['use_case'] = uc

        sim_emb_variation = get_ft_impact_on_uc_embeddings(uc_conf, sampler_conf, num_pairs=num_pairs,
                                                           precomputed_embs=precomputed_embs, save=save)

        uc_emb_impacts[uc] = sim_emb_variation

    # uc_map = ConfCreator().use_case_map
    # uc_emb_impacts = pd.DataFrame(list(uc_emb_impacts.values()),
    #                               index=[uc_map[uc] for uc in list(uc_emb_impacts.keys())])

    return uc_emb_impacts


def get_ft_impact_on_embeddings_multi_process(conf, sampler_conf, num_pairs, precomputed_embs=False, save=False):
    processes = []
    m = Manager()
    queue = m.Queue()
    for use_case in conf['use_case']:
        uc_conf = conf.copy()
        uc_conf['use_case'] = use_case

        p = Process(target=get_ft_impact_on_uc_embeddings,
                    args=(uc_conf, sampler_conf, num_pairs, precomputed_embs, save, queue,))
        processes.append(p)

    for p in processes:
        p.start()

    for p in processes:
        p.join()

    uc_emb_impacts = {}
    while not queue.empty():
        item = queue.get()
        uc_emb_impacts[item[0]] = item[1]

    return uc_emb_impacts


def plot_embedding_variation(emb_impacts):

    # prepare data for plot
    target_metrics = ['pt_sim_gt_thr', 'ft_sim_gt_thr']  # 'avg_micro_mae', 'avg_gt_micro_acc'
    uc_map = ConfCreator().use_case_map

    selected_use_cases = [uc for uc in uc_map if uc in emb_impacts]
    assert len(selected_use_cases) == len(emb_impacts)

    plot_data = []
    for uc in selected_use_cases:
        uc_emb_impacts = emb_impacts[uc]

        for k in uc_emb_impacts:
            for m in target_metrics:
                if m.startswith('pt_'):
                    method = 'pt'
                elif m.startswith('ft_'):
                    method = 'ft'
                else:
                    method = None

                plot_data.append({
                    'cat': k,
                    'use_case': uc,
                    'method': method,
                    'score': uc_emb_impacts[k][m]
                })

    plot_data_tab = pd.DataFrame(plot_data)
    plot_data_tab['cat'] = plot_data_tab['cat'].map({0: 'non-match', 'random': 'random', 1: 'match'})
    plot_data_tab['method'] = plot_data_tab['method'].map({'pt': 'pre-trained', 'ft': 'fine-tuned'})
    plot_data_tab['score'] = plot_data_tab['score'] * 100

    # plot the results
    sns.set(style="ticks")
    fig = plt.gcf()
    fig.set_size_inches(5, 2)
    # my_pal = {'pre-trained': '#1f77b4', 'fine-tuned': '#ff7f0e'}
    g = sns.boxplot(y='score', x='cat',
                    data=plot_data_tab,
                    palette='Blues',
                    hue='method', linewidth=1, width=0.5, flierprops=dict(marker='o', markersize=2))
    ax = plt.gca()
    handles, labels = ax.get_legend_handles_labels()
    ax.legend([], [], frameon=False)
    fig.legend(handles=handles, labels=labels, ncol=2, bbox_to_anchor=(.8, 1.08))
    ax.grid(True)
    plt.xlabel('')
    plt.ylabel('Freq. (%)')
    # plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, 'PLOT_ft_pt_emb_dist.pdf')
    plt.savefig(plot_path, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Impact of the EM fine-tuning on the BERT embeddings')

    # General parameters
    parser.add_argument('-use_cases', '--use_cases', nargs='+', required=True, choices=DM_USE_CASES + ['all'],
                        help='the names of the datasets')
    parser.add_argument('-data_type', '--data_type', type=str, default='train', choices=['train', 'test', 'valid'],
                        help='dataset types: train, test or valid')
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
    parser.add_argument('-return_offset', '--return_offset', default=True,
                        type=lambda x: bool(distutils.util.strtobool(x)),
                        help='boolean flag for extracting EM entry word indexes')
    parser.add_argument('-multi_process', '--multi_process', default=False,
                        type=lambda x: bool(distutils.util.strtobool(x)),
                        help='multi process modality')

    # Parameters for data sampling
    parser.add_argument('-sample_size', '--sample_size', type=int,
                        help='size of the sample')
    parser.add_argument('-sample_target_class', '--sample_target_class', default='both', choices=['both', 0, 1],
                        help='classes to sample: match, non-match or both')
    parser.add_argument('-sample_seeds', '--sample_seeds', nargs='+', default=[42, 42],
                        help='seeds for each class sample. <seed non match> <seed match>')

    # Parameters for the embedding analysis
    parser.add_argument('-precomputed_embs', '--precomputed_embs', default=False,
                        type=lambda x: bool(distutils.util.strtobool(x)),
                        help='avoid re-computing the embeddings by loading previously saved results \
                            (with the --save_embs option)')
    parser.add_argument('-save_embs', '--save_embs', default=True,
                        type=lambda x: bool(distutils.util.strtobool(x)),
                        help='boolean flag for saving the computed embeddings')
    parser.add_argument('-precomputed_res', '--precomputed_res', default=False,
                        type=lambda x: bool(distutils.util.strtobool(x)),
                        help='avoid re-computing the impacts of the EM fine-tuning on the BERT embeddings by loading \
                            previously saved results (by default they are saved at each run)')
    parser.add_argument('-num_pairs', '--num_pairs', default=1000, type=int,
                        help='number of random pairs to sample')

    args = parser.parse_args()

    use_cases = args.use_cases
    if use_cases == ['all']:
        use_cases = DM_USE_CASES

    conf = {
        'use_case': use_cases,
        'data_type': args.data_type,
        'model_name': args.bert_model,
        'tok': args.tok,
        'label_col': args.label_col,
        'left_prefix': args.left_prefix,
        'right_prefix': args.right_prefix,
        'max_len': args.max_len,
        'permute': args.permute,
        'verbose': args.verbose,
        'return_offset': args.return_offset,
    }

    assert conf['return_offset'] is True

    sampler_conf = {
        'size': args.sample_size,
        'target_class': args.sample_target_class,
        'seeds': args.sample_seeds,
    }

    res_path = os.path.join(RESULTS_DIR, 'ft_pt_emb_dist_with_random_thr.pkl')

    if not args.precomputed_res:
        if args.multi_process:
            emb_impacts = get_ft_impact_on_embeddings_multi_process(conf, sampler_conf, args.num_pairs,
                                                                    precomputed_embs=args.precomputed_embs,
                                                                    save=args.save_embs)
        else:
            emb_impacts = get_ft_impact_on_embeddings(conf, sampler_conf, args.num_pairs,
                                                      precomputed_embs=args.precomputed_embs,
                                                      save=args.save_embs)

        # save the results
        with open(res_path, 'wb') as f:
            pickle.dump(emb_impacts, f)

    else:
        # load pre-computed results
        with open(res_path, 'rb') as f:
            emb_impacts = pickle.load(f)

    plot_embedding_variation(emb_impacts)

    print(":)")
