from utils.general import get_use_case, get_dataset, get_model, get_sample
from utils.nlp import get_similar_word_pairs
import os
import pandas as pd
from pathlib import Path
from tqdm import trange, tqdm
import torch
from transformers import AutoModel, AutoModelForSequenceClassification
import numpy as np
from scipy import spatial
import unicodedata
from utils.test_utils import ConfCreator
import matplotlib.pyplot as plt
import pickle
import gensim
import argparse
import distutils.util
from utils.data_collector import DM_USE_CASES


PROJECT_DIR = Path(__file__).parent.parent
MODELS_DIR = os.path.join(PROJECT_DIR, 'results', 'models')
RESULTS_DIR = os.path.join(PROJECT_DIR, 'results')
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
FAST_TEXT_PATH = os.path.join(DATA_DIR, 'wiki-news-300d-1M.vec', 'wiki-news-300d-1M.vec')


def get_use_case_entity_pairs(conf, sampler_conf):
    dataset = get_dataset(conf)

    complete_sampler_conf = sampler_conf.copy()
    complete_sampler_conf['permute'] = conf['permute']
    sample = get_sample(dataset, complete_sampler_conf)

    return sample


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


def get_word_pairs_emb_idxs(encoded_dataset, word_pairs):
    assert isinstance(word_pairs, dict)
    assert all(k in word_pairs for k in ['idxs', 'pairs'])

    def get_bert_tokenized_word(bert_tokens, idxs):
        word = ''
        for word_piece in bert_tokens[idxs[0]: idxs[1]]:
            while word_piece.startswith('#'):
                word_piece = word_piece[1:]
            word += word_piece

        return word

    def strip_accents(s):
        return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

    dataset_emb_idxs = []
    skips = []
    it = 0
    for i in trange(len(word_pairs['pairs'])):
        record_id = word_pairs['idxs'][i]
        record_word_pairs = word_pairs['pairs'][i]
        encoded_record = encoded_dataset[record_id][2]
        sent1 = encoded_record['sent1']
        sent2 = encoded_record['sent2']

        # get the indexes of all the words in the current record inside the BERT's embedding matrix
        left_word_idxs, right_word_idxs = get_pair_sent_word_idxs(encoded_record, sent1, sent2)

        sent1 = ' '.join([w.replace('.0', '') for w in sent1.split()])
        sent2 = ' '.join([w.replace('.0', '') for w in sent2.split()])

        # select the indexes for only the input word pairs
        emb_idxs = []
        for pair in record_word_pairs:
            left_target_idx = sent1.split().index(pair[0])
            right_target_idx = sent2.split().index(pair[1])

            # check if the target word has been truncated
            if left_target_idx >= len(left_word_idxs) or right_target_idx >= len(right_word_idxs):
                skips.append(it)
                it += 1
                continue

            left_emb_idx = left_word_idxs[left_target_idx]
            right_emb_idx = right_word_idxs[right_target_idx]

            # check consistency
            bert_tokens = encoded_dataset.tokenizer.convert_ids_to_tokens(encoded_record['input_ids'])
            left = strip_accents(pair[0]).lower()
            pred_left = get_bert_tokenized_word(bert_tokens, left_emb_idx).replace('.0', '').lower()
            right = strip_accents(pair[1]).lower()
            pred_right = get_bert_tokenized_word(bert_tokens, right_emb_idx).replace('.0', '').lower()

            if left != pred_left or right != pred_right:
                skips.append(it)
                it += 1
                continue

            emb_idxs.append((left_emb_idx, right_emb_idx))
            it += 1

        dataset_emb_idxs.append({
            'record_id': record_id,
            'encoded_record': encoded_record,
            'word_pair_emb_idxs': emb_idxs
        })

    return dataset_emb_idxs, skips


def get_word_pair_embs(encoded_dataset, word_pairs, model, emb_path=None):
    # Find the indexes of the word pairs in the BERT embedding matrix
    features_for_embs, skips = get_word_pairs_emb_idxs(encoded_dataset, word_pairs)

    embs = {}
    if emb_path is not None:
        with open(emb_path, 'rb') as f:
            embs = pickle.load(f)

    # Get word pair embeddings based on the previous indexes
    # Loop over the tokenized records and apply a forward pass in the BERT architecture
    word_pair_embs = []
    for feature_for_embs in tqdm(features_for_embs):
        record_id = feature_for_embs['record_id']
        word_pair_emb_idxs = feature_for_embs['word_pair_emb_idxs']
        encoded_record = feature_for_embs['encoded_record']
        label = encoded_record['labels'].item()

        if record_id in embs:
            all_embs = embs[record_id]
        else:
            input_ids = encoded_record['input_ids'].unsqueeze(0)
            attention_mask = encoded_record['attention_mask'].unsqueeze(0)
            token_type_ids = encoded_record['token_type_ids'].unsqueeze(0)
            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            # get all the hidden states
            all_embs = outputs['hidden_states'][-1].squeeze(0)

        for emb_idxs in word_pair_emb_idxs:
            # if the target word has been split in multiple pieces average the embeddings
            left_target_emb = all_embs[emb_idxs[0][0]:emb_idxs[0][1]].mean(0)
            right_target_emb = all_embs[emb_idxs[1][0]:emb_idxs[1][1]].mean(0)
            word_pair_embs.append((label, (left_target_emb, right_target_emb)))

    return word_pair_embs, skips


def get_bert_emb_knowledge(conf, sampler_conf, fine_tune, sim_type, sim_metric, sim_thrs, sim_op_eq=False,
                           bert_emb_thr=0.8, precomputed_embs=False, sem_emb_model=None, continuous_res=False):
    emb_knowledge = []
    for uc in conf['use_case']:
        print("\n\n", uc)
        uc_conf = conf.copy()
        uc_conf['use_case'] = uc

        # Get data
        encoded_dataset = get_use_case_entity_pairs(conf=uc_conf, sampler_conf=sampler_conf)

        # Get similar words
        pair_of_entities = [(row[0], row[1]) for row in encoded_dataset]
        similar_word_pair_map = get_similar_word_pairs(pair_of_entities=pair_of_entities, sim_type=sim_type,
                                                       metric=sim_metric, thrs=sim_thrs, op_eq=sim_op_eq,
                                                       sem_emb_model=sem_emb_model, continuous_res=continuous_res)

        # Get model
        emb_path = None
        if fine_tune is True:
            model_path = os.path.join(MODELS_DIR, f"{uc}_{conf['tok']}_tuned")
            model = AutoModelForSequenceClassification.from_pretrained(model_path, output_hidden_states=True)
            if precomputed_embs:
                emb_path = os.path.join(RESULTS_DIR, 'embs', uc, 'ft_sample_embs.pkl')
        else:
            model = AutoModel.from_pretrained(conf['model_name'], output_hidden_states=True)
            if precomputed_embs:
                emb_path = os.path.join(RESULTS_DIR, 'embs', uc, 'pt_sample_embs.pkl')

        # Get word pair embeddings and measure if they encode some similarity knowledge
        # Loop over the tested word pair similarity thresholds
        for thr in similar_word_pair_map:
            print(f"THR: {thr}")
            similar_word_pairs = similar_word_pair_map[thr]
            num_all_pairs = similar_word_pairs['num_all_pairs']
            sem_model_sims = [s for pair_sims in similar_word_pairs['sims'] for s in pair_sims]

            if len(similar_word_pairs['idxs']) > 0:
                # Get word pair embeddings from BERT
                embs, skips = get_word_pair_embs(encoded_dataset, similar_word_pairs, model, emb_path)

                # Get the total number of paired embeddings that have a similarity greater than the input threshold
                acc_numerator = 0
                bert_sims = []
                labels = []
                for label, (emb1, emb2) in embs:
                    sim = 1 - spatial.distance.cosine(emb1, emb2)
                    bert_sims.append(sim)
                    labels.append(label)
                    if sim > bert_emb_thr:
                        acc_numerator += 1
                acc = acc_numerator / len(embs)

                print(f"Accuracy: {acc * 100} ({acc_numerator}/{len(embs)}).")
                print(f"Coverage: {len(embs) / num_all_pairs * 100} ({len(embs)}/{num_all_pairs}).")

                emb_knowledge.append({
                    'use_case': uc,
                    'thr': thr,
                    'hits_sim_word_pairs': acc_numerator,
                    'tot_sim_word_pairs': len(embs),
                    'acc': acc * 100,
                    'tot_pairs': num_all_pairs,
                    'sim_pairs_coverage': (len(embs) / num_all_pairs) * 100,
                    'skips': len(skips),
                    'dataset_coverage': (len(similar_word_pairs) / len(pair_of_entities)) * 100,
                    'bert_sims': bert_sims,
                    'sem_model_sims': [sem_model_sims[x] for x in range(len(sem_model_sims)) if x not in skips],
                    'labels': labels
                })

                assert len(bert_sims) == len([sem_model_sims[x] for x in range(len(sem_model_sims)) if x not in skips])
                assert len(bert_sims) == len(labels)

            else:
                emb_knowledge.append({
                    'use_case': uc,
                    'thr': thr,
                    'hits_sim_word_pairs': None,
                    'tot_sim_word_pairs': 0,
                    'acc': None,
                    'tot_pairs': num_all_pairs,
                    'sim_pairs_coverage': None,
                    'skips': None,
                    'dataset_coverage': None,
                    'sims': None,
                    'bert_sims': None,
                    'sem_model_sims': None,
                    'labels': None
                })

    return emb_knowledge


def plot_results(results, save_path=None):
    plot_table = results.pivot_table(values='acc', index=['use_case', 'thr'], columns=['method'])

    ncols = 4
    nrows = 3
    figsize = (21, 8)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, sharey=True)
    axes = axes.flat

    # loop over the use cases
    for idx, use_case in enumerate(use_case_map.values()):
        ax = axes[idx]
        use_case_plot_data = plot_table[plot_table.index.get_level_values(0) == use_case]
        use_case_plot_data = use_case_plot_data.reset_index(level=0, drop=True)
        use_case_plot_data = use_case_plot_data[['pt', 'ft']]

        use_case_plot_data.plot(ax=ax, kind='bar', legend=False, rot=0)
        ax.set_title(use_case, fontsize=16)

        if idx % ncols == 0:
            ax.set_ylabel('Accuracy', fontsize=16)
        ax.set_xlabel('Thr', fontsize=16)
        ax.xaxis.set_tick_params(labelsize=14)
        ax.yaxis.set_tick_params(labelsize=14)
        start, end = ax.get_ylim()
        ax.yaxis.set_ticks(np.arange(start, end, 25))

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, bbox_to_anchor=(.58, 0.01), ncol=2, fontsize=16)

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.02, hspace=0.5)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')

    plt.show()


def _res_to_df(res, res_type):
    res_tab = pd.DataFrame(res)
    res_tab['use_case'] = res_tab['use_case'].map(use_case_map)
    if res_type is not True:
        method = 'pt'
    else:
        method = 'ft'
    res_tab['method'] = method

    return res_tab


def load_results(res_type, res_metric):

    # get pt results
    ft_model = False
    with open(os.path.join(RESULTS_DIR, 'embs', f'bert_emb_{res_type}_knowledge_{ft_model}_{res_metric}.pkl'), 'rb')\
            as f:
        pt_res = pickle.load(f)
    pt_res = _res_to_df(pt_res, ft_model)

    # get ft results
    ft_model = True
    with open(os.path.join(RESULTS_DIR, 'embs', f'bert_emb_{res_type}_knowledge_{ft_model}_{res_metric}.pkl'), 'rb')\
            as f:
        ft_res = pickle.load(f)
    ft_res = _res_to_df(ft_res, ft_model)

    res = pd.concat([pt_res, ft_res])

    return res


def plot_continuous_results(results, res_type, pt, ft, save_path=None):
    ncols = 4
    nrows = 3
    figsize = (18, 8)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, sharey=True, sharex=True)
    axes = axes.flat

    # loop over the use cases
    filter_use_cases = [x for x in use_case_map.values() if x in results['use_case'].unique()]
    for idx, use_case in enumerate(filter_use_cases):
        ax = axes[idx]
        uc_results = results[results['use_case'] == use_case]

        if ft:
            uc_ft_results = uc_results[uc_results['method'] == 'ft']
            ft_labels = np.array(uc_ft_results['labels'].values[0])
            ft_match_labels = ft_labels == 1
            ft_bert_sims = np.array(uc_ft_results["bert_sims"].values[0])
            ft_sem_model_sims = np.array(uc_ft_results["sem_model_sims"].values[0])
            ax.scatter(x=ft_bert_sims, y=ft_sem_model_sims, alpha=1, color='tab:blue', label='fine-tuned')
            # ax.scatter(x=ft_bert_sims[ft_match_labels], y=ft_sem_model_sims[ft_match_labels], alpha=0.3, color='orange')
            # ax.scatter(x=ft_bert_sims[~ft_match_labels], y=ft_sem_model_sims[~ft_match_labels], alpha=0.3, color='blue')

        if pt:
            uc_pt_results = uc_results[uc_results['method'] == 'pt']
            pt_labels = np.array(uc_pt_results['labels'].values[0])
            pt_match_labels = pt_labels == 1
            pt_bert_sims = np.array(uc_pt_results["bert_sims"].values[0])
            pt_sem_model_sims = np.array(uc_pt_results["sem_model_sims"].values[0])
            ax.scatter(x=pt_bert_sims, y=pt_sem_model_sims, alpha=0.3, color='tab:green', label='pre-trained')
            # ax.scatter(x=pt_bert_sims[pt_match_labels], y=pt_sem_model_sims[pt_match_labels], alpha=0.3, color='orange')
            # ax.scatter(x=pt_bert_sims[~pt_match_labels], y=pt_sem_model_sims[~pt_match_labels], alpha=0.3, color='blue')

        ax.set_title(use_case, fontsize=16)

        if idx % ncols == 0:
            if res_type == 'syntax':
                ylabel = 'Jaccard sim.'
            elif res_type == 'semantic':
                ylabel = 'FastText cosine sim.'
            else:
                raise NotImplementedError()
            ax.set_ylabel(ylabel, fontsize=16)

        if idx // ncols == nrows - 1:
            ax.set_xlabel('BERT cosine sim.', fontsize=16)
        ax.xaxis.set_tick_params(labelsize=14)
        ax.yaxis.set_tick_params(labelsize=14)
        ax.set_xlim((-1, 1))
        # start, end = ax.get_ylim()
        # ax.yaxis.set_ticks(np.arange(start, end, 25))

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, bbox_to_anchor=(.64, 0.01), ncol=2, fontsize=16)

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.1, hspace=0.2)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')

    plt.show()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Analysis of BERT and other model embeddings')

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

    # Parameters for data sampling
    parser.add_argument('-sample_size', '--sample_size', type=int,
                        help='size of the sample')
    parser.add_argument('-sample_target_class', '--sample_target_class', default='both', choices=['both', 0, 1],
                        help='classes to sample: match, non-match or both')
    parser.add_argument('-sample_seeds', '--sample_seeds', nargs='+', default=[42, 42],
                        help='seeds for each class sample. <seed non match> <seed match>')
    parser.add_argument('-ft', '--fine_tune', default=False,
                        type=lambda x: bool(distutils.util.strtobool(x)),
                        help='boolean flag for selecting fine-tuned or pre-trained model')

    # Parameters for embedding comparison
    parser.add_argument('-sim_metric', '--sim_metric', required=True, choices=['jaccard', 'edit', 'cosine'],
                        help='the similarity/distance function')
    parser.add_argument('-sim_thrs', '--sim_thrs', nargs='+', default=[0.7],
                        help='similarity threshold to reduce the exploration space')
    parser.add_argument('-sim_op_eq', '--sim_op_eq', default=False,
                        type=lambda x: bool(distutils.util.strtobool(x)),
                        help='if consider only words with the specified similarity thresholds. \
                            Otherwise the thresholds represent a lower bound for similarity functions \
                            or an upper bound for distance functions')
    parser.add_argument('-sem_embs', '--sem_embs', default=None, choices=['fasttext'],
                        help='the embedding model for measuring the semantic similarity. For fasttext download the \
                            embeddings from "https://fasttext.cc/docs/en/english-vectors.html" and save them in the data \
                            folder. Only the version "wiki-news-300d-1M" has been tested.')
    parser.add_argument('-task', '--task', default='compute', choices=['compute', 'visualize'], type=str,
                        help='task to run: 1) "compute": run the computation of the results, 2) "visualize": show \
                            pre-computed results')
    parser.add_argument('-bert_emb_thr', '--bert_emb_thr', type=float, default=0.8,
                        help='consider only the pair of words with a cosine similarity greater than the one provided \
                        in input')
    parser.add_argument('-precomputed_embs', '--precomputed_embs', default=False,
                        type=lambda x: bool(distutils.util.strtobool(x)),
                        help='compute the embeddings or load pre-compute ones')

    args = parser.parse_args()

    use_cases = args.use_cases
    if use_cases == ['all']:
        use_cases = DM_USE_CASES

    assert args.return_offset is True

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

    sampler_conf = {
        'size': args.sample_size,
        'target_class': args.sample_target_class,
        'seeds': args.sample_seeds,
    }

    fine_tune = args.fine_tune

    use_case_map = ConfCreator().use_case_map

    sim_metric = args.sim_metric
    sim_thrs = [float(x) for x in args.sim_thrs]
    sim_op_eq = args.sim_op_eq
    sem_embs = args.sem_embs
    bert_emb_thr = args.bert_emb_thr
    precomputed_embs = args.precomputed_embs
    task = args.task

    if sim_metric in ['jaccard', 'edit']:
        sim_type = 'syntax'

    elif sim_metric in ['cosine']:
        sim_type = 'semantic'
        if task == 'compute':
            assert sem_embs is not None

    else:
        raise NotImplementedError()

    if sim_type == 'syntax':

        if task == 'compute':
            syntactic_knowledge = get_bert_emb_knowledge(conf, sampler_conf, fine_tune, sim_type=sim_type,
                                                         sim_metric=sim_metric, sim_thrs=sim_thrs, sim_op_eq=sim_op_eq,
                                                         bert_emb_thr=bert_emb_thr, precomputed_embs=precomputed_embs,
                                                         continuous_res=True)
            out_fname = os.path.join(RESULTS_DIR, 'embs', f'bert_emb_{sim_type}_knowledge_{fine_tune}_{sim_metric}.pkl')

            with open(out_fname, 'wb') as f:
                pickle.dump(syntactic_knowledge, f)

        elif task == 'visualize':
            res = load_results(res_type=sim_type, res_metric=sim_metric)
            # plot_results(res)
            plot_save_path = os.path.join(RESULTS_DIR, f'PLOT_emb_{sim_type}_knowledge.pdf')
            plot_continuous_results(res, sim_type, pt=True, ft=True, save_path=plot_save_path)
            # plot_continuous_results(res, sim_type, pt=True, ft=False, save_path=plot_save_path)
            # plot_continuous_results(res, sim_type, pt=False, ft=True, save_path=plot_save_path)

        else:
            raise NotImplementedError()

    elif sim_type == 'semantic':

        if task == 'compute':

            if sem_embs == 'fasttext':

                if not os.path.exists(FAST_TEXT_PATH):
                    fasttext_repo = 'https://fasttext.cc/docs/en/english-vectors.html'
                    raise Exception(f"Download fastText embeddings from {fasttext_repo} and save them in {DATA_DIR}.")

                print("Loading fasttext embeddings...")
                sem_emb_model = gensim.models.KeyedVectors.load_word2vec_format(FAST_TEXT_PATH, binary=False,
                                                                                encoding='utf8')
                print("fasttext embeddings loaded.")

            else:
                raise NotImplementedError()

            semantic_knowledge = get_bert_emb_knowledge(conf, sampler_conf, fine_tune, sim_type=sim_type,
                                                        sim_metric=sim_metric, sim_thrs=sim_thrs, sim_op_eq=sim_op_eq,
                                                        bert_emb_thr=bert_emb_thr, precomputed_embs=precomputed_embs,
                                                        sem_emb_model=sem_emb_model, continuous_res=True)
            out_fname = os.path.join(RESULTS_DIR, 'embs', f'bert_emb_{sim_type}_knowledge_{fine_tune}_{sim_metric}.pkl')

            with open(out_fname, 'wb') as f:
                pickle.dump(semantic_knowledge, f)

        elif task == 'visualize':
            res = load_results(res_type=sim_type, res_metric=sim_metric)
            # plot_results(res)
            plot_save_path = os.path.join(RESULTS_DIR, f'PLOT_emb_{sim_type}_knowledge.png')
            plot_continuous_results(res, sim_type, pt=True, ft=True, save_path=plot_save_path)
            # plot_continuous_results(res, sim_type, pt=True, ft=False, save_path=plot_save_path)
            # plot_continuous_results(res, sim_type, pt=False, ft=True, save_path=plot_save_path)

        else:
            raise NotImplementedError()

    print(":)")
