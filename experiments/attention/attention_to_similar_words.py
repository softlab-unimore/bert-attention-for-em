from utils.general import get_dataset, get_sample
from utils.nlp import get_similar_word_pairs
import os
import pandas as pd
from pathlib import Path
from tqdm import trange, tqdm
from utils.test_utils import ConfCreator
import matplotlib.pyplot as plt
import pickle
import gensim
import numpy as np
from utils.attention_utils import load_saved_attn_data
import argparse
import distutils.util
from utils.data_collector import DM_USE_CASES


PROJECT_DIR = Path(__file__).parent.parent.parent
MODELS_DIR = os.path.join(PROJECT_DIR, 'results', 'models')
RESULTS_DIR = os.path.join(PROJECT_DIR, 'results', 'attention')
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
FAST_TEXT_PATH = os.path.join(DATA_DIR, 'wiki-news-300d-1M.vec', 'wiki-news-300d-1M.vec')


def get_use_case_entity_pairs(conf, sampler_conf):
    dataset = get_dataset(conf)

    complete_sampler_conf = sampler_conf.copy()
    complete_sampler_conf['permute'] = conf['permute']
    sample = get_sample(dataset, complete_sampler_conf)

    return sample


def get_top_word_pairs_by_attn(attns, tokenization='sent_pair'):
    attn_data = {}
    for record_id, attn in enumerate(attns):
        attn_features = attn[2]
        attn_values = attn_features['attns']
        attn_text_units = attn_features['text_units']
        label = int(attn_features['labels'].item())
        pred = int(attn_features['preds'].item()) if attn_features['preds'] is not None else None

        # find the [SEP] token used to delimit the two entities
        sep_idxs = np.where(np.array(attn_text_units) == '[SEP]')[0]

        # filter out truncated rows
        if len(sep_idxs) % 2 != 0 or attn_values is None:
            print("Skip truncated row.")
            continue

        if tokenization == 'sent_pair':
            entity_delimit = attn_text_units.index('[SEP]')  # get first occurrence of the [SEP] token

        else:  # attr-pair
            # in the attr-pair tokenization the [SEP] token is also used to delimit the attributes
            entity_delimit = sep_idxs[(len(sep_idxs) // 2) - 1]

        # get an average attention map for each layer by averaging all the heads referring to the same layer
        attn_values = attn_values.mean(axis=1)

        # ignore the special tokens and related attention weights
        left_idxs = list(range(entity_delimit + 1))
        right_idxs = list(range(entity_delimit, attn_values.shape[1]))
        left_idxs = left_idxs[1:]  # remove [CLS]
        left_idxs = sorted(list(set(left_idxs).difference(set(sep_idxs))))  # remove [SEP]s
        right_idxs = sorted(list(set(right_idxs).difference(set(sep_idxs))))  # remove [SEP]s
        valid_idxs = np.array(left_idxs + right_idxs)
        attn_values = attn_values[:, valid_idxs, :][:, :, valid_idxs]
        valid_attn_text_units = list(np.array(attn_text_units)[valid_idxs])
        left_words = list(np.array(attn_text_units)[left_idxs])
        right_words = list(np.array(attn_text_units)[right_idxs])

        assert attn_values.shape[1] == len(valid_attn_text_units)
        original_left = [w for attr in attn[0] for w in str(attr).split()]
        original_right = [w for attr in attn[1] for w in str(attr).split()]
        assert all([left_words[i] == original_left[i] for i in range(len(left_words))])
        assert all([right_words[i] == original_right[i] for i in range(len(right_words))])

        for layer in range(attn_values.shape[0]):
            layer_attn_values = attn_values[layer]

            attn_scores = np.maximum(layer_attn_values[:len(left_idxs), len(left_idxs):],
                                     layer_attn_values[len(left_idxs):, :len(left_idxs)].T)

            thr = np.quantile(attn_scores, 0.8)
            row_idxs, col_idxs = np.where(attn_scores >= thr)
            words = [(attn_scores[i, j], (left_words[i], right_words[j])) for (i, j) in zip(row_idxs, col_idxs)]

            top_words = list(sorted(words, key=lambda x: x[0], reverse=True))
            if len(top_words) > int(attn_scores.size * 0.2):
                new_topk = int(attn_scores.size * 0.2)
                top_words = top_words[:new_topk]
            else:
                new_topk = len(top_words)

            if layer not in attn_data:
                attn_data[layer] = {record_id: top_words}
            else:
                attn_data[layer][record_id] = top_words

    return attn_data


def get_attn_to_similarity(conf, sampler_conf, fine_tune, attn_params, sim_type, sim_metric, sim_thrs, sim_op_eq,
                           sem_emb_model=None):
    attn_to_sim = []
    for uc in conf['use_case']:
        print("\n\n", uc)
        uc_conf = conf.copy()
        uc_conf['use_case'] = uc

        # Get data
        encoded_dataset = get_use_case_entity_pairs(conf=uc_conf, sampler_conf=sampler_conf)

        # Get similar words
        pair_of_entities = [(row[0], row[1]) for row in encoded_dataset]
        top_word_pairs_by_similarity = get_similar_word_pairs(pair_of_entities=pair_of_entities, sim_type=sim_type,
                                                              metric=sim_metric, thrs=sim_thrs, op_eq=sim_op_eq,
                                                              sem_emb_model=sem_emb_model, word_min_len=None)

        # Get top word pairs by BERT attention weights
        uc_attn = load_saved_attn_data(uc, conf, sampler_conf, fine_tune, attn_params, RESULTS_DIR)
        top_word_pairs_by_attn = get_top_word_pairs_by_attn(uc_attn)

        # Compute the percentage of word pairs shared between top word pairs by similarity and attention
        # loop over similarity thresholds
        skips = 0
        for thr in top_word_pairs_by_similarity:
            print(f"THR: {thr}")
            thr_sim_word_pairs = top_word_pairs_by_similarity[thr]
            sim_record_ids = thr_sim_word_pairs['idxs']
            sim_word_pairs = [[(p[0], p[1]) for p in p_list] for p_list in thr_sim_word_pairs['pairs']]

            # loop over BERT layers
            for layer in top_word_pairs_by_attn:
                print(f"\tLAYER: {layer}")
                layer_attn_word_pairs = top_word_pairs_by_attn[layer]

                overlaps = []
                weights = []
                for k in trange(len(sim_record_ids)):
                    record_id = sim_record_ids[k]
                    if record_id in layer_attn_word_pairs:
                        attn_word_pairs = [p[1] for p in layer_attn_word_pairs[record_id]]
                        weight = min(len(sim_word_pairs[k]), len(attn_word_pairs))
                        overlap = len(set(attn_word_pairs).intersection(set(sim_word_pairs[k]))) / weight
                        overlaps.append(overlap)
                        weights.append(weight)
                    else:
                        skips += 1

                layer_overlap = np.average(overlaps, weights=weights)

                attn_to_sim.append({
                    'use_case': uc,
                    'layer': layer,
                    'overlap': layer_overlap,
                    'num_records': len(overlaps)
                })

        print(f"SKIPS: {skips}")

    return attn_to_sim


def _res_to_df(res, res_type):
    res_tab = pd.DataFrame(res)
    res_tab['use_case'] = res_tab['use_case'].map(use_case_map)
    if res_type is False:
        method = 'pre-trained'
    else:
        method = 'fine-tuned'
    res_tab['method'] = method

    return res_tab


def load_results(res_type, tok, res_metric):

    # get pt results
    ft_model = False
    with open(os.path.join(RESULTS_DIR, f'attn_to_{res_type}_{tok}_{ft_model}_{res_metric}.pkl'), 'rb') as f:
        pt_res = pickle.load(f)
    pt_res = _res_to_df(pt_res, ft_model)

    # get ft results
    ft_model = True
    with open(os.path.join(RESULTS_DIR, f'attn_to_{res_type}_{tok}_{ft_model}_{res_metric}.pkl'), 'rb') as f:
        ft_res = pickle.load(f)
    ft_res = _res_to_df(ft_res, ft_model)

    res = pd.concat([pt_res, ft_res])

    return res


def plot_results(plot_data, plot_type, save_path=None):
    plt.figure(figsize=(6, 2.5))

    for method in ['pre-trained', 'fine-tuned']:
        method_data = plot_data[plot_data['method'] == method]
        if plot_type == 'by_use_case':
            method_pivot_data = method_data.pivot_table(index='layer', columns=['use_case'], values='overlap')
            filtered_use_cases = [x for x in use_case_map.values() if x in method_pivot_data.columns]
            method_pivot_data = method_pivot_data[filtered_use_cases]
        elif plot_type == 'by_layer':
            method_pivot_data = method_data.pivot_table(index='use_case', columns=['layer'], values='overlap')
            method_pivot_data.columns = range(1, len(method_pivot_data.columns) + 1)
        else:
            raise NotImplementedError()
        method_pivot_data = method_pivot_data * 100
        method_plot_stats = method_pivot_data.describe()
        medians = method_plot_stats.loc['50%', :].values
        percs_25 = method_plot_stats.loc['25%', :].values
        percs_75 = method_plot_stats.loc['75%', :].values

        plot_stats = {
            'x': range(len(method_pivot_data.columns)),
            'y': medians,
            'yerr': [medians - percs_25, percs_75 - medians],
        }

        if plot_type == 'by_use_case':
            plt.errorbar(**plot_stats, alpha=.75, fmt='.', capsize=5, label=method)
        elif plot_type == 'by_layer':
            plt.errorbar(**plot_stats, alpha=.75, fmt='.-', capsize=5, label=method)
        else:
            raise NotImplementedError()

    ax = plt.gca()
    ax.set_ylim(0, 100)
    plt.xticks(rotation=0)
    plt.legend(ncol=2, loc='best')
    plt.ylabel("Freq. (%)")
    if plot_type == 'by_use_case':
        plt.xlabel('Datasets')
        plt.xticks(range(len(method_pivot_data.columns)), method_pivot_data.columns)
    elif plot_type == 'by_layer':
        plt.xlabel('Layers')
        plt.xticks(range(len(method_pivot_data.columns)), list(method_pivot_data.columns))
    else:
        raise NotImplementedError()
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')

    plt.show()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='BERT attention to similar words')

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
    parser.add_argument('-return_offset', '--return_offset', default=False,
                        type=lambda x: bool(distutils.util.strtobool(x)),
                        help='boolean flag for extracting EM entry word indexes')

    # Parameters for data sampling
    parser.add_argument('-sample_size', '--sample_size', type=int,
                        help='size of the sample')
    parser.add_argument('-sample_target_class', '--sample_target_class', default='both', choices=['both', 0, 1],
                        help='classes to sample: match, non-match or both')
    parser.add_argument('-sample_seeds', '--sample_seeds', nargs='+', default=[42, 42],
                        help='seeds for each class sample. <seed non match> <seed match>')

    # Parameters for attention computation
    parser.add_argument('-attn_extractor', '--attn_extractor', default='word_extractor',
                        choices=['attr_extractor', 'word_extractor', 'token_extractor'],
                        help='type of attention to extract: 1) "attr_extractor": the attention weights are aggregated \
                            by attributes, 2) "word_extractor": the attention weights are aggregated by words, \
                            3) "token_extractor": the original attention weights are retrieved')
    parser.add_argument('-special_tokens', '--special_tokens', default=True,
                        type=lambda x: bool(distutils.util.strtobool(x)),
                        help='consider or ignore special tokens (e.g., [SEP], [CLS])')
    parser.add_argument('-agg_metric', '--agg_metric', default='mean', choices=['mean', 'max'],
                        help='method for aggregating the attention weights')
    parser.add_argument('-ft', '--fine_tune', default=False,
                        type=lambda x: bool(distutils.util.strtobool(x)),
                        help='boolean flag for selecting fine-tuned or pre-trained model')

    # Parameters for attention-to-similarity
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

    sampler_conf = {
        'size': args.sample_size,
        'target_class': args.sample_target_class,
        'seeds': args.sample_seeds,
    }

    fine_tune = args.fine_tune

    assert args.attn_extractor == 'word_extractor'
    assert args.special_tokens is True
    assert args.agg_metric == 'mean'

    attn_params = {
        'attn_extractor': args.attn_extractor,
        'attn_extr_params': {'special_tokens': args.special_tokens, 'agg_metric': args.agg_metric},
    }

    use_case_map = ConfCreator().use_case_map

    sim_metric = args.sim_metric
    sim_thrs = [float(x) for x in args.sim_thrs]
    sim_op_eq = args.sim_op_eq
    sem_embs = args.sem_embs
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
            attn_to_syntax = get_attn_to_similarity(conf, sampler_conf, fine_tune, attn_params, sim_type=sim_type,
                                                    sim_metric=sim_metric, sim_thrs=sim_thrs, sim_op_eq=sim_op_eq)
            out_fname = os.path.join(RESULTS_DIR, f'attn_to_{sim_type}_{conf["tok"]}_{fine_tune}_{sim_metric}.pkl')

            with open(out_fname, 'wb') as f:
                pickle.dump(attn_to_syntax, f)

        elif task == 'visualize':
            res = load_results(res_type=sim_type, tok=conf['tok'], res_metric=sim_metric)
            plot_results(res, plot_type='by_use_case',
                         save_path=os.path.join(RESULTS_DIR, f'PLOT_attn_to_{sim_type}_by_use_case.pdf'))
            plot_results(res, plot_type='by_layer',
                         save_path=os.path.join(RESULTS_DIR, f'PLOT_attn_to_{sim_type}_by_layer.pdf'))

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

            attn_to_semantic = get_attn_to_similarity(conf, sampler_conf, fine_tune, attn_params, sim_type=sim_type,
                                                      sim_metric=sim_metric, sim_thrs=sim_thrs, sim_op_eq=sim_op_eq,
                                                      sem_emb_model=sem_emb_model)
            out_fname = os.path.join(RESULTS_DIR, f'attn_to_{sim_type}_{conf["tok"]}_{fine_tune}_{sim_metric}.pkl')

            with open(out_fname, 'wb') as f:
                pickle.dump(attn_to_semantic, f)

        elif task == 'visualize':
            res = load_results(res_type=sim_type, tok=conf['tok'], res_metric=sim_metric)
            plot_results(res, plot_type='by_use_case',
                         save_path=os.path.join(RESULTS_DIR, f'PLOT_attn_to_{sim_type}_by_use_case.pdf'))
            plot_results(res, plot_type='by_layer',
                         save_path=os.path.join(RESULTS_DIR, f'PLOT_attn_to_{sim_type}_by_layer.pdf'))

        else:
            raise NotImplementedError()

    print(":)")
