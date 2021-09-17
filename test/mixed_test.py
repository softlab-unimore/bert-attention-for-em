import os
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, kendalltau
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle

from test.gradient_test import load_saved_grads_data
from test.attention.attention_test import load_saved_attn_data
from core.attention.extractors import WordAttentionExtractor
from core.explanation.gradient.extractors import EntityGradientExtractor
from utils.result_collector import BinaryClassificationResultsAggregator
from utils.test_utils import ConfCreator


PROJECT_DIR = Path(__file__).parent.parent
RESULTS_DIR = os.path.join(PROJECT_DIR, 'results', 'mixed')
GRAD_RESULTS_DIR = os.path.join(PROJECT_DIR, 'results', 'gradient_analysis')
ATTN_RESULTS_DIR = os.path.join(PROJECT_DIR, 'results', 'attention')


def get_gradient_vs_attention_correlation(gradients: list, attns: list, target_layer: int = None,
                                          target_categories: list = None):
    EntityGradientExtractor.check_extracted_grad(gradients)
    WordAttentionExtractor.check_batch_attn_features(attns)
    assert len(gradients) == len(attns)
    if target_layer is not None:
        assert isinstance(target_layer, int)
        assert target_layer in list(range(0, 12))

    # summarize the attention map in order to obtain an attention score for each word
    attn_features = []
    for j, attn_item in enumerate(attns):
        # get the attention maps (layers, heads, sequence_length, sequence_length)
        attn_maps = attn_item[2]['attns']

        if attn_maps is None:
            cls_attns = []
        else:
            if target_layer is None:    # compute the correlation with all the layers
                # get for each layer an average attention map by aggregating along the heads
                avg_attn_maps = np.mean(attn_maps, axis=1)
                # select only the row related to the CLS token (i.e., the first row of the attention maps)
                cls_attns = np.zeros((avg_attn_maps.shape[0], avg_attn_maps.shape[1]))
                for l in range(avg_attn_maps.shape[0]):
                    cls_attns[l, :] = avg_attn_maps[l][0]

            else:
                layer_attn_maps = attn_maps[target_layer]
                # get an average attention map by aggregating along the heads belonging to the target layer
                avg_layer_attn_map = np.mean(layer_attn_maps, axis=0)
                # select only the row related to the CLS token (i.e., the first row of the attention map)
                cls_attns = avg_layer_attn_map[0].reshape((1, -1))

        att_item_features = {
            'label': attn_item[2]['labels'].item(),
            'pred': attn_item[2]['preds'].item(),
            'data': cls_attns,
        }

        attn_features.append(att_item_features)

    grad_features = []
    for k, grad_item in enumerate(gradients):

        grad_item_features = {
            'label': grad_item['label'],
            'pred': grad_item['pred'],
            'data': grad_item['grad']['all_grad']['avg'],
        }

        grad_features.append(grad_item_features)

    assert len(attn_features) == len(grad_features)

    # group the attention and grad results by label/pred
    attn_aggregator = BinaryClassificationResultsAggregator('data', target_categories=target_categories)
    agg_attns, _, _, _ = attn_aggregator.add_batch_data(attn_features)
    grad_aggregator = BinaryClassificationResultsAggregator('data', target_categories=target_categories)
    agg_grads, _, _, _ = grad_aggregator.add_batch_data(grad_features)

    corr_by_cat = {}
    rank_by_cat = {}
    for cat in agg_attns:
        agg_attns_cat = agg_attns[cat]
        agg_grad_cat = agg_grads[cat]

        if agg_attns_cat is None or agg_grad_cat is None:
            continue

        corrs = np.zeros((len(agg_attns_cat), agg_attns_cat[0].shape[0]))
        ranks = np.zeros((len(agg_attns_cat), agg_attns_cat[0].shape[0]))

        for idx in tqdm(range(len(agg_attns_cat))):
            row_attns = agg_attns_cat[idx]
            row_grads = agg_grad_cat[idx]
            if len(row_attns) == 0:
                print("Skip truncated row.")
                continue
            assert row_attns.shape[1] == len(row_grads)
            corrs[idx, :] = [pearsonr(row_attns[i], row_grads)[0] for i in range(row_attns.shape[0])]
            ranks[idx, :] = [kendalltau(row_attns[i], row_grads)[0] for i in range(row_attns.shape[0])]

        avg_corrs = corrs.mean(axis=0)
        std_corrs = corrs.std(axis=0)
        avg_ranks = ranks.mean(axis=0)
        std_ranks = ranks.std(axis=0)

        corr_by_cat[cat] = {'avg': avg_corrs, 'std': std_corrs, 'raw': corrs}
        rank_by_cat[cat] = {'avg': avg_ranks, 'std': std_ranks, 'raw': ranks}

    return corr_by_cat, rank_by_cat


def compute_benchmark_correlations(use_cases: list, conf: dict, sampler_conf: dict, fine_tune: str, grad_conf: dict,
                                   attn_params: dict, grad_res_dir: str, attn_res_dir: str, target_layer: int = None,
                                   target_categories: list = None, save_path: str = None):
    avg_corr_results = {}
    std_corr_results = {}
    raw_corr_results = {}
    avg_rank_results = {}
    std_rank_results = {}
    raw_rank_results = {}
    use_case_map = ConfCreator().use_case_map
    for use_case in use_cases:
        print(use_case)
        uc_grad = load_saved_grads_data(use_case, conf, sampler_conf, fine_tune, grad_conf, grad_res_dir)
        uc_attn = load_saved_attn_data(use_case, conf, sampler_conf, fine_tune, attn_params, attn_res_dir)

        corr_res, rank_res = get_gradient_vs_attention_correlation(uc_grad, uc_attn, target_layer=target_layer,
                                                                   target_categories=target_categories)

        for cat in corr_res:
            avg_corr_res = corr_res[cat]['avg']
            std_corr_res = corr_res[cat]['std']
            raw_corr_res = corr_res[cat]['raw']
            avg_rank_res = rank_res[cat]['avg']
            std_rank_res = rank_res[cat]['std']
            raw_rank_res = rank_res[cat]['raw']

            if cat not in avg_corr_results:
                avg_corr_results[cat] = [avg_corr_res]
                std_corr_results[cat] = [std_corr_res]
                raw_corr_results[cat] = {use_case_map[use_case]: raw_corr_res}
                avg_rank_results[cat] = [avg_rank_res]
                std_rank_results[cat] = [std_rank_res]
                raw_rank_results[cat] = {use_case_map[use_case]: raw_rank_res}
            else:
                avg_corr_results[cat].append(avg_corr_res)
                std_corr_results[cat].append(std_corr_res)
                raw_corr_results[cat][use_case_map[use_case]] = raw_corr_res
                avg_rank_results[cat].append(avg_rank_res)
                std_rank_results[cat].append(std_rank_res)
                raw_rank_results[cat][use_case_map[use_case]] = raw_rank_res

    layers = [target_layer] if target_layer is not None else range(1, 13)
    new_use_cases = [use_case_map[uc] for uc in use_cases]
    for cat in avg_corr_results:

        avg_corr_results[cat] = pd.DataFrame(avg_corr_results[cat], index=new_use_cases, columns=layers)
        std_corr_results[cat] = pd.DataFrame(std_corr_results[cat], index=new_use_cases, columns=layers)
        for use_case in raw_corr_results[cat]:
            raw_corr_results[cat][use_case] = pd.DataFrame(raw_corr_results[cat][use_case], columns=layers)
        avg_rank_results[cat] = pd.DataFrame(avg_rank_results[cat], index=new_use_cases, columns=layers)
        std_rank_results[cat] = pd.DataFrame(std_rank_results[cat], index=new_use_cases, columns=layers)
        for use_case in raw_rank_results[cat]:
            raw_rank_results[cat][use_case] = pd.DataFrame(raw_rank_results[cat][use_case], columns=layers)

    out_results = {
        'avg_corr': avg_corr_results,
        'std_corr': std_corr_results,
        'raw_corr': raw_corr_results,
        'avg_rank': avg_rank_results,
        'std_rank': std_rank_results,
        'raw_rank': raw_rank_results,
    }

    # save
    if save_path:
        with open(save_path, 'wb') as f:
            pickle.dump(out_results, f)

    return out_results


def plot_corr_distribution(correlation_results: dict, ylabel: str, save_path: str = None):
    assert isinstance(correlation_results, dict)
    assert isinstance(ylabel, str)
    if save_path is not None:
        assert isinstance(save_path, str)
    xlabel = 'Layers'

    for cat in correlation_results:
        cat_corr_res = correlation_results[cat]

        first_use_case = list(cat_corr_res.values())[0]
        nlayers = len(first_use_case.columns)
        if nlayers == 1:
            max_len = 0
            for uc in cat_corr_res:
                if len(cat_corr_res[uc]) > max_len:
                    max_len = len(cat_corr_res[uc])
            concat_res = np.zeros((max_len, len(cat_corr_res)))
            concat_res[:] = np.nan
            for j, uc in enumerate(cat_corr_res):
                uc_res = cat_corr_res[uc].values.reshape(-1)
                concat_res[:len(uc_res), j] = uc_res
            cat_corr_res = {'': pd.DataFrame(concat_res, columns=cat_corr_res.keys())}
            xlabel = 'Datasets'

        ncols = 4
        nrows = 3
        figsize = (20, 10)
        if len(cat_corr_res) == 1:
            ncols = 1
            nrows = 1
            figsize = (12, 3)
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, sharey=True)
        if len(cat_corr_res) > 1:
            axes = axes.flat
        for idx, use_case in enumerate(cat_corr_res):

            if len(cat_corr_res) > 1:
                ax = axes[idx]
            else:
                ax = axes

            use_case_corr = cat_corr_res[use_case]
            if len(cat_corr_res) > 1:
                use_case_corr.plot.box(ax=ax, legend=False, rot=0)
            else:
                use_case_corr.plot.box(ax=ax, legend=False, rot=0)

            ax.set_title(use_case, fontsize=18)
            if idx % ncols == 0:
                if ylabel is not None:
                    ax.set_ylabel(ylabel, fontsize=20)
            ax.set_xlabel(xlabel, fontsize=20)
            ax.xaxis.set_tick_params(labelsize=18)
            ax.yaxis.set_tick_params(labelsize=20)
            ax.set_ylim(-1, 1)
            ax.set_yticks(np.arange(-1, 1.1, 0.5))

        # if len(cat_corr_res) > 1:
        #     handles, labels = ax.get_legend_handles_labels()
        #     fig.legend(handles, labels, bbox_to_anchor=(0.74, 0.05), ncol=4, fontsize=20)
        plt.subplots_adjust(wspace=0.01, hspace=0.6)

        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        # plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    use_cases = ["Structured_Fodors-Zagats", "Structured_DBLP-GoogleScholar", "Structured_DBLP-ACM",
                 "Structured_Amazon-Google", "Structured_Walmart-Amazon", "Structured_Beer",
                 "Structured_iTunes-Amazon", "Textual_Abt-Buy", "Dirty_iTunes-Amazon", "Dirty_DBLP-ACM",
                 "Dirty_DBLP-GoogleScholar", "Dirty_Walmart-Amazon"]

    conf = {
        'data_type': 'train',  # 'train', 'test', 'valid'
        'model_name': 'bert-base-uncased',
        'tok': 'sent_pair',  # 'sent_pair', 'attr', 'attr_pair'
        'label_col': 'label',
        'left_prefix': 'left_',
        'right_prefix': 'right_',
        'max_len': 128,
        'permute': False,
        'verbose': False,
    }

    sampler_conf = {
        'size': None,
        'target_class': 'both',  # 'both', 0, 1
        'seeds': [42, 42],  # [42 -> class 0, 42 -> class 1]
    }

    fine_tune = 'simple'  # None, 'simple', 'advanced'

    grad_conf = {
        'text_unit': 'words',
        'special_tokens': True,
        'agg': None,  # 'mean'
        'agg_target_cat': ['all', 'all_pos', 'all_neg', 'all_pred_pos', 'all_pred_neg', 'tp', 'tn', 'fp', 'fn']
    }

    attn_params = {
        'attn_extractor': 'word_extractor',  # 'attr_extractor', 'word_extractor', 'token_extractor'
        'attn_extr_params': {'special_tokens': True, 'agg_metric': 'mean'},
    }

    target_layer = 11
    target_categories = ['all']

    precomputed = True
    save_path = os.path.join(RESULTS_DIR, f'RES_attn_vs_grad_corr_l={target_layer}.pkl')

    if not precomputed:
        all_corr_results = compute_benchmark_correlations(use_cases, conf, sampler_conf, fine_tune, grad_conf,
                                                          attn_params, GRAD_RESULTS_DIR, ATTN_RESULTS_DIR, target_layer,
                                                          target_categories, save_path=save_path)
    else:
        # load precomputed results
        all_corr_results = pickle.load(open(save_path, "rb"))

    avg_corr_results = all_corr_results['avg_corr']
    std_corr_results = all_corr_results['std_corr']
    raw_corr_results = all_corr_results['raw_corr']
    avg_rank_results = all_corr_results['avg_rank']
    std_rank_results = all_corr_results['std_rank']
    raw_rank_results = all_corr_results['raw_rank']

    save_path_corr_plot = os.path.join(RESULTS_DIR, f'PLOT_DISTR_attn_vs_grad_corr_l={target_layer}.pdf')
    plot_corr_distribution(raw_corr_results, ylabel='Corr.', save_path=save_path_corr_plot)
    plot_corr_distribution(raw_rank_results, ylabel='Kendall rank corr.',
                           save_path=save_path_corr_plot.replace("_corr_", "_rank_"))
