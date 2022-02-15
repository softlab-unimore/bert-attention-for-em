import os
import numpy as np
from pathlib import Path
import pandas as pd
import itertools
import matplotlib.pyplot as plt
from utils.result_collector import TestResultCollector, BinaryClassificationResultsAggregator
from utils.test_utils import ConfCreator
from utils.attention_utils import get_analysis_results
from utils.data_collector import DM_USE_CASES
import argparse
import distutils.util

PROJECT_DIR = Path(__file__).parent.parent.parent
RESULTS_DIR = os.path.join(PROJECT_DIR, 'results', 'attention')


def extract_pattern_data_by_conf(data: dict, target_layer: str, target_head: str, target_pattern: str = None,
                                 target_pattern_metric: str = None, target_metric: str = None):
    assert isinstance(data, dict)
    assert isinstance(target_layer, str)
    assert target_layer in ['tot', 'layers', 'avg']
    assert isinstance(target_head, str)
    assert target_head in ['all', 'q0', 'q1', 'q2', 'q3']
    patterns = ['vertical', 'diag', 'match']
    if target_pattern is not None:
        assert isinstance(target_pattern, str)
        assert target_pattern in ['all'] + patterns
    if target_pattern_metric is not None:
        assert isinstance(target_pattern_metric, str)
        assert target_pattern_metric in ['freq', 'locs']
    if target_pattern is not None:
        assert target_pattern_metric is not None
    if target_pattern_metric is not None:
        assert target_pattern is not None
    if target_metric is not None:
        assert isinstance(target_metric, str)
        assert target_metric in ['avg', 'entropy']
    if target_pattern is not None:
        assert target_metric is None
    else:
        assert target_metric is not None
    if target_metric is not None:
        assert target_pattern is None
    else:
        assert target_pattern is not None

    if target_pattern is not None and target_pattern == 'match' and target_head != 'all':
        raise ValueError("Param combination not valid.")

    layer_comb = [target_layer]
    head_combs = [target_head] if target_head != 'q_all' else ['q0', 'q1', 'q2', 'q3']
    patterns = ['vertical', 'diag', 'match']
    if target_metric is None:
        pattern_comb = [target_pattern] if target_pattern != 'all' else patterns
        pattern_or_metric = [(p, target_pattern_metric) for p in pattern_comb]
    else:
        pattern_or_metric = [target_metric]

    conf_combs = (layer_comb, head_combs, pattern_or_metric)
    confs = list(itertools.product(*conf_combs))
    out_data = {}
    for data_conf in confs:
        if isinstance(data_conf[2], tuple):
            sub_key = f'{data_conf[2][0]}_{data_conf[2][1]}'
        else:
            sub_key = data_conf[2]
        data_key = f'{data_conf[0]}_{data_conf[1]}_{sub_key}'

        # if the freq pattern metric is selected, combine the values of the 'existence' and 'freq' parameters
        try:
            item = data[data_key]
        except KeyError:
            if 'freq' in data_key:
                item = data[data_key.replace('freq', 'existence')]
            else:
                raise KeyError(f"Data key {data_key} not found.")

        # convert single-value array into a scalar value
        if len(item.reshape(-1)) == 1:
            item = item.reshape(-1)[0]

        if 'layers' in data_key:
            if 'entropy' not in data_key and not 'avg' in data_key:
                item = {i: item[i] for i in range(len(item))}

        out_data[data_key] = item

    first_out_val = list(out_data.values())[0]
    if isinstance(first_out_val, dict):
        out_data = pd.DataFrame.from_dict(out_data)
        if target_pattern == 'all':
            out_data.columns = patterns
        else:
            out_data.columns = [target_pattern]

    elif isinstance(first_out_val, (int, float)):
        out_data = pd.DataFrame([out_data])
        if target_pattern == 'all':
            out_data.columns = patterns
        else:
            out_data.columns = [target_pattern]

    elif isinstance(first_out_val, np.ndarray):
        assert len(out_data) == 1
        out_data = pd.DataFrame(first_out_val)

    else:
        raise NotImplementedError()

    return out_data


def plot_pattern_freq(stats, save_path: str = None):
    ncols = 6
    nrows = 2
    if len(stats) == 1:
        ncols = 1
        nrows = 1
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(22, 5), sharey=True)
    if len(stats) > 1:
        axes = axes.flat

    max_val = 0
    for uc in stats:
        uc_stats = stats[uc]
        if uc_stats.values.max() > max_val:
            max_val = uc_stats.values.max()

    for idx, uc in enumerate(stats):
        use_case_stats = stats[uc]

        if len(stats) > 1:
            ax = axes[idx]
            use_case_stats.plot(kind='bar', ax=ax, legend=False, rot=0)
        else:
            ax = axes
            use_case_stats.plot(kind='bar', ax=ax, legend=True, rot=0)

        ax.set_title(uc, fontsize=18)
        if idx % ncols == 0:
            ax.set_ylabel("Freq. (%)", fontsize=20)
        # ax.set_xlabel("", fontsize=20)
        ax.xaxis.set_tick_params(labelsize=18)
        ax.yaxis.set_tick_params(labelsize=20)
        ax.set_yticks(np.arange(0, max_val, 15))

    if len(stats) > 1:
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, bbox_to_anchor=(0.72, 0.05), ncol=4, fontsize=20)
    plt.subplots_adjust(wspace=0.01, hspace=0.4)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')

    # plt.tight_layout()
    plt.show()


def plot_single_pattern_freq_by_layer(stats, small_plot=False, save_path: str = None):
    ncols = 4
    nrows = 3
    figsize = (18, 7)
    legend_loc = (0.76, 0.06)
    legend_items = len(list(stats.values())[0].columns)

    if small_plot:
        ncols = 4
        nrows = 1
        figsize = (18, 2.5)
    if legend_items < 4:
        legend_loc = (0.645, 0.05)

    if len(stats) == 1:
        ncols = 1
        nrows = 1
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, sharey=True)
    if len(stats) > 1:
        axes = axes.flat

    max_val = 0
    for uc in stats:
        uc_stats = stats[uc]
        if uc_stats.values.max() > max_val:
            max_val = uc_stats.values.max()

    for idx, uc in enumerate(stats):
        uc_stats = stats[uc]

        if len(stats) > 1:
            ax = axes[idx]
            legend = False
        else:
            ax = axes
            legend = True

        uc_stats.plot(kind='line', ax=ax, legend=legend, rot=0, marker='o')

        ax.set_title(uc, fontsize=18)
        if idx % ncols == 0:
            ax.set_ylabel("Freq. (%)", fontsize=18)
        # ax.set_xlabel("", fontsize=20)
        ax.xaxis.set_tick_params(labelsize=15)
        ax.yaxis.set_tick_params(labelsize=18)
        ax.set_xticks(range(len(uc_stats)))
        ax.set_xticklabels(range(1, len(uc_stats) + 1))
        ax.set_yticks(np.arange(0, max_val, 15))

    if len(stats) > 1:
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, bbox_to_anchor=legend_loc, ncol=legend_items, fontsize=20)
    plt.subplots_adjust(wspace=0.03, hspace=0.5)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')

    # plt.tight_layout()
    plt.show()


def plot_pattern_freq_by_layer(stats, target_head, target_pattern, small_plot=False, save_path=None):
    if save_path is not None:
        assert isinstance(save_path, str)

    plot_data = {}
    for uc in stats:
        uc_plot_data = {}
        uc_stats = stats[uc]
        for method in uc_stats:
            method_collector = uc_stats[method]
            assert isinstance(method_collector, TestResultCollector)
            method_stats = method_collector.get_results()
            method_stats = extract_pattern_data_by_conf(data=method_stats, target_layer='layers',
                                                        target_head=target_head, target_pattern=target_pattern,
                                                        target_pattern_metric='freq', target_metric=None)

            uc_plot_data[method] = method_stats

        plot_data[uc] = uc_plot_data

    if target_pattern != 'all':
        new_plot_data = {}
        for uc, uc_plot_data in plot_data.items():
            uc_plot_data_concat = pd.concat(list(uc_plot_data.values()), axis=1)
            uc_plot_data_concat.columns = list(uc_plot_data.keys())
            new_plot_data[uc] = uc_plot_data_concat
        plot_data = new_plot_data.copy()

        plot_single_pattern_freq_by_layer(plot_data, small_plot=small_plot, save_path=save_path)

    else:
        raise NotImplementedError()


def plot_pattern_freq_stats(stats: dict, target_head: str, layer_agg: bool = False, target_pattern: str = None,
                            save_path: str = None):
    target_layer = 'tot' if layer_agg is False else 'avg'
    first_stats_item = list(stats.values())[0]
    num_methods = len(first_stats_item)

    plot_data = {}
    for uc in stats:
        uc_plot_data = {}
        uc_stats = stats[uc]
        for method in uc_stats:
            method_collector = uc_stats[method]
            assert isinstance(method_collector, TestResultCollector)
            method_stats = method_collector.get_results()
            method_stats = extract_pattern_data_by_conf(data=method_stats, target_layer=target_layer,
                                                        target_head=target_head, target_pattern=target_pattern,
                                                        target_pattern_metric='freq', target_metric=None)
            if num_methods == 1 and len(method_stats) == 1:
                method_stats = method_stats.T

            uc_plot_data[method] = method_stats

        plot_data[uc] = uc_plot_data

    if num_methods == 1:
        plot_data = {uc: list(plot_data[uc].values())[0] for uc in plot_data}
    else:
        new_plot_data = {}
        for uc, uc_plot_data in plot_data.items():
            uc_plot_data_concat = pd.concat(list(uc_plot_data.values())).reset_index(drop=True)
            uc_plot_data = uc_plot_data_concat.rename(index={i: key for i, key in enumerate(uc_plot_data)})
            uc_plot_data = uc_plot_data.T
            new_plot_data[uc] = uc_plot_data
        plot_data = new_plot_data.copy()

    plot_data = {uc: plot_data[uc].rename(index={'match': 'MAA'}) for uc in plot_data}
    plot_pattern_freq(plot_data, save_path=save_path)


def plot_entropy(stats, save_path=None):
    ncols = 4
    nrows = 3
    if len(stats) == 1:
        ncols = 1
        nrows = 1
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 10), sharey=True)
    if len(stats) > 1:
        axes = axes.flat

    for idx, uc in enumerate(stats):
        uc_stats = stats[uc]

        if len(stats) > 1:
            ax = axes[idx]
            legend = False
        else:
            ax = axes
            legend = True

        for method, method_res in uc_stats.items():
            method_res_stats = method_res.describe()
            medians = method_res_stats.loc['50%', :].values
            percs_25 = method_res_stats.loc['25%', :].values
            percs_75 = method_res_stats.loc['75%', :].values
            plot_data = {
                'x': range(len(method_res_stats.columns)),
                'y': medians,
                'yerr': [medians - percs_25, percs_75 - medians],
            }
            ax.errorbar(**plot_data, alpha=.75, fmt='--', capsize=3, capthick=1, label=method)

        ax.set_title(uc, fontsize=18)
        if idx % ncols == 0:
            ax.set_ylabel("Entropy", fontsize=20)
        # ax.set_xlabel("", fontsize=20)
        ax.xaxis.set_tick_params(labelsize=18)
        ax.yaxis.set_tick_params(labelsize=20)
        ax.set_xticks(range(len(method_res.columns)))
        ax.set_xticklabels(range(1, len(method_res.columns) + 1))
        # ax.set_yticks(np.arange(0, max_val, 5))

    if len(stats) > 1:
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, bbox_to_anchor=(0.74, 0.08), ncol=4, fontsize=20)
    plt.subplots_adjust(wspace=0.01, hspace=0.3)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')

    # plt.tight_layout()
    plt.show()


def plot_entropy_by_layer(stats, target_head='all', save_path=None):
    plot_data = {}
    for uc in stats:
        uc_plot_data = {}
        uc_stats = stats[uc]
        for method in uc_stats:
            method_collector = uc_stats[method]
            assert isinstance(method_collector, TestResultCollector)
            method_stats = method_collector.get_results()
            method_stats = extract_pattern_data_by_conf(data=method_stats, target_layer='layers',
                                                        target_head=target_head, target_pattern=None,
                                                        target_pattern_metric=None, target_metric='entropy')

            uc_plot_data[method] = method_stats.T

        plot_data[uc] = uc_plot_data

    plot_entropy(plot_data, save_path=save_path)


def plot_vertical_pattern_distribution(stats, save_path=None):
    ncols = 4
    nrows = 3
    if len(stats) == 1:
        ncols = 1
        nrows = 1
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 10), sharey=True)
    if len(stats) > 1:
        axes = axes.flat

    # max_val = 0
    # for uc in stats:
    #     uc_stats = stats[uc]
    #     for method_stats in uc_stats:
    #         if method_stats.values.max() > max_val:
    #             max_val = method_stats.values.max()

    for idx, uc in enumerate(stats):
        uc_stats = stats[uc]

        if len(stats) > 1:
            ax = axes[idx]
            legend = False
        else:
            ax = axes
            legend = True

        uc_stats.plot(kind='bar', ax=ax, legend=legend, rot=0)

        ax.set_title(uc, fontsize=18)
        if idx % ncols == 0:
            ax.set_ylabel("Freq. (%)", fontsize=20)
        # ax.set_xlabel("", fontsize=20)
        ax.xaxis.set_tick_params(labelsize=18)
        ax.yaxis.set_tick_params(labelsize=20)
        # ax.set_xticks(range(len(uc_stats)))
        # ax.set_xticklabels(range(1, len(uc_stats) + 1))
        # ax.set_yticks(np.arange(0, max_val, 5))

    if len(stats) > 1:
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, bbox_to_anchor=(0.74, 0.08), ncol=4, fontsize=20)
    plt.subplots_adjust(wspace=0.01, hspace=0.3)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')

    # plt.tight_layout()
    plt.show()


def plot_vertical_pattern_locations(stats, target_head='all', save_path=None):
    plot_data = {}
    for uc in stats:
        uc_plot_data = {}
        uc_stats = stats[uc]
        for method in uc_stats:
            method_collector = uc_stats[method]
            assert isinstance(method_collector, TestResultCollector)
            method_stats = method_collector.get_results()
            method_stats = extract_pattern_data_by_conf(data=method_stats, target_layer='tot',
                                                        target_head=target_head, target_pattern='vertical',
                                                        target_pattern_metric='locs', target_metric=None)
            method_stats = method_stats.T
            if target_head == 'all':
                num_attrs = len(method_stats) // 2
                attr_keys = ['l'] * num_attrs + ['r'] * num_attrs
                method_stats = method_stats.rename(
                    index={i: f'{attr_keys[i]}{(i % num_attrs) + 1}' for i in range(len(method_stats))})
            elif target_head in ['q0', 'q2']:
                attr_keys = ['l'] * len(method_stats)
                method_stats = method_stats.rename(
                    index={i: f'{attr_keys[i]}{i + 1}' for i in range(len(method_stats))})
            else:
                attr_keys = ['r'] * len(method_stats)
                method_stats = method_stats.rename(
                    index={i: f'{attr_keys[i]}{i + 1}' for i in range(len(method_stats))})

            uc_plot_data[method] = method_stats

        plot_data[uc] = uc_plot_data

    new_plot_data = {}
    for uc in plot_data:
        uc_plot_data = plot_data[uc]
        uc_plot_data_concat = pd.concat(list(uc_plot_data.values()), axis=1)
        uc_plot_data_concat.columns = list(uc_plot_data.keys())
        new_plot_data[uc] = uc_plot_data_concat

    plot_vertical_pattern_distribution(new_plot_data, save_path=save_path)


def plot_vertical_freq_by_quadrants(stats, save_path=None):
    ncols = 4
    nrows = 3
    if len(stats) == 1:
        ncols = 1
        nrows = 1
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 10), sharey=True)
    if len(stats) > 1:
        axes = axes.flat

    # max_val = 0
    # for uc in stats:
    #     uc_stats = stats[uc]
    #     for method_stats in uc_stats:
    #         if method_stats.values.max() > max_val:
    #             max_val = method_stats.values.max()

    for idx, uc in enumerate(stats):
        uc_stats = stats[uc]

        if len(stats) > 1:
            ax = axes[idx]
            legend = False
        else:
            ax = axes
            legend = True

        uc_stats.plot(kind='bar', ax=ax, legend=legend, rot=0)

        ax.set_title(uc, fontsize=18)
        if idx % ncols == 0:
            ax.set_ylabel("Freq. (%)", fontsize=20)
        # ax.set_xlabel("", fontsize=20)
        ax.xaxis.set_tick_params(labelsize=18)
        ax.yaxis.set_tick_params(labelsize=20)
        # ax.set_xticks(range(len(uc_stats)))
        # ax.set_xticklabels(range(1, len(uc_stats) + 1))
        # ax.set_yticks(np.arange(0, max_val, 5))

    if len(stats) > 1:
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, bbox_to_anchor=(0.74, 0.08), ncol=4, fontsize=20)
    plt.subplots_adjust(wspace=0.01, hspace=0.3)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')

    # plt.tight_layout()
    plt.show()


def plot_quadrant_vertical_freq(stats, save_path=None):
    plot_data = {}
    for uc in stats:
        uc_plot_data = {}
        uc_stats = stats[uc]
        for method in uc_stats:
            method_collector = uc_stats[method]
            assert isinstance(method_collector, TestResultCollector)
            method_stats = method_collector.get_results()
            target_heads = ['all', 'q0', 'q1', 'q2', 'q3']
            quadrant_freqs = []
            for target_head in target_heads:
                freq = extract_pattern_data_by_conf(data=method_stats, target_layer='tot', target_head=target_head,
                                                    target_pattern='vertical', target_pattern_metric='freq',
                                                    target_metric=None)
                quadrant_freqs.append(freq.values.reshape(-1)[0])

            same_entity_freq = np.mean([quadrant_freqs[1], quadrant_freqs[4]])
            cross_entity_freq = np.mean([quadrant_freqs[2], quadrant_freqs[3]])

            method_stats = pd.DataFrame([{'all': quadrant_freqs[0], 'same_entity': same_entity_freq,
                                          'cross_entity': cross_entity_freq}], index=[method])

            uc_plot_data[method] = method_stats

        plot_data[uc] = uc_plot_data

    new_plot_data = {}
    for uc in plot_data:
        uc_plot_data = plot_data[uc]
        uc_plot_data_concat = pd.concat(list(uc_plot_data.values()))
        if len(uc_plot_data_concat) > 1:
            uc_plot_data_concat = uc_plot_data_concat.T
        new_plot_data[uc] = uc_plot_data_concat

    plot_vertical_freq_by_quadrants(new_plot_data, save_path=save_path)


def plot_sub_experiment_results(results: dict, sub_experiment: str, small_plot: bool = False, save_path: str = None):
    assert isinstance(results, dict)
    assert isinstance(sub_experiment, str)
    assert isinstance(small_plot, bool)
    if save_path is not None:
        assert isinstance(save_path, str)

    if sub_experiment == 'all_freq':
        plot_pattern_freq_stats(results, target_head='all', layer_agg=False, target_pattern='all',
                                save_path=save_path)

    elif sub_experiment == 'match_freq_by_layer':
        plot_pattern_freq_by_layer(results, target_head='all', target_pattern='match', small_plot=small_plot,
                                   save_path=save_path)

    elif sub_experiment == 'entropy_by_layer':
        plot_entropy_by_layer(results, target_head='all', save_path=save_path)

    elif sub_experiment == 'vertical_loc':
        plot_vertical_pattern_locations(results, target_head='all', save_path=save_path)

    elif sub_experiment == 'quadrant_vertical_freq':
        plot_quadrant_vertical_freq(results, save_path=save_path)

    else:
        raise ValueError("No sub experiment found.")


def get_conf_name(tune_or_pretrain, sent_or_attr_pair):
    tune_or_pretrain = 'pretrain' if tune_or_pretrain is False else 'tune'
    sent_or_attr_pair = 'sent' if sent_or_attr_pair == 'sent_pair' else 'attr'
    return f'{tune_or_pretrain}_{sent_or_attr_pair}'


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Analysis of attention patterns')

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
    parser.add_argument('-attn_extractor', '--attn_extractor', required=True,
                        choices=['attr_extractor', 'word_extractor', 'token_extractor'],
                        help='type of attention to extract: 1) "attr_extractor": the attention weights are aggregated \
                        by attributes, 2) "word_extractor": the attention weights are aggregated by words, \
                        3) "token_extractor": the original attention weights are retrieved')
    parser.add_argument('-special_tokens', '--special_tokens', default=True,
                        type=lambda x: bool(distutils.util.strtobool(x)),
                        help='consider or ignore special tokens (e.g., [SEP], [CLS])')
    parser.add_argument('-agg_metric', '--agg_metric', required=True, choices=['mean', 'max'],
                        help='method for aggregating the attention weights')
    parser.add_argument('-ft', '--fine_tune', default=False,
                        type=lambda x: bool(distutils.util.strtobool(x)),
                        help='boolean flag for selecting fine-tuned or pre-trained model')

    # Parameters for attention analysis
    analysis_results = ['match_attr_attn_loc', 'match_attr_attn_over_mean',
                        'avg_attr_attn', 'attr_attn_3_last', 'attr_attn_last_1',
                        'attr_attn_last_2', 'attr_attn_last_3',
                        'avg_attr_attn_3_last', 'avg_attr_attn_last_1',
                        'avg_attr_attn_last_2', 'avg_attr_attn_last_3']
    available_categories = BinaryClassificationResultsAggregator.categories
    available_experiments = ['all_freq', 'match_freq_by_layer', 'entropy_by_layer', 'vertical_loc',
                             'quadrant_vertical_freq']
    parser.add_argument('-attn_tester', '--attn_tester', required=True, choices=['attr_tester', 'attr_pattern_tester'],
                        help='method for analysing the extracted attention weights')
    parser.add_argument('-attn_tester_ignore_special', '--attn_tester_ignore_special', default=True,
                        type=lambda x: bool(distutils.util.strtobool(x)),
                        help='whether the attention weights analyzer has to ignore special tokens')
    parser.add_argument('-experiment', '--experiment', required=True, choices=available_experiments,
                        help='the name of the analysis to perform')
    parser.add_argument('-analysis_type', '--analysis_type', required=True, choices=['simple', 'comparison'],
                        help='whether to compute attention weights analysis (i.e., the "simple" option) or compare \
                             previous analysis')
    parser.add_argument('-comparison_param', '--comparison_param', choices=['tune', 'tok', 'tune_tok'],
                        help='the dimension where to compare previous analysis')
    parser.add_argument('-data_categories', '--data_categories', default=['all'], nargs='+',
                        choices=available_categories,
                        help='the categories of records where to apply the attention analysis')
    parser.add_argument('-small_plot', '--small_plot', default=False,
                        type=lambda x: bool(distutils.util.strtobool(x)),
                        help='create a small plot')

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
        'size': args.sample_size,
        'max_len': args.max_len,
        'permute': args.permute,
        'verbose': args.verbose,
        'return_offset': args.return_offset,
        'fine_tune_method': args.fine_tune,
        'extractor': {
            'attn_extractor': args.attn_extractor,
            'attn_extr_params': {'special_tokens': args.special_tokens, 'agg_metric': args.agg_metric},
        },
        'tester': {
            'tester': args.attn_tester,
            'tester_params': {'ignore_special': args.attn_tester_ignore_special}
        },
    }

    extractor_name = conf['extractor']['attn_extractor']
    tester_name = conf['tester']['tester']
    agg_metric = conf['extractor']['attn_extr_params']['agg_metric']

    analysis_type = args.analysis_type
    comparison = args.comparison_param
    categories = args.data_categories
    conf_creator = ConfCreator()
    use_case_map = conf_creator.use_case_map
    sub_experiment = args.experiment
    small_plot = args.small_plot
    assert len(categories) == 1
    cat = categories[0]

    if analysis_type == 'simple':
        res, _ = get_analysis_results(conf, use_cases, RESULTS_DIR)
        res_name = get_conf_name(conf["fine_tune_method"], conf["tok"])
        res = {use_case_map[uc]: {res_name: res[uc][cat]} for uc in res}

        template_file_name = '{}_{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(conf['data_type'], conf["fine_tune_method"],
                                                                    conf["tok"], extractor_name, tester_name,
                                                                    conf['permute'], conf['size'], agg_metric,
                                                                    analysis_type, sub_experiment)

        save_path = os.path.join(RESULTS_DIR, f'PLOT_PATTERN_{template_file_name}.pdf')

        plot_sub_experiment_results(res, sub_experiment, small_plot=small_plot, save_path=save_path)

    elif analysis_type == 'comparison':

        if comparison == 'tune':
            new_conf = conf.copy()
            new_conf['fine_tune_method'] = False
            pretrain_res, _ = get_analysis_results(new_conf, use_cases, RESULTS_DIR)
            pretrain_res = {uc: pretrain_res[uc][cat] for uc in pretrain_res}

            new_conf['fine_tune_method'] = True
            tuned_res, _ = get_analysis_results(new_conf, use_cases, RESULTS_DIR)
            tuned_res = {uc: tuned_res[uc][cat] for uc in tuned_res}

            plot_res = {}
            for uc in pretrain_res:
                uc_pretrain_res = pretrain_res[uc]
                uc_tuned_res = tuned_res[uc]
                plot_res[use_case_map[uc]] = {'pre-trained': uc_pretrain_res, 'fine-tuned': uc_tuned_res}

            template_file_name = '{}_{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(conf['data_type'], conf["tok"],
                                                                        extractor_name, tester_name,
                                                                        conf['permute'], conf['size'], agg_metric,
                                                                        analysis_type, sub_experiment, comparison)

            save_path = os.path.join(RESULTS_DIR, f'PLOT_PATTERN_{template_file_name}.pdf')

            plot_sub_experiment_results(plot_res, sub_experiment, small_plot=small_plot, save_path=save_path)

        elif comparison == 'tok':
            new_conf = conf.copy()
            new_conf['tok'] = 'sent_pair'
            sent_res, _ = get_analysis_results(new_conf, use_cases, RESULTS_DIR)
            sent_res = {uc: sent_res[uc][cat] for uc in sent_res}

            new_conf['tok'] = 'attr_pair'
            attr_res, _ = get_analysis_results(new_conf, use_cases, RESULTS_DIR)
            attr_res = {uc: attr_res[uc][cat] for uc in attr_res}

            plot_res = {}
            for uc in sent_res:
                uc_sent_res = sent_res[uc]
                uc_attr_res = attr_res[uc]
                plot_res[use_case_map[uc]] = {'sent': uc_sent_res, 'attr': uc_attr_res}

            template_file_name = '{}_{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(conf['data_type'], conf["fine_tune_method"],
                                                                        extractor_name, tester_name,
                                                                        conf['permute'], conf['size'], agg_metric,
                                                                        analysis_type, sub_experiment, comparison)

            save_path = os.path.join(RESULTS_DIR, f'PLOT_PATTERN_{template_file_name}.pdf')

            plot_sub_experiment_results(plot_res, sub_experiment, small_plot=small_plot, save_path=save_path)

        elif comparison == 'tune_tok':

            template_file_name = '{}_{}_{}_{}_{}_{}_{}_{}'.format(conf['data_type'], extractor_name, tester_name,
                                                                  conf['permute'], conf['size'], agg_metric,
                                                                  analysis_type, comparison)

            new_conf = conf.copy()
            new_conf['fine_tune_method'] = False
            new_conf['tok'] = 'sent_pair'
            pretrain_sent_res, _ = get_analysis_results(new_conf, use_cases, RESULTS_DIR)
            pretrain_sent_res = {uc: pretrain_sent_res[uc][cat] for uc in pretrain_sent_res}

            new_conf['tok'] = 'attr_pair'
            pretrain_attr_res, _ = get_analysis_results(new_conf, use_cases, RESULTS_DIR)
            pretrain_attr_res = {uc: pretrain_attr_res[uc][cat] for uc in pretrain_attr_res}

            new_conf['fine_tune_method'] = True
            new_conf['tok'] = 'sent_pair'
            tuned_sent_res, _ = get_analysis_results(new_conf, use_cases, RESULTS_DIR)
            tuned_sent_res = {uc: tuned_sent_res[uc][cat] for uc in tuned_sent_res}

            new_conf['tok'] = 'attr_pair'
            tuned_attr_res, _ = get_analysis_results(new_conf, use_cases, RESULTS_DIR)
            tuned_attr_res = {uc: tuned_attr_res[uc][cat] for uc in tuned_attr_res}

            plot_res = {}
            for uc in pretrain_sent_res:
                uc_pt_sent_res = pretrain_sent_res[uc]
                uc_pt_attr_res = pretrain_attr_res[uc]
                uc_ft_sent_res = tuned_sent_res[uc]
                uc_ft_attr_res = tuned_attr_res[uc]

                plot_res[use_case_map[uc]] = {'pt_sent': uc_pt_sent_res, 'pt_attr': uc_pt_attr_res,
                                              'ft_sent': uc_ft_sent_res,
                                              'ft_attr': uc_ft_attr_res}

            template_file_name = '{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(conf['data_type'], extractor_name, tester_name,
                                                                     conf['permute'], conf['size'], agg_metric,
                                                                     analysis_type, sub_experiment, comparison)

            save_path = os.path.join(RESULTS_DIR, f'PLOT_PATTERN_{template_file_name}.pdf')

            plot_sub_experiment_results(plot_res, sub_experiment, small_plot=small_plot, save_path=save_path)
