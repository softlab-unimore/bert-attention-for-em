import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from utils.result_collector import TestResultCollector
from utils.test_utils import ConfCreator
from utils.plot import plot_left_to_right_heatmap
from core.attention.extractors import AttributeAttentionExtractor
import copy
from scipy.stats import entropy as entropy_fn


class GenericAttributeAttentionTest(object):
    """
    This class analyzes the attention paid by some model on matching attributes by
    examining its attention maps.
    It produces the following results:
    - lr_match_attr_attn_loc: mask that displays for each layer and head if the
    attention paid by the model on the corresponding attributes of the left and
    right entities is greater than the average attention between each pair of
    attributes
    - rl_match_attr_attn_loc: mask that displays for each layer and head if the
    attention paid by the model on the corresponding attributes of the right and
    left entities is greater than the average attention between each pair of
    attributes
    - match_attr_attn_loc: mask obtained by getting the maximum values between the
    previous two masks
    - match_attr_attn_over_mean: above-average attention paid by the model on each
    pair of attributes of the two entities
    """

    def __init__(self, permute: bool = False, model_attention_grid: tuple = (12, 12), ignore_special: bool = True):
        assert isinstance(permute, bool), "Wrong data type for parameter 'permute'."
        assert isinstance(model_attention_grid, tuple), "Wrong data type for parameter 'model_attention_grid'."
        assert len(model_attention_grid) == 2, "'model_attention_grid' has to specify two dimensions."
        assert model_attention_grid[0] > 0 and model_attention_grid[
            1] > 0, "Wrong value for parameter 'model_attention_grid'."
        assert isinstance(ignore_special, bool), "Wrong data type for parameter 'ignore_special'."

        self.permute = permute
        self.model_attention_grid = model_attention_grid
        self.ignore_special = ignore_special
        self.result_names = ['lr_match_attr_attn_loc', 'rl_match_attr_attn_loc',
                             'match_attr_attn_loc']

        self.property_mask_res = ['match_attr_attn_over_mean', 'avg_attr_attn']
        self.result_names += self.property_mask_res

        mask = np.zeros(model_attention_grid)
        attr_attn_3_last = mask.copy()
        attr_attn_3_last[-3:, :] = 1
        attr_attn_last_1 = mask.copy()
        attr_attn_last_1[-1, :] = 1
        attr_attn_last_2 = mask.copy()
        attr_attn_last_2[-2, :] = 1
        attr_attn_last_3 = mask.copy()
        attr_attn_last_3[-3, :] = 1
        self.cond_prop_mask_res = {'attr_attn_3_last': attr_attn_3_last,
                                   'attr_attn_last_1': attr_attn_last_1,
                                   'attr_attn_last_2': attr_attn_last_2,
                                   'attr_attn_last_3': attr_attn_last_3,
                                   'avg_attr_attn_3_last': attr_attn_3_last.copy(),
                                   'avg_attr_attn_last_1': attr_attn_last_1.copy(),
                                   'avg_attr_attn_last_2': attr_attn_last_2.copy(),
                                   'avg_attr_attn_last_3': attr_attn_last_3.copy(), }
        self.result_names += list(self.cond_prop_mask_res)

    def _test_attr_attention(self, attn_map: np.ndarray):

        assert isinstance(attn_map, np.ndarray), "Wrong data type for parameter 'attn_map'."

        res = {}
        n = attn_map.shape[0] // 2

        # extract the attention between corresponding left-to-right and
        # right-to-left attributes
        lr_match_attr_attn = np.array([])
        rl_match_attr_attn = np.array([])
        for idx in range(n):
            if self.permute:
                lr = attn_map[idx, n - idx].item()
                rl = attn_map[idx - n, idx].item()
            else:
                lr = attn_map[idx, n + idx].item()
                rl = attn_map[idx + n, idx].item()
            lr_match_attr_attn = np.append(lr_match_attr_attn, lr)
            rl_match_attr_attn = np.append(rl_match_attr_attn, rl)

        # check if these attention scores are greater than the average score
        # if all these attention scores are over the mean then output 1 in the mask
        m = attn_map.mean().item()
        res['lr_match_attr_attn_loc'] = int((lr_match_attr_attn > m).sum() >= n - 1)
        res['rl_match_attr_attn_loc'] = int((rl_match_attr_attn > m).sum() >= n - 1)

        # diff_attr_attn_lr = np.array([])
        # diff_attr_attn_rl = np.array([])
        # for idx in range(n):
        #   lr = np.concatenate(
        #       (attn_map[idx, n : n + idx], attn_map[idx, n + idx + 1:])
        #       ).mean().item()
        #   rl = np.concatenate(
        #       (attn_map[n : n + idx, idx], attn_map[n + idx + 1:, idx])
        #       ).mean().item()
        #   diff_attr_attn_lr = np.append(diff_attr_attn_lr, lr)
        #   diff_attr_attn_rl = np.append(diff_attr_attn_rl, rl)
        # res['diff_lr'] = diff_attr_attn_lr
        # res['diff_rl'] = diff_attr_attn_rl

        # save the mask that indicates which pair of attributes generates an
        # attention greater than the average score
        res['match_attr_attn_over_mean'] = attn_map > m
        res['attr_attn_3_last'] = attn_map > m
        res['attr_attn_last_1'] = attn_map > m
        res['attr_attn_last_2'] = attn_map > m
        res['attr_attn_last_3'] = attn_map > m
        res['avg_attr_attn'] = attn_map.copy()
        res['avg_attr_attn_3_last'] = attn_map.copy()
        res['avg_attr_attn_last_1'] = attn_map.copy()
        res['avg_attr_attn_last_2'] = attn_map.copy()
        res['avg_attr_attn_last_3'] = attn_map.copy()

        return res

    def test(self, left_entity: pd.Series, right_entity: pd.Series,
             attn_params: dict):

        assert isinstance(left_entity, pd.Series), "Wrong data type for parameter 'left_entity'."
        assert isinstance(right_entity, pd.Series), "Wrong data type for parameter 'right_entity'."
        assert isinstance(attn_params, dict), "Wrong data type for parameter 'attn_params'."
        assert 'attns' in attn_params, "Attention maps parameter non found."
        assert 'text_units' in attn_params, "Text units non found."

        attr_attns = attn_params['attns']
        text_units = attn_params['text_units']

        if attr_attns is None:
            return None

        n_layers = attr_attns.shape[0]
        n_heads = attr_attns.shape[1]
        n_attrs = attr_attns[0][0].shape[0]
        if self.ignore_special is True and text_units is not None and text_units[0] == '[CLS]':
            sep_idxs = np.where(np.array(text_units) == '[SEP]')[0]
            if len(sep_idxs) > 1:  # attr-pair mode
                n_attrs -= len(sep_idxs) * 2 + 1
            else:  # sent-pair mode
                n_attrs -= 3  # [CLS] + 2 x [SEP]
        assert self.model_attention_grid == (n_layers, n_heads)

        # initialize the result collector
        res_collector = TestResultCollector()
        for result_name in self.result_names:
            if result_name in self.property_mask_res or result_name in self.cond_prop_mask_res:
                res_collector.save_result(np.zeros((n_attrs, n_attrs)), result_name)
            else:
                res_collector.save_result(np.zeros((n_layers, n_heads)), result_name)

        # loop over the attention maps and analyze them
        for layer in range(n_layers):
            for head in range(n_heads):
                attn_map = attr_attns[layer][head]

                # (optional) remove special tokens
                if self.ignore_special is True and text_units is not None and text_units[0] == '[CLS]':
                    complete_text_units = text_units + text_units[1:]
                    sep_idxs = list(np.where(np.array(complete_text_units) == '[SEP]')[0])

                    valid_idxs = np.array(list(set(range(1, len(attn_map))).difference(sep_idxs)))
                    attn_map = attn_map[valid_idxs][:, valid_idxs]

                # analyze the current attention map
                test_res = self._test_attr_attention(attn_map)

                # save the results in the collector
                for result_name in test_res:
                    if result_name in self.property_mask_res:
                        res_collector.transform_result(result_name,
                                                       lambda x: x + test_res[result_name])
                    elif result_name in self.cond_prop_mask_res:
                        if self.cond_prop_mask_res[result_name][layer][head]:
                            res_collector.transform_result(result_name,
                                                           lambda x: x + test_res[result_name])
                    elif result_name in self.result_names:
                        res_collector.update_result_value(layer, head,
                                                          test_res[result_name],
                                                          result_name)
                    else:
                        ValueError("No result name found.")

        # update/add some results
        res_collector.combine_results('lr_match_attr_attn_loc',
                                      'rl_match_attr_attn_loc',
                                      lambda x, y: np.maximum(x, y),
                                      'match_attr_attn_loc')

        for result_name in self.property_mask_res:
            res_collector.transform_result(result_name,
                                           lambda x: x / (n_layers * n_heads))
        for result_name in self.cond_prop_mask_res:
            res_collector.transform_result(result_name,
                                           lambda x: x / (self.cond_prop_mask_res[result_name].sum()))

        return res_collector

    def _check_result_params(self, result: dict):
        assert isinstance(result, dict), "Wrong data type for parameter 'result'."

        params = self.result_names
        assert np.sum([param in result for param in params]) == len(params)

    def plot(self, res_collector: TestResultCollector, plot_params: list = None,
             out_dir=None, out_file_name_prefix=None, title_prefix=None, ax=None, labels=None,
             vmin=0, vmax=1, plot_type='simple'):

        assert isinstance(res_collector, TestResultCollector), "Wrong data type for parameter 'res_collector'."
        result = res_collector.get_results()
        self._check_result_params(result)
        if plot_params is not None:
            assert isinstance(plot_params, list)
            assert len(plot_params) > 0
            for plot_param in plot_params:
                assert plot_param in result
        assert isinstance(plot_type, str)
        assert plot_type in ['simple', 'advanced']

        for param, score in result.items():

            if plot_params is not None:
                if param not in plot_params:
                    continue

            if '_loc' not in param and plot_type == 'advanced':
                map = ConfCreator().use_case_map
                # assert out_file_name_prefix is not None
                if out_file_name_prefix is None:
                    out_file_name = None
                else:
                    out_file_name = f'{out_file_name_prefix}_{param}.png'
                plot_left_to_right_heatmap(score, vmin=vmin, vmax=vmax, title=map[title_prefix], is_annot=True,
                                           out_file_name=out_file_name)

            else:

                if ax is None:
                    fig, new_ax = plt.subplots(figsize=(10, 5))
                    title = param
                    if title_prefix is not None:
                        title = f'{title_prefix}_{title}'
                    fig.suptitle(title)

                else:
                    new_ax = ax

                assert new_ax is not None

                score = score.mean(axis=1).reshape((-1, 1))
                _ = sns.heatmap(score, annot=True, fmt='.1f', ax=new_ax, vmin=vmin, vmax=vmax)
                ylabel = 'layers'
                xlabel = 'heads'
                if param in self.property_mask_res:
                    xlabel, ylabel = 'attributes', 'attributes'

                new_ax.set_xlabel(xlabel)

                if labels:
                    new_ax.set_ylabel(ylabel)
                else:
                    new_ax.set_yticks([])

                if out_dir is not None:
                    if out_file_name_prefix is not None:
                        out_plot_file_name = '{}_{}.pdf'.format(out_file_name_prefix, param)
                    else:
                        out_plot_file_name = '{}.pdf'.format(param)

                    if ax is None:
                        plt.savefig(os.path.join(out_dir, out_plot_file_name), bbox_inches='tight')

                if ax is None:
                    plt.show()

    def plot_comparison(self, res_coll1: TestResultCollector, res_coll2: TestResultCollector,
                        cmp_res_coll: TestResultCollector, plot_params: list = None, out_dir=None,
                        out_file_name_prefix=None, title_prefix=None, labels=None):

        assert isinstance(res_coll1, TestResultCollector), "Wrong data type for parameter 'res_coll1'."
        res1 = res_coll1.get_results()
        self._check_result_params(res1)
        assert isinstance(res_coll2, TestResultCollector), "Wrong data type for parameter 'res_coll2'."
        res2 = res_coll2.get_results()
        self._check_result_params(res2)
        assert isinstance(cmp_res_coll, TestResultCollector), "Wrong data type for parameter 'cmp_res_coll'."
        cmp_res = cmp_res_coll.get_results()
        self._check_result_params(cmp_res)

        if plot_params is not None:
            assert isinstance(plot_params, list)
            assert len(plot_params) > 0
            for plot_param in plot_params:
                assert plot_param in res1
                assert plot_param in res2
                assert plot_param in cmp_res

        for param in res1:
            score1 = res1[param]
            score2 = res2[param]
            cmp_score = cmp_res[param]

            if plot_params is not None:
                if param not in plot_params:
                    continue

            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            title = param
            if title_prefix is not None:
                title = f'{title_prefix}_{title}'
            fig.suptitle(title)

            _ = sns.heatmap(score1, annot=True, fmt='.1f', ax=axes[0], vmin=0, vmax=1)
            _ = sns.heatmap(score2, annot=True, fmt='.1f', ax=axes[1], vmin=0, vmax=1)
            _ = sns.heatmap(cmp_score, annot=True, fmt='.1f', ax=axes[2], vmin=-0.5, vmax=0.5)

            # fig1, ax1 = plt.subplots(1, 1, figsize=(5, 4))
            # sns.heatmap(cmp_score, annot=True, fmt='.1f', ax=ax1, vmin=-0.5, vmax=0.5)
            # ax1.set_xlabel('heads')
            # ax1.set_ylabel('layers')
            # plt.savefig(f'{title}.pdf', bbox_inches='tight')

            ylabel = 'layers'
            xlabel = 'heads'
            if param in self.property_mask_res:
                xlabel, ylabel = 'attributes', 'attributes'

            for ax in axes:
                ax.set_xlabel(xlabel)

            if labels:
                for ax in axes:
                    ax.set_ylabel(ylabel)
            else:
                for ax in axes:
                    ax.set_yticks([])

            if out_dir is not None:
                if out_file_name_prefix is not None:
                    out_plot_file_name = '{}_{}.pdf'.format(out_file_name_prefix, param)
                else:
                    out_plot_file_name = '{}.pdf'.format(param)

                plt.savefig(os.path.join(out_dir, out_plot_file_name), bbox_inches='tight')

            plt.show()


class AttnPatternStats(object):
    def __init__(self, name: str, existence: np.ndarray, freq: np.ndarray = None, locs: np.ndarray = None):
        assert isinstance(name, str)
        assert isinstance(existence, np.ndarray)
        if freq is not None:
            assert isinstance(freq, np.ndarray)
        if locs is not None:
            assert isinstance(locs, np.ndarray)

        self.name = name
        self.existence = existence
        self.freq = freq
        self.locs = locs

    def get_data(self):
        return {
            'name': self.name,
            'existence': self.existence,
            'freq': self.freq,
            'locs': self.locs,
        }

    def _check_stats_format(self, stats):
        assert isinstance(stats, AttnPatternStats)
        stats_data = stats.get_data()
        assert self.name == stats_data['name']
        assert stats_data['freq'] is None if self.freq is None else stats_data['freq'] is not None
        assert stats_data['locs'] is None if self.locs is None else stats_data['locs'] is not None
        if self.locs is not None:
            assert self.locs.shape[-1] == len(stats_data['locs'])

        return stats_data

    def update(self, stats):
        stats_data = self._check_stats_format(stats)
        updated_stats = stats_data.copy()
        updated_stats['existence'] = stats_data['existence'] + self.existence
        if self.freq is not None:
            updated_stats['freq'] = stats_data['freq'] + self.freq
        if self.locs is not None:
            updated_stats['locs'] = stats_data['locs'] + self.locs

        return AttnPatternStats(**updated_stats)

    def concat(self, stats):
        stats_data = self._check_stats_format(stats)
        updated_stats = stats_data.copy()
        updated_stats['existence'] = np.concatenate((self.existence, stats_data['existence']))
        if self.freq is not None:
            updated_stats['freq'] = np.concatenate((self.freq, stats_data['freq']))
        if self.locs is not None:
            if self.locs.ndim == 1:
                locs1 = self.locs.reshape((1, -1))
            else:
                locs1 = self.locs
            if stats_data['locs'].ndim == 1:
                locs2 = stats_data['locs'].reshape((1, -1))
            else:
                locs2 = stats_data['locs']
            updated_stats['locs'] = np.concatenate((locs1, locs2))

        return AttnPatternStats(**updated_stats)

    def transform(self, transform_fn, by_dict=False):
        stats = self.get_data()

        if not by_dict:
            stats['existence'] = transform_fn(stats['existence'])
            if self.freq is not None:
                stats['freq'] = transform_fn(stats['freq'])
            if self.locs is not None:
                stats['locs'] = transform_fn(stats['locs'])
        else:
            for k, v in self.get_data().items():
                if k != 'name':
                    if v is not None:
                        stats[k] = transform_fn(k, v)

        return AttnPatternStats(**stats)

    def save_data(self, res_collector: TestResultCollector, prefix_save_name: str = ''):
        assert isinstance(res_collector, TestResultCollector)
        assert isinstance(prefix_save_name, str)
        if prefix_save_name != '':
            template_save_name = f'{prefix_save_name}_{self.name}'
        else:
            template_save_name = self.name

        res_collector.save_result(self.existence, f'{template_save_name}_existence')
        if self.freq is not None:
            res_collector.save_result(self.freq, f'{template_save_name}_freq')
        if self.locs is not None:
            res_collector.save_result(self.locs, f'{template_save_name}_locs')

        return res_collector


class AttnHeadPatternStats(object):
    def __init__(self, diag: AttnPatternStats, vertical: AttnPatternStats, diag_vertical: AttnPatternStats,
                 avg: np.ndarray, entropy: np.ndarray, match: AttnPatternStats = None):
        assert isinstance(diag, AttnPatternStats)
        assert isinstance(vertical, AttnPatternStats)
        assert isinstance(diag_vertical, AttnPatternStats)
        assert isinstance(avg, np.ndarray)
        assert isinstance(entropy, np.ndarray)
        if match is not None:
            assert isinstance(match, AttnPatternStats)

        self.diag = diag
        self.vertical = vertical
        self.diag_vertical = diag_vertical
        self.match = match
        self.avg = avg
        self.entropy = entropy

    def get_data(self):
        return {
            'diag': self.diag,
            'vertical': self.vertical,
            'diag-vertical': self.diag_vertical,
            'match': self.match,
            'avg': self.avg,
            'entropy': self.entropy,
        }

    def _check_stats_format(self, stats):
        assert isinstance(stats, AttnHeadPatternStats)
        stats_data = stats.get_data()
        assert isinstance(stats_data, dict)
        assert all([k in ['diag', 'vertical', 'diag-vertical', 'match', 'avg', 'entropy'] for k in stats_data])
        assert all([isinstance(v, (AttnPatternStats, np.ndarray)) for v in stats_data.values() if v is not None])
        assert stats_data['match'] is None if self.match is None else stats_data['match'] is not None

        return stats_data

    def update(self, stats):
        stats_data = self._check_stats_format(stats)
        updated_stats = stats_data.copy()
        updated_stats['diag'] = self.diag.update(stats_data['diag'])
        updated_stats['vertical'] = self.vertical.update(stats_data['vertical'])
        updated_stats['diag_vertical'] = self.diag_vertical.update(stats_data['diag-vertical'])
        updated_stats['avg'] = np.concatenate((self.avg, stats_data['avg']))
        updated_stats['entropy'] = np.concatenate((self.entropy, stats_data['entropy']))
        del updated_stats['diag-vertical']
        if self.match is not None:
            updated_stats['match'] = self.match.update(stats_data['match'])

        return AttnHeadPatternStats(**updated_stats)

    def concat(self, stats):

        def make_2d(arr1: np.ndarray, arr2: np.ndarray):
            if arr1.ndim == 1:
                arr1 = arr1.reshape((1, -1))
            if arr2.ndim == 1:
                arr2 = arr2.reshape((1, -1))

            return arr1, arr2

        stats_data = self._check_stats_format(stats)
        updated_stats = stats_data.copy()
        updated_stats['diag'] = self.diag.concat(stats_data['diag'])
        updated_stats['vertical'] = self.vertical.concat(stats_data['vertical'])
        updated_stats['diag_vertical'] = self.diag_vertical.concat(stats_data['diag-vertical'])
        avg1, avg2 = make_2d(self.avg, stats_data['avg'])
        updated_stats['avg'] = np.concatenate((avg1, avg2))
        entropy1, entropy2 = make_2d(self.entropy, stats_data['entropy'])
        updated_stats['entropy'] = np.concatenate((entropy1, entropy2))
        del updated_stats['diag-vertical']
        if self.match is not None:
            updated_stats['match'] = self.match.concat(stats_data['match'])

        return AttnHeadPatternStats(**updated_stats)

    def save_data(self, res_collector: TestResultCollector, prefix_save_name: str = '', ignore_metrics: list = None):
        assert isinstance(res_collector, TestResultCollector)
        assert isinstance(prefix_save_name, str)
        if ignore_metrics is not None:
            assert isinstance(ignore_metrics, list)
            assert len(ignore_metrics) > 0
            assert all([v in ['avg', 'entropy'] for v in ignore_metrics])
        if prefix_save_name != '':
            avg_save_name = f'{prefix_save_name}_avg'
        else:
            avg_save_name = 'avg'
        if prefix_save_name != '':
            entropy_save_name = f'{prefix_save_name}_entropy'
        else:
            entropy_save_name = 'entropy'

        self.diag.save_data(res_collector, prefix_save_name)
        self.vertical.save_data(res_collector, prefix_save_name)
        self.diag_vertical.save_data(res_collector, prefix_save_name)
        if (ignore_metrics is None) or (ignore_metrics is not None and 'avg' not in ignore_metrics):
            res_collector.save_result(self.avg, f'{avg_save_name}')
        if (ignore_metrics is None) or (ignore_metrics is not None and 'entropy' not in ignore_metrics):
            res_collector.save_result(self.entropy, f'{entropy_save_name}')
        if self.match:
            self.match.save_data(res_collector, prefix_save_name)

        return res_collector


class AttributeAttentionPatternFreqTest(object):

    def __init__(self, ignore_special: bool = True):
        assert isinstance(ignore_special, bool), "Wrong data type for parameter 'ignore_special'."

        self.ignore_special = ignore_special

    @staticmethod
    def test_attr_attention_patterns(attn_map: np.ndarray):

        assert isinstance(attn_map, np.ndarray), "Wrong data type for parameter 'attn_map'."
        assert attn_map.ndim == 2

        def check_diagonal_old(matrix):
            # get the average matrix value
            avg_val = matrix.mean().item()
            # get the diagonal
            diag = matrix.diagonal()
            # check if all the diagonal values are greater (or equal) than the average value
            return int(all(diag >= avg_val))

        def check_diagonal(matrix):
            n = len(matrix)
            extended_matrix = np.concatenate([matrix, matrix[:, :n - 1]], axis=1)
            extended_matrix = np.concatenate([extended_matrix, extended_matrix[:n - 1, :]])
            main_diagonal = extended_matrix.diagonal()[:n]
            avg_main_diag = main_diagonal.mean()
            found = True
            for i in range(1, n - 1):
                other_diag = extended_matrix.diagonal(i)[:n]
                if avg_main_diag < other_diag.mean():
                    found = False
                    break

                other_diag = extended_matrix.diagonal(-i)[:n]
                if avg_main_diag < other_diag.mean():
                    found = False
                    break

            return int(found)

        def check_vertical_old(matrix):
            vertical_freq = 0
            vertical_locs = []

            # get the average matrix value
            avg_val = matrix.mean().item()

            # loop over the columns and check if all the column values are greater (or equal) than the average value
            for col_idx in range(matrix.shape[1]):
                col = matrix[:, col_idx]
                if all(col >= avg_val) is True:
                    vertical_freq += 1
                    vertical_locs.append(1)
                else:
                    vertical_locs.append(0)

            return vertical_freq, np.array(vertical_locs)

        def check_vertical(matrix):
            vertical_freq = 0
            vertical_locs = []

            # get the average matrix value
            avg_val = matrix.mean().item()
            majority_num = len(matrix) // 2

            # loop over the columns and check if all the column values are greater (or equal) than the average value
            for col_idx in range(matrix.shape[1]):
                col = matrix[:, col_idx]
                if np.sum(col >= avg_val) >= majority_num:
                    vertical_freq += 1
                    vertical_locs.append(1)
                else:
                    vertical_locs.append(0)

            return vertical_freq, np.array(vertical_locs)

        def check_vertical_bis(matrix):
            avg_per_cols = matrix.mean(0)
            max_col = np.argmax(avg_per_cols)
            vertical_locs = np.zeros(len(avg_per_cols))
            vertical_locs[max_col] = 1

            return vertical_locs.sum(), vertical_locs

        def check_match_old(matrix):
            n = matrix.shape[0] // 2
            match_exist = 0
            m = matrix.mean().item()
            lr_match_attr_attn = np.array([])
            rl_match_attr_attn = np.array([])
            for idx in range(n):
                lr = matrix[idx, n + idx].item()
                rl = matrix[idx + n, idx].item()
                lr_match_attr_attn = np.append(lr_match_attr_attn, lr)
                rl_match_attr_attn = np.append(rl_match_attr_attn, rl)

            if (lr_match_attr_attn >= m).sum() >= n - 1 or (rl_match_attr_attn >= m).sum() >= n - 1:
                match_exist = 1

            return match_exist

        def check_match(matrix):
            n = matrix.shape[0] // 2
            rl_maa = check_diagonal(matrix[n:, :n])
            lr_maa = check_diagonal(matrix[:n, n:])

            return int(rl_maa and lr_maa)

        def check_diagonal_and_vertical(matrix):
            diag_exist = check_diagonal(matrix)
            v_freq, v_locs = check_vertical(matrix)
            v_exist = 1 if v_freq > 0 else 0

            diag_vert_exist = int(diag_exist and v_exist)
            diag_v_freq = v_freq if diag_vert_exist else 0
            diag_v_locs = v_locs if diag_vert_exist else np.zeros(len(v_locs))

            return diag_vert_exist, diag_v_freq, diag_v_locs

        def check_patterns(matrix, ignore_match=False):

            # check the existence of the diagonal pattern
            diag_existence = check_diagonal(matrix)
            diag_pattern = AttnPatternStats('diag', np.array([diag_existence]))

            # check the existence of the vertical pattern
            vertical_freq, vertical_locs = check_vertical(matrix)
            vertical_existence = 1 if vertical_freq > 0 else 0
            vertical_pattern = AttnPatternStats('vertical', np.array([vertical_existence]), np.array([vertical_freq]),
                                                vertical_locs)

            # check the existence of the diagonal+vertical pattern
            diag_vert_existence, diag_vert_freq, diag_vert_locs = check_diagonal_and_vertical(matrix)
            diag_vertical_pattern = AttnPatternStats('diag-vertical', np.array([diag_vert_existence]),
                                                     np.array([diag_vert_freq]), diag_vert_locs)

            # check the existence of the matching pattern
            match_pattern = None
            if not ignore_match:
                match_existence = check_match(matrix)
                match_pattern = AttnPatternStats('match', np.array([match_existence]))

            entropy = np.array([entropy_fn(matrix.reshape(-1))])
            avg = matrix.mean(keepdims=True)[0]

            return AttnHeadPatternStats(diag=diag_pattern, vertical=vertical_pattern, avg=avg, entropy=entropy,
                                        diag_vertical=diag_vertical_pattern, match=match_pattern)

        # check the existence of the patterns in all the attention map
        all_patterns = {'all': check_patterns(attn_map, ignore_match=False)}

        # check the existence of the patterns in single attention map quadrants
        quadrant_patterns = []
        n = attn_map.shape[0] // 2
        for i in range(2):
            for j in range(2):
                quadrant = attn_map[i * n: i * n + n, j * n: j * n + n]
                quad_patterns = check_patterns(quadrant, ignore_match=True)
                quadrant_patterns.append(quad_patterns)

        all_patterns.update({'quadrants': quadrant_patterns})

        return all_patterns

    def test(self, left_entity: pd.Series, right_entity: pd.Series, attn_params: dict):

        def aggregate_patterns(patterns):
            patterns_data = patterns.get_data()
            patterns_data['diag_vertical'] = patterns_data['diag-vertical']
            del patterns_data['diag-vertical']
            total_res = patterns_data.copy()
            avg_res = patterns_data.copy()
            std_res = patterns_data.copy()
            for k in patterns_data:
                pattern_or_metric = patterns_data[k]

                if pattern_or_metric is None:
                    continue

                if isinstance(pattern_or_metric, AttnPatternStats):  # pattern
                    total_res[k] = pattern_or_metric.transform(lambda x: x.sum(axis=0, keepdims=True))
                    avg_res[k] = pattern_or_metric.transform(lambda x: x.mean(axis=0, keepdims=True))
                    std_res[k] = pattern_or_metric.transform(lambda x: x.mean(axis=0, keepdims=True))

                else:  # metric
                    avg_res[k] = pattern_or_metric.mean(keepdims=True)
                    std_res[k] = pattern_or_metric.std(keepdims=True)

            return AttnHeadPatternStats(**total_res), AttnHeadPatternStats(**avg_res), AttnHeadPatternStats(**std_res)

        def normalize_pattern_freqs(patterns, n_attrs, input_normalize_fn=None):

            def normalize_fn(k, v, n_attrs):
                if k == 'existence':
                    return (v / 12) * 100

                elif k == 'freq':
                    return (v / (12 * n_attrs)) * 100

                elif k == 'locs':
                    return (v / 12) * 100

                else:
                    raise ValueError("Wrong normalization key.")

            patterns_data = patterns.get_data()
            patterns_data['diag_vertical'] = patterns_data['diag-vertical']
            del patterns_data['diag-vertical']

            norm_res = patterns_data.copy()
            for k in patterns_data:
                pattern_or_metric = patterns_data[k]

                if pattern_or_metric is None:
                    continue

                if isinstance(pattern_or_metric, AttnPatternStats):     # pattern
                    if input_normalize_fn is not None:
                        norm_res[k] = pattern_or_metric.transform(input_normalize_fn, by_dict=True)
                    else:
                        norm_res[k] = pattern_or_metric.transform(lambda x, y: normalize_fn(x, y, n_attrs), by_dict=True)
                else:   # metric
                    norm_res[k] = pattern_or_metric

            return AttnHeadPatternStats(**norm_res)

        def normalize_agg_pattern_freqs(tot, avg, std, n_attrs):

            def tot_normalize_fn(k, v, n_attrs):
                if k == 'existence':
                    return (v / (12 * 12)) * 100

                elif k == 'freq':
                    return (v / (12 * 12 * n_attrs)) * 100

                elif k == 'locs':
                    return (v / (12 * 12)) * 100

                else:
                    raise ValueError("Wrong normalization key.")

            norm_tot = normalize_pattern_freqs(tot, n_attrs, input_normalize_fn=lambda x, y: tot_normalize_fn(x, y, n_attrs))
            norm_avg = normalize_pattern_freqs(avg, n_attrs)
            norm_std = normalize_pattern_freqs(std, n_attrs)

            return norm_tot, norm_avg, norm_std

        patterns_by_layer = {}
        quadrant_patterns_by_layer = {}

        AttributeAttentionExtractor.check_attn_features((left_entity, right_entity, attn_params))
        attr_attns = attn_params['attns']
        text_units = attn_params['text_units']

        if attr_attns is None:
            return None

        n_layers = attr_attns.shape[0]
        n_heads = attr_attns.shape[1]
        n_attrs = attr_attns.shape[2]

        # loop over the attention maps and analyze them
        for layer in range(n_layers):
            for head in range(n_heads):
                attn_map = attr_attns[layer][head]

                # (optional) remove special tokens
                if self.ignore_special is True and text_units is not None and text_units[0] == '[CLS]':
                    complete_text_units = text_units + text_units[1:]
                    sep_idxs = list(np.where(np.array(complete_text_units) == '[SEP]')[0])

                    valid_idxs = np.array(list(set(range(1, len(attn_map))).difference(sep_idxs)))
                    attn_map = attn_map[valid_idxs][:, valid_idxs]
                    n_attrs = attn_map.shape[0]

                # analyze the current attention map
                attn_map_patterns = AttributeAttentionPatternFreqTest.test_attr_attention_patterns(attn_map)

                # save the results
                if layer not in patterns_by_layer:
                    patterns_by_layer[layer] = attn_map_patterns['all']
                    quadrant_patterns_by_layer[layer] = attn_map_patterns['quadrants']
                else:
                    patterns_by_layer[layer] = patterns_by_layer[layer].update(attn_map_patterns['all'])
                    quadrant_patterns_by_layer[layer] = [
                        quadrant_patterns_by_layer[layer][i].update(q_pattern) for i, q_pattern in
                        enumerate(attn_map_patterns['quadrants'])]

        layer_patterns = patterns_by_layer[0]
        for i in range(1, len(patterns_by_layer)):
            layer_patterns = layer_patterns.concat(patterns_by_layer[i])
        quadrant_layer_patterns = []
        for i in range(4):
            q = quadrant_patterns_by_layer[0][i]
            for l in range(1, len(quadrant_patterns_by_layer)):
                q = q.concat(quadrant_patterns_by_layer[l][i])
            quadrant_layer_patterns.append(q)

        # normalize the pattern frequencies
        norm_layer_patterns = normalize_pattern_freqs(layer_patterns, n_attrs=n_attrs)
        norm_quadrant_layer_patterns = [normalize_pattern_freqs(qp, n_attrs=n_attrs//2) for qp in quadrant_layer_patterns]

        # compute avg and total pattern frequencies by aggregating the layer results
        tot_res, avg_res, std_res = aggregate_patterns(layer_patterns)
        agg_quad_res = [aggregate_patterns(qp) for qp in quadrant_layer_patterns]

        # normalize the pattern frequencies also for the aggregated data
        norm_tot_res, norm_avg_res, norm_std_res = normalize_agg_pattern_freqs(tot_res, avg_res, std_res, n_attrs=n_attrs)
        norm_agg_quad_res = [normalize_agg_pattern_freqs(qp[0], qp[1], qp[2], n_attrs=n_attrs//2) for qp in agg_quad_res]

        # save the results
        res_collector = TestResultCollector()
        res_collector = norm_layer_patterns.save_data(res_collector, 'layers_all')
        for i, q in enumerate(norm_quadrant_layer_patterns):
            res_collector = q.save_data(res_collector, f'layers_q{i}')
        res_collector = norm_tot_res.save_data(res_collector, 'tot_all', ignore_metrics=['avg', 'entropy'])
        res_collector = norm_avg_res.save_data(res_collector, 'avg_all')
        res_collector = norm_std_res.save_data(res_collector, 'std_all')
        for i, agg_q in enumerate(norm_agg_quad_res):
            q_tot = agg_q[0]
            q_avg = agg_q[1]
            q_std = agg_q[2]
            res_collector = q_tot.save_data(res_collector, f'tot_q{i}', ignore_metrics=['avg', 'entropy'])
            res_collector = q_avg.save_data(res_collector, f'avg_q{i}')
            res_collector = q_std.save_data(res_collector, f'std_q{i}')

        for k, v in res_collector.get_results().items():
            assert not (v > 100).any()

        return res_collector
