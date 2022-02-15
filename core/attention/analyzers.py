import copy
import pickle
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from utils.result_collector import TestResultCollector
from core.attention.extractors import AttributeAttentionExtractor, WordAttentionExtractor, AttentionExtractor
import numpy as np
from utils.result_collector import BinaryClassificationResultsAggregator
from scipy.stats import entropy
import spacy
from spacy.tokenizer import Tokenizer
import re
import string
from utils.nlp import get_pos_tag, get_most_similar_words_from_sent_pair


class AttentionMapAnalyzer(object):
    """
    This class applies the analyzes implemented by appropriate tester classes to
    the attention maps extracted by an appropriate attention extractor class.
    It automatically categorizes the results of such analyzes into the following
    categories:
    - all: all the records of the dataset
    - true_match: ground truth match records
    - true_non_match: ground truth non-match records
    - pred_match: match records predicted by the model
    - pred_non_match: non-match records predicted by the model
    It accepts multiple classes as input for the application of different tests.
    Such classes have to implement the following interface:
    - test(left_entity, right_entity, attn_params): applies some tests on the
        attention maps (integrated with additional params) and returns the results
    """

    def __init__(self, attn_extractor, testers: list, **kwargs):

        assert isinstance(testers, list), "Wrong data type for parameter 'testers'."
        assert len(testers) > 0, "Empty tester list."

        self.attn_extractor = attn_extractor
        self.testers = testers
        self.pre_computed_attns = None
        if kwargs is not None:
            if 'pre_computed_attns' in kwargs and kwargs['pre_computed_attns'] is not False:
                pre_computed_attns = pickle.load(open(f"{kwargs['pre_computed_attns']}", "rb"))
                attn_extractor.check_batch_attn_features(pre_computed_attns)
                print("Using pre-computed attentions.")
                self.pre_computed_attns = pre_computed_attns

        res = {
            'all': None,
            'true_match': None,
            'true_non_match': None,
            'pred_match': None,
            'pred_non_match': None,
            'tp': None,
            'tn': None,
            'fp': None,
            'fn': None
        }

        self.res_history = [copy.deepcopy(res) for _ in range(len(self.testers))]
        self.counts = {
            'all': 0,
            'true_match': 0,
            'true_non_match': 0,
            'pred_match': 0,
            'pred_non_match': 0,
            'tp': 0,
            'tn': 0,
            'fp': 0,
            'fn': 0
        }
        self.preds = []
        self.labels = []
        self.text_units = []

    def __len__(self):
        return len(self.attn_extractor)

    def _save_result(self, res_history: dict, result: (dict, TestResultCollector),
                     result_name: str):

        assert isinstance(res_history, dict), "Wrong data type for parameter 'res_history'."
        assert isinstance(result, (dict, TestResultCollector)), "Wrong data type for parameter 'result'."
        assert isinstance(result_name, str), "Wrong data type for parameter 'result_name'."
        assert result_name in res_history

        if res_history[result_name] is None:  # init the result collector

            res_history[result_name] = copy.deepcopy(result)

        else:  # cumulative sum of the results

            if isinstance(result, dict):

                for key in result:
                    assert key in res_history[result_name]
                    assert isinstance(res_history[result_name][key], TestResultCollector)
                    assert isinstance(result[key], TestResultCollector)

                    # res_history[result_name][key].add_collector(result[key])
                    res_history[result_name].transform_collector(result, transform_fn=lambda x, y: x + y)

            elif isinstance(result, TestResultCollector):

                assert isinstance(res_history[result_name], TestResultCollector)

                # res_history[result_name].add_collector(result)
                res_history[result_name].transform_collector(result, transform_fn=lambda x, y: x + y)

        self.counts[result_name] += 1

    def _save(self, results: list, label: int, pred: int, category=None):

        assert isinstance(results, list), "Wrong data type for parameter 'results'."
        assert len(results) == len(self.testers), "Length mismatch between 'results' and 'self.testers'."
        assert isinstance(label, int), "Wrong data type for parameter 'label'."
        if pred is not None:
            assert isinstance(pred, int), "Wrong data type for parameter 'pred'."

        for tester_idx in range(len(self.testers)):
            tester_res_history = self.res_history[tester_idx]
            tester_res = results[tester_idx]

            self._save_result(tester_res_history, tester_res, 'all')

            if label == 1:  # match row

                self._save_result(tester_res_history, tester_res, 'true_match')

                if pred is not None:
                    if pred == 1:  # true positive
                        self._save_result(tester_res_history, tester_res, 'pred_match')
                        self._save_result(tester_res_history, tester_res, 'tp')
                    else:  # false negative
                        self._save_result(tester_res_history, tester_res, 'pred_non_match')
                        self._save_result(tester_res_history, tester_res, 'fn')

            else:  # non-match row

                self._save_result(tester_res_history, tester_res, 'true_non_match')

                if pred is not None:
                    if pred == 1:  # false positive
                        self._save_result(tester_res_history, tester_res, 'pred_match')
                        self._save_result(tester_res_history, tester_res, 'fp')
                    else:  # true negative
                        self._save_result(tester_res_history, tester_res, 'pred_non_match')
                        self._save_result(tester_res_history, tester_res, 'tn')

            if category is not None:
                if category not in tester_res_history:
                    tester_res_history[category] = copy.deepcopy(tester_res)
                else:
                    self._save_result(tester_res_history, tester_res, category)

    def __getitem__(self, idx: int):

        if self.pre_computed_attns is not None:
            left_entity, right_entity, attn_params = self.pre_computed_attns[idx]

        else:
            left_entity, right_entity, attn_params = self.attn_extractor[idx]
            self.attn_extractor.check_attn_features((left_entity, right_entity, attn_params))

        label = attn_params['labels'].item()
        pred = None
        if attn_params['preds'] is not None:
            pred = attn_params['preds'].item()
        text_units = attn_params['text_units']
        category = None
        if 'category' in attn_params:
            category = attn_params['category']

        tester_results = []
        for tester_id, tester in enumerate(self.testers):
            result = tester.test(left_entity, right_entity, attn_params)

            if result is not None:
                assert isinstance(result, (dict, TestResultCollector))
                if isinstance(result, dict):
                    for key in result:
                        assert isinstance(result[key], TestResultCollector)

            tester_results.append(result)

        return tester_results, label, pred, category, text_units

    def analyze(self, idx: int):
        return self[idx]

    def get_labels_and_preds(self):
        return self.labels, self.preds

    def get_text_units(self):
        return self.text_units

    def analyze_all(self):

        # retrieve all the results
        for row_tester_results, row_label, row_pred, row_category, row_text_units in tqdm(self):
            assert len(row_tester_results) == len(self.testers)
            if row_tester_results[0] is not None:
                self._save(row_tester_results, row_label, row_pred, row_category)
                self.labels.append(row_label)
                self.preds.append(row_pred)
                self.text_units.append(row_text_units)

        assert len(self.res_history) == len(self.testers)

        # result post-processing
        # now the results are stored and categorized in the 'res_history' variable
        for tester_res in self.res_history:

            assert isinstance(tester_res, dict)

            for category in tester_res:
                tester_res_by_cat = tester_res[category]

                if tester_res_by_cat is not None:

                    assert isinstance(tester_res_by_cat, (dict, TestResultCollector))

                    if isinstance(tester_res_by_cat, dict):

                        for key in tester_res_by_cat:
                            assert isinstance(tester_res_by_cat[key], TestResultCollector)
                            assert len(tester_res_by_cat[key]) > 0

                            # normalization
                            tester_res_by_cat[key].transform_all(lambda x: x / self.counts[category])

                    elif isinstance(tester_res_by_cat, TestResultCollector):

                        assert len(tester_res_by_cat) > 0

                        # normalization
                        tester_res_by_cat.transform_all(lambda x: x / self.counts[category])

        return copy.deepcopy(self.res_history)


class AttrToClsAttentionAnalyzer(object):

    @staticmethod
    def group_or_aggregate(attn_results: list, target_categories: list = None, agg_metric: str = None):

        AttributeAttentionExtractor.check_batch_attn_features(attn_results)

        attrs = None
        records_cls_attn = []
        for attn_res in attn_results:
            attn_params = attn_res[2]

            if attn_params['attns'] is None:
                continue

            attn_text_units = attn_params['text_units']
            label = attn_params['labels'].item()
            pred = attn_params['preds'].item() if attn_params['preds'] is not None else None
            assert attn_text_units[0] == '[CLS]' and attn_text_units[-1] == '[SEP]'
            if attrs is None:
                attrs = [f'{lr}{attr}' for lr in ['l_', 'r_'] for attr in attn_text_units if
                         attr not in ['[CLS]', '[SEP]']]
            else:
                assert attrs == [f'{lr}{attr}' for lr in ['l_', 'r_'] for attr in attn_text_units if
                                 attr not in ['[CLS]', '[SEP]']]
            text_unit_idxs = [i for i, tu in enumerate(attn_text_units) if tu not in ['[CLS]', '[SEP]']]
            text_unit_idxs += [len(attn_text_units) - 1 + i for i in text_unit_idxs]

            # select only the last layer
            attns = attn_params['attns'][-1]
            # get an average attention map by aggregating along the heads belonging to the last layer
            attns = np.mean(attns, axis=0)
            # select only the row related to the CLS token (i.e., the first row of the attention map)
            attns = attns[0]
            # filter out the attention of other special tokens
            attns = attns[text_unit_idxs]

            record_cls_attn = {
                'label': label,
                'pred': pred,
                'attn': attns
            }
            records_cls_attn.append(record_cls_attn)

        aggregator = BinaryClassificationResultsAggregator('attn', target_categories=target_categories)
        grouped_cls_attn, _, _, _ = aggregator.add_batch_data(records_cls_attn)

        if agg_metric is not None:
            attr2cls_attn = aggregator.aggregate(agg_metric)
        else:
            attr2cls_attn = {}
            for cat in grouped_cls_attn:
                if grouped_cls_attn is not None:
                    attr2cls_attn[cat] = pd.DataFrame(grouped_cls_attn[cat], columns=attrs)

        return attr2cls_attn

    @staticmethod
    def analyze_multi_results(attr2cls_attn: dict, analysis_type: str):

        assert isinstance(attr2cls_attn, dict)
        assert isinstance(analysis_type, str)
        assert analysis_type in ['entropy']

        if analysis_type == 'entropy':

            def get_entropy(distribution):
                return entropy(distribution, base=2)

            out_data = []
            for use_case in attr2cls_attn:
                uc_attr2cls_attn = attr2cls_attn[use_case]
                AttrToClsAttentionAnalyzer.check_attr_to_cls_attn_results(uc_attr2cls_attn, agg=True)

                entropy_by_cat = {}
                for cat in uc_attr2cls_attn:
                    uc_cat_attn = uc_attr2cls_attn[cat]
                    entropy_by_cat[cat] = get_entropy(uc_cat_attn['mean'])

                out_data.append(entropy_by_cat)

            out_data = pd.DataFrame(out_data, index=attr2cls_attn.keys())

        else:
            raise NotImplementedError()

        return out_data

    @staticmethod
    def check_attr_to_cls_attn_results(attr2cls_attn: dict, agg: bool = False):
        assert isinstance(attr2cls_attn, dict), "Wrong results data type."
        err_msg = 'Wrong results format.'
        # assert all([k in BinaryClassificationResultsAggregator.categories for k in attr2cls_attn.keys()]), err_msg
        for cat in attr2cls_attn:
            attr2cls_attn_by_cat = attr2cls_attn[cat]
            if attr2cls_attn_by_cat is not None:
                if not agg:
                    assert isinstance(attr2cls_attn_by_cat, pd.DataFrame), err_msg
                else:
                    assert isinstance(attr2cls_attn_by_cat, dict), err_msg
                    assert all([p in attr2cls_attn_by_cat for p in ['mean', 'std']]), err_msg
                    for metric in attr2cls_attn_by_cat:
                        assert isinstance(attr2cls_attn_by_cat[metric], np.ndarray), err_msg

    @staticmethod
    def plot_attr_to_cls_attn_entropy(entropy_res: pd.DataFrame, save_path: str = None):
        assert isinstance(entropy_res, pd.DataFrame)

        entropy_res = entropy_res.rename(columns={'all_pred_pos': 'match', 'all_pred_neg': 'non_match'})
        entropy_res.plot.bar(figsize=(12, 4))
        plt.ylabel('Entropy')
        plt.xticks(rotation=45)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, bbox_inches='tight')

        plt.show()

    @staticmethod
    def plot_attr_to_cls_attn(attr2cls_attn, ax=None, title=None, legend=True):
        AttrToClsAttentionAnalyzer.check_attr_to_cls_attn_results(attr2cls_attn)

        if ax is None:
            fig, ax = plt.subplots(figsize=(20, 12))

        for cat_idx, cat in enumerate(attr2cls_attn):
            attr2cls_attn_by_cat = attr2cls_attn[cat]
            attr2cls_table_stats = attr2cls_attn_by_cat.describe()
            medians = attr2cls_table_stats.loc['50%', :].values
            percs_25 = attr2cls_table_stats.loc['25%', :].values
            percs_75 = attr2cls_table_stats.loc['75%', :].values
            plot_data = {
                'x': range(len(attr2cls_table_stats.columns)),
                'y': medians,
                'yerr': [medians - percs_25, percs_75 - medians],
            }

            plot_cat = cat
            if cat == 'all_pred_pos' or cat == 'all_pos':
                plot_cat = 'match'
            if cat == 'all_pred_neg' or cat == 'all_neg':
                plot_cat = 'non-match'

            if cat_idx == 0:
                color = 'tab:red'
            else:
                color = 'tab:green'

            ax.errorbar(**plot_data, alpha=.75, fmt='o-', capsize=6, label=plot_cat, color=color)
            # plot_data_area = {
            #     'x': plot_data['x'],
            #     'y1': percs_25,
            #     'y2': percs_75
            # }
            # ax.fill_between(**plot_data_area, alpha=.25)
            ax.set_xticks(range(len(attr2cls_attn_by_cat.columns)))
            ax.set_xticklabels(attr2cls_attn_by_cat.columns, rotation=45)
            # ax.xaxis.set_tick_params(labelsize=14)
            ax.yaxis.set_tick_params(labelsize=16)
            if legend:
                ax.legend()

        if title is not None:
            ax.set_title(title, fontsize=18)
        # ax.set_xlabel('Attributes')

    @staticmethod
    def plot_multi_attr_to_cls_attn(attr2cls_attn: dict, small_plot: bool = False, save_path: str = None):

        assert isinstance(attr2cls_attn, dict)

        ncols = 4
        nrows = 3
        figsize = (21, 12)

        if small_plot:
            ncols = 4
            nrows = 1
            figsize = (20, 4)

        if len(attr2cls_attn) == 1:
            ncols = 1
            nrows = 1
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, sharey=True)
        if len(attr2cls_attn) > 1:
            axes = axes.flat
        # loop over the use cases
        for idx, use_case in enumerate(attr2cls_attn):
            if len(attr2cls_attn) == 1:
                ax = axes
                legend = True
            else:
                ax = axes[idx]
                legend = False
            AttrToClsAttentionAnalyzer.plot_attr_to_cls_attn(attr2cls_attn[use_case], ax=ax, title=use_case,
                                                             legend=legend)
            if idx % ncols == 0:
                ax.set_ylabel('[CLS] to attr attention', fontsize=16)

        if not legend:
            handles, labels = ax.get_legend_handles_labels()
            if small_plot:
                fig.legend(handles, labels, bbox_to_anchor=(.6, 0.05), ncol=4, fontsize=16)
            else:
                fig.legend(handles, labels, bbox_to_anchor=(.63, 0.01), ncol=4, fontsize=16)

        plt.tight_layout()
        if small_plot:
            plt.subplots_adjust(wspace=0.02, hspace=0.2)
        else:
            plt.subplots_adjust(wspace=0.02, hspace=0.75)

        if save_path:
            plt.savefig(save_path, bbox_inches='tight')

        plt.show()


class EntityToEntityAttentionAnalyzer(object):

    def __init__(self, attn_data: list, text_unit: str, tokenization: str, analysis_type: str,
                 ignore_special: bool = True, target_categories: list = None):

        assert isinstance(text_unit, str)
        assert text_unit in ['attr', 'word', 'token']
        assert isinstance(tokenization, str)
        assert tokenization in ['sent_pair', 'attr_pair']
        assert isinstance(analysis_type, str)
        assert analysis_type in ['same_entity', 'cross_entity']
        assert isinstance(ignore_special, bool)

        if text_unit == 'attr':
            AttributeAttentionExtractor.check_attn_features(attn_data[0])
        elif text_unit == 'word':
            WordAttentionExtractor.check_attn_features(attn_data[0])
        else:
            AttentionExtractor.check_attn_features(attn_data[0])

        self.attn_data = attn_data
        self.text_unit = text_unit
        self.tokenization = tokenization
        self.analysis_type = analysis_type
        self.ignore_special = ignore_special
        self.target_categories = target_categories

    def __len__(self):
        return len(self.attn_data)

    def __getitem__(self, idx):

        return self.analyze(self.attn_data[idx])

    def analyze_all(self):

        entity_to_entity_attn = []
        for idx in tqdm(range(len(self.attn_data))):
            e2e_attn = self[idx]
            if e2e_attn is None:
                continue
            entity_to_entity_attn.append(e2e_attn)

        aggregator = BinaryClassificationResultsAggregator('score', target_categories=self.target_categories)
        grouped_e2e_attn, _, _, _ = aggregator.add_batch_data(entity_to_entity_attn)

        e2e_attn_results = {}
        for cat in grouped_e2e_attn:
            if grouped_e2e_attn[cat] is not None:
                e2e_attn_results[cat] = pd.DataFrame(grouped_e2e_attn[cat],
                                                     columns=range(1, entity_to_entity_attn[0]['score'].shape[0] + 1))

        return e2e_attn_results

    def analyze(self, attn_item):

        attn_features = attn_item[2]
        attn_values = attn_features['attns']
        if self.text_unit == 'token':
            attn_text_units = attn_features['tokens']
            attn_values = np.concatenate(attn_values)
            if '[PAD]' in attn_text_units:
                pad_idx = attn_text_units.index('[PAD]')
                attn_text_units = attn_text_units[:pad_idx]
                attn_values = attn_values[:, :, :pad_idx, :pad_idx]
        else:
            attn_text_units = attn_features['text_units']
            if self.text_unit == 'attr':
                attn_text_units = attn_text_units + attn_text_units[1:]

        attn_row = {
            'attns': attn_values,
            'text_units': attn_text_units,
            'label': attn_features['labels'].item(),
            'pred': attn_features['preds'].item() if attn_features['preds'] is not None else None,
        }

        # get an average attention map for each layer by averaging all the heads that refer to the same layer
        attns = np.mean(attn_row['attns'], axis=1)

        # find the [SEP] token used to delimit the two entities
        sep_idxs = np.where(np.array(attn_row['text_units']) == '[SEP]')[0]

        # filter out truncated rows
        if len(sep_idxs) % 2 != 0:
            print("Skip truncated row.")
            return None

        if self.tokenization == 'sent_pair':
            entity_delimit = attn_row['text_units'].index('[SEP]')  # get first occurrence of the [SEP] token

        else:  # attr-pair
            # in the attr-pair tokenization the [SEP] token is also used to delimit the attributes
            entity_delimit = sep_idxs[(len(sep_idxs) // 2) - 1]

        # select the top attention scores for each layer-wise attention map
        top_attns = np.zeros((attns.shape[0], attns.shape[1], attns.shape[2]))
        for layer in range(attns.shape[0]):
            layer_attn_map = attns[layer]
            thr = np.quantile(layer_attn_map, 0.8)
            top_layer_attn_map = layer_attn_map >= thr
            top_attns[layer] = top_layer_attn_map

        # count the number of attention scores that passed the previous test in a 'same_entity' or 'cross_entity'
        # perspective

        left_target_idxs = list(range(entity_delimit + 1))
        right_target_idxs = list(range(entity_delimit, top_attns.shape[1]))
        if self.ignore_special:
            left_target_idxs = left_target_idxs[1:]  # remove [CLS]
            left_target_idxs = sorted(list(set(left_target_idxs).difference(set(sep_idxs))))  # remove [SEP]s
            right_target_idxs = sorted(list(set(right_target_idxs).difference(set(sep_idxs))))  # remove [SEP]s

        e2e_attn = np.zeros(top_attns.shape[0])
        for layer in range(top_attns.shape[0]):

            if self.analysis_type == 'same_entity':
                left_hits = top_attns[layer, left_target_idxs, :][:, left_target_idxs].sum()
                right_hits = top_attns[layer, right_target_idxs, :][:, right_target_idxs].sum()

            else:  # cross_entity
                left_hits = top_attns[layer, left_target_idxs, :][:, right_target_idxs].sum()
                right_hits = top_attns[layer, right_target_idxs, :][:, left_target_idxs].sum()

            total_normalized_hits = (left_hits + right_hits) / (len(left_target_idxs) * len(right_target_idxs) * 2)
            e2e_attn[layer] = total_normalized_hits

        return {
            'score': e2e_attn,
            'label': attn_row['label'],
            'pred': attn_row['pred'],
        }

    @staticmethod
    def check_entity_to_entity_attn_results(e2e_results: dict):
        assert isinstance(e2e_results, dict), "Wrong results data type."
        err_msg = 'Wrong results format.'
        # assert all([k in BinaryClassificationResultsAggregator.categories for k in e2e_results.keys()]), err_msg
        for cat in e2e_results:
            e2e_attn_by_cat = e2e_results[cat]
            if e2e_attn_by_cat is not None:
                assert isinstance(e2e_attn_by_cat, pd.DataFrame), err_msg

    @staticmethod
    def plot_entity_to_entity_attn(e2e_attn, ax=None, title=None, legend=True):
        EntityToEntityAttentionAnalyzer.check_entity_to_entity_attn_results(e2e_attn)

        if ax is None:
            fig, ax = plt.subplots(figsize=(20, 12))

        for cat_idx, cat in enumerate(e2e_attn):
            e2e_attn_by_cat = e2e_attn[cat]
            e2e_table_stats = e2e_attn_by_cat.describe()
            medians = e2e_table_stats.loc['50%', :].values
            percs_25 = e2e_table_stats.loc['25%', :].values
            percs_75 = e2e_table_stats.loc['75%', :].values
            plot_data = {
                'x': range(len(e2e_table_stats.columns)),
                'y': medians,
                'yerr': [medians - percs_25, percs_75 - medians],
            }

            plot_cat = cat
            if cat == 'all_pred_pos':
                plot_cat = 'match'
            if cat == 'all_pred_neg':
                plot_cat = 'non-match'

            if cat_idx == 0:
                color = 'tab:red'
            else:
                color = 'tab:green'

            ax.errorbar(**plot_data, alpha=.75, fmt='o-', capsize=3, capthick=1, label=plot_cat, color=color)
            # plot_data_area = {
            #     'x': plot_data['x'],
            #     'y1': percs_25,
            #     'y2': percs_75
            # }
            # ax.fill_between(**plot_data_area, alpha=.25)
            ax.set_xticks(range(len(e2e_attn_by_cat.columns)))
            ax.set_xticklabels(e2e_attn_by_cat.columns, fontsize=16)
            ax.yaxis.set_tick_params(labelsize=18)
            if legend:
                ax.legend()

        if title is not None:
            ax.set_title(title, fontsize=18)
        # ax.set_xlabel('Layers')

    @staticmethod
    def plot_multi_entity_to_entity_attn(e2e_results: dict, small_plot: bool = False, save_path: str = None):

        assert isinstance(e2e_results, dict)

        ncols = 4
        nrows = 3
        figsize = (20, 12)

        if small_plot:
            ncols = 4
            nrows = 1
            figsize = (20, 3)

        if len(e2e_results) == 1:
            ncols = 1
            nrows = 1
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, sharey=True)
        if len(e2e_results) > 1:
            axes = axes.flat
        # loop over the use cases
        for idx, use_case in enumerate(e2e_results):
            if len(e2e_results) == 1:
                ax = axes
            else:
                ax = axes[idx]
            EntityToEntityAttentionAnalyzer.plot_entity_to_entity_attn(e2e_results[use_case], ax=ax, title=use_case,
                                                                       legend=False)
            if idx % ncols == 0:
                ax.set_ylabel('Entity to entity attention', fontsize=16)
            ax.set_xlabel('Layers', fontsize=16)

        handles, labels = ax.get_legend_handles_labels()
        label_map = {'all_pos': 'match', 'all_neg': 'non-match'}
        labels = [label_map[l] if l in label_map else l for l in labels]
        if small_plot:
            fig.legend(handles, labels, bbox_to_anchor=(.62, 0.05), ncol=2, fontsize=16)
        else:
            fig.legend(handles, labels, bbox_to_anchor=(.7, 0.02), ncol=4, fontsize=16)

        plt.tight_layout()
        plt.subplots_adjust(wspace=0.02, hspace=0.3)

        if save_path:
            plt.savefig(save_path, bbox_inches='tight')

        plt.show()


class TopKAttentionAnalyzer(object):
    def __init__(self, attns: list, topk: int = None, topk_method: str = None, tokenization: str = 'sent_pair',
                 pair_mode: bool = False):
        WordAttentionExtractor.check_batch_attn_features(attns)
        if topk is not None:
            assert isinstance(topk, int), "Wrong data type for parameter 'topk'."
        if topk_method is not None:
            assert isinstance(topk_method, str)
            assert topk_method in ['quantile']
        if topk is None:
            assert topk_method is not None
        else:
            assert topk_method is None
        assert isinstance(tokenization, str)
        assert tokenization in ['sent_pair', 'attr_pair']
        assert isinstance(pair_mode, bool)

        attn_data = {}
        for attn in attns:
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

            # ignore (or not) the special tokens and related attention weights
            left_idxs = list(range(entity_delimit + 1))
            right_idxs = list(range(entity_delimit, attn_values.shape[1]))
            left_idxs = left_idxs[1:]  # remove [CLS]
            left_idxs = sorted(list(set(left_idxs).difference(set(sep_idxs))))  # remove [SEP]s
            right_idxs = sorted(list(set(right_idxs).difference(set(sep_idxs))))  # remove [SEP]s
            valid_idxs = np.array(left_idxs + right_idxs)
            attn_values = attn_values[:, valid_idxs, :][:, :, valid_idxs]
            attn_text_units = list(np.array(attn_text_units)[valid_idxs])

            assert attn_values.shape[1] == len(attn_text_units)

            for layer in range(attn_values.shape[0]):
                layer_attn_values = attn_values[layer]

                if not pair_mode:
                    # aggregate the attention values in order to obtain a single attention score for each word
                    # for each word calculate the average attention per row and per column and obtain the maximum value
                    agg_layer_attns = np.maximum(layer_attn_values.mean(axis=0), layer_attn_values.mean(axis=1))
                    attn_scores = agg_layer_attns
                else:
                    attn_scores = layer_attn_values

                item = {
                    'label': label,
                    'pred': pred,
                    'attns': {'text_units': attn_text_units, 'values': attn_scores, 'left': attn[0],
                              'right': attn[1], 'sep_idx': len(left_idxs)},
                }

                if layer not in attn_data:
                    attn_data[layer] = [item]
                else:
                    attn_data[layer].append(item)

        self.attn_data = attn_data
        self.topk = topk
        self.topk_method = topk_method
        self.tokenization = tokenization
        self.pair_mode = pair_mode
        self.analysis_types = ['pos', 'str_type', 'sim']
        self.pos_model = spacy.load('en_core_web_sm')
        self.pos_model.tokenizer = Tokenizer(self.pos_model.vocab, token_match=re.compile(r'\S+').match)

    @staticmethod
    def get_topk_text_units(attn_record: dict, topk: int = None, topk_method: str = None):
        assert isinstance(attn_record, dict), "Wrong data type for parameter 'attn_record'."
        assert all([p in attn_record for p in ['text_units', 'values']])
        if topk is not None:
            assert isinstance(topk, int), "Wrong data type for parameter 'topk'."
        if topk_method is not None:
            assert isinstance(topk_method, str)
            assert topk_method in ['quantile']
        if topk is None:
            assert topk_method is not None
        else:
            assert topk_method is None

        text_units = attn_record['text_units']
        attns = attn_record['values']
        assert len(text_units) == len(attns)

        if topk is not None:
            top_words = list(sorted(zip(attns, text_units), key=lambda x: x[0], reverse=True))[:topk]
            new_topk = topk
        else:
            if topk_method == 'quantile':
                thr = np.quantile(attns, 0.8)
                if attns.ndim == 1:
                    idxs = np.where(attns >= thr)[0]
                    words = [(attns[i], text_units[i]) for i in idxs]
                else:   # attns.ndim == 2
                    row_idxs, col_idxs = np.where(attns >= thr)
                    words = [(attns[i, j], (text_units[i], text_units[j])) for (i, j) in zip(row_idxs, col_idxs)]

                top_words = list(sorted(words, key=lambda x: x[0], reverse=True))
                if len(top_words) > int(len(attns) * 0.2):
                    new_topk = int(len(attns) * 0.2)
                    top_words = top_words[:new_topk]
                else:
                    new_topk = len(top_words)
            else:
                raise ValueError("No method found.")

        if attns.ndim == 1:
            topk_words_idx = np.array(attns).argsort()[-new_topk:][::-1]
        else:
            n = attns.shape[1]
            topk_words_idx = list(zip(*np.argsort(attns, axis=None).__divmod__(n)))[-new_topk:][::-1]

        return {'sent': {'text_units': [t[1] for t in top_words], 'values': [t[0] for t in top_words],
                         'idxs': topk_words_idx, 'original_text': text_units}}

    @staticmethod
    def get_topk_text_units_in_entity_pair_by_attr(attn_record, tok: str, topk, pair_mode=False):
        assert isinstance(attn_record, dict), "Wrong data type for parameter 'attn_record'."
        assert all([p in attn_record for p in ['text_units', 'values', 'sep_idx']])
        assert isinstance(tok, str)
        assert isinstance(topk, int), "Wrong data type for parameter 'topk'."

        def get_topk_text_units_in_entity_by_attr(attn, tokens, entity, topk, offset=0):
            first_attr = str(entity.iloc[0]).split()
            attn_first_attr = tokens[:len(first_attr)]
            if len(attn_first_attr) < len(first_attr):
                first_attr = first_attr[:len(attn_first_attr)]
            assert attn_first_attr == first_attr

            topk_text_units_by_attr = {}
            cum_idx = 0
            for attr, val in entity.iteritems():
                val = str(val).split()
                attn_attr = attn[cum_idx:cum_idx + len(val)]
                attn_attr_tokens = tokens[cum_idx:cum_idx + len(val)]

                if len(val) == len(attn_attr_tokens):
                    assert val == attn_attr_tokens

                topk_words = list(sorted(zip(attn_attr, attn_attr_tokens), key=lambda x: x[0], reverse=True))[:topk]
                topk_words_idx = np.array(attn_attr).argsort()[-topk:][::-1]
                topk_words_idx += (cum_idx + offset)
                cum_idx += len(val)
                topk_text_units_by_attr[attr] = {'text_units': [w[1] for w in topk_words],
                                                 'values': [w[0] for w in topk_words],
                                                 'idxs': topk_words_idx, 'original_text': val}
                if len(val) > len(attn_attr):
                    break

            return topk_text_units_by_attr

        def get_paired_topk_text_units_in_entity_by_attr(attn, left_tokens, right_tokens, left_entity, right_entity,
                                                         topk, sep_idx):

            first_left_attr = str(left_entity.iloc[0]).split()
            attn_first_left_attr = left_tokens[:len(first_left_attr)]
            if len(attn_first_left_attr) < len(first_left_attr):
                first_left_attr = first_left_attr[:len(attn_first_left_attr)]
            assert attn_first_left_attr == first_left_attr

            first_right_attr = str(right_entity.iloc[0]).split()
            attn_first_right_attr = right_tokens[:len(first_right_attr)]
            if len(attn_first_right_attr) < len(first_right_attr):
                first_right_attr = first_right_attr[:len(attn_first_right_attr)]
            assert attn_first_right_attr == first_right_attr

            topk_text_units_by_attr = {}
            left_cum_idx = 0
            right_cum_idx = sep_idx
            for attr in left_entity.index:
                left_val = str(left_entity[attr]).split()
                right_val = str(right_entity[attr]).split()

                left_idxs = (left_cum_idx, left_cum_idx + len(left_val))
                right_idxs = (right_cum_idx, right_cum_idx + len(right_val))

                # check consistency
                attr_left_tokens = left_tokens[left_idxs[0]:left_idxs[1]]
                attr_right_tokens = right_tokens[right_idxs[0] - sep_idx:right_idxs[1] - sep_idx]
                if len(left_val) == len(attr_left_tokens):
                    assert left_val == attr_left_tokens
                if len(right_val) == len(attr_right_tokens):
                    assert right_val == attr_right_tokens

                lr_attr_attn = attn[left_idxs[0]: left_idxs[1]][:, right_idxs[0]: right_idxs[1]]
                rl_attr_attn = attn[right_idxs[0]: right_idxs[1]][:, left_idxs[0]: left_idxs[1]].T
                attr_attn = (lr_attr_attn + rl_attr_attn) / 2
                top_pair_idxs = np.dstack(np.unravel_index(np.argsort(attr_attn.ravel()), (attr_attn.shape[0], attr_attn.shape[1])))[0][-topk:]
                top_pair_text_units = [(attr_left_tokens[p[0]], attr_right_tokens[p[1]]) for p in top_pair_idxs]
                top_pair_attn = [attr_attn[p[0], p[1]] for p in top_pair_idxs]
                top_pair_original_idxs = [(left_cum_idx + p[0], right_cum_idx + p[1]) for p in top_pair_idxs]

                topk_text_units_by_attr[attr] = {'text_units': top_pair_text_units, 'values': top_pair_attn,
                                                 'idxs': top_pair_original_idxs,
                                                 'original_text': (attr_left_tokens, attr_right_tokens)}
                left_cum_idx += len(left_val)
                right_cum_idx += len(right_val)

            return topk_text_units_by_attr

        all_topk = {}
        sep_idx = attn_record['sep_idx']
        left_entity = attn_record['left']
        right_entity = attn_record['right']
        attns = attn_record['values']
        text_units = attn_record['text_units']
        left_tokens = list(np.array(text_units)[:sep_idx])
        right_tokens = list(np.array(text_units)[sep_idx:])

        if not pair_mode:

            left_attns = np.array(attns)[:sep_idx]
            left_topk = get_topk_text_units_in_entity_by_attr(left_attns, left_tokens, left_entity, topk, offset=0)

            right_attns = np.array(attns)[sep_idx:]
            right_topk = get_topk_text_units_in_entity_by_attr(right_attns, right_tokens, right_entity, topk,
                                                               offset=sep_idx)

            all_topk.update({f'l_{attr}': left_topk[attr] for attr in left_topk})
            all_topk.update({f'r_{attr}': right_topk[attr] for attr in right_topk})

        else:
            all_topk = get_paired_topk_text_units_in_entity_by_attr(attns, left_tokens, right_tokens, left_entity,
                                                                    right_entity, topk, sep_idx)

        return all_topk

    @staticmethod
    def get_topk(data: list, target_key: str, tok: str, by_attr: bool, target_categories: list = None,
                 topk: int = None, topk_method: str = None, pair_mode: bool = False):

        aggregator = BinaryClassificationResultsAggregator(target_key, target_categories=target_categories)
        agg_attn_data, _, _, _ = aggregator.add_batch_data(data)

        top_word_by_cat = {}
        for cat in agg_attn_data:
            cat_attn_data = agg_attn_data[cat]
            assert isinstance(agg_attn_data[cat], list)
            assert len(agg_attn_data[cat]) > 0

            if cat_attn_data is None:
                continue

            for attn_idx, attn_data in enumerate(cat_attn_data):

                if by_attr:
                    top_words = TopKAttentionAnalyzer.get_topk_text_units_in_entity_pair_by_attr(attn_data, tok=tok,
                                                                                                 topk=topk,
                                                                                                 pair_mode=pair_mode)
                else:
                    top_words = TopKAttentionAnalyzer.get_topk_text_units(attn_data, topk=topk, topk_method=topk_method)

                if cat not in top_word_by_cat:
                    top_word_by_cat[cat] = [top_words]
                else:
                    top_word_by_cat[cat].append(top_words)

        return top_word_by_cat

    def analyze(self, analysis_type: str, by_attr: bool, target_layer: int = None, target_categories: list = None):
        assert isinstance(analysis_type, str), "Wrong data type for parameter 'analysis_type'."
        assert analysis_type in self.analysis_types, f"Wrong type: {analysis_type} not in {self.analysis_types}."
        assert isinstance(by_attr, bool), "Wrong data type for parameter 'by_attr'."
        if target_layer is not None:
            assert isinstance(target_layer, int)
            assert target_layer in self.attn_data

        # get top-k words at attribute or sentence level
        out_data = {}
        for layer in self.attn_data:
            print(f"Layer#{layer}")
            layer_attn_data = self.attn_data[layer]

            if target_layer is not None:
                if layer != target_layer:
                    continue

            top_words_by_cat = TopKAttentionAnalyzer.get_topk(data=layer_attn_data, target_key='attns',
                                                              tok=self.tokenization, by_attr=by_attr,
                                                              target_categories=target_categories, topk=self.topk,
                                                              topk_method=self.topk_method, pair_mode=self.pair_mode)

            top_results = {}
            # loop over the data categories (e.g., match, non-match, fp, etc.)
            for cat in top_words_by_cat:
                print(f"\tCategory: {cat}")
                top_words_in_cat = top_words_by_cat[cat]

                top_stats = {}
                # loop over the topk words extracted from each record
                for top_words in tqdm(top_words_in_cat):

                    # loop over the topk words related to a specific attribute or the whole sentence
                    for key in top_words:

                        top_words_by_key = top_words[key]

                        if analysis_type == 'str_type':
                            stats_cat = ['alpha', 'punct', 'num', 'no-alpha']
                            for tu in top_words_by_key['text_units']:
                                if tu in string.punctuation:
                                    text_cat = 'punct'
                                elif any(c.isdigit() for c in tu):
                                    text_cat = 'num'
                                elif not tu.isalpha():
                                    text_cat = 'no-alpha'
                                elif tu.isalpha():
                                    text_cat = 'alpha'
                                else:
                                    text_cat = 'other'
                                # print(tu, text_cat)
                                if key not in top_stats:
                                    top_stats[key] = {text_cat: 1}
                                else:
                                    if text_cat not in top_stats[key]:
                                        top_stats[key][text_cat] = 1
                                    else:
                                        top_stats[key][text_cat] += 1

                        elif analysis_type == 'pos':
                            stats_cat = ['TEXT', 'PUNCT', 'NUM&SYM', 'CONN']
                            sent = ' '.join(top_words_by_key['original_text'])
                            sent = self.pos_model(sent)
                            for word_idx, word in enumerate(sent):
                                if word_idx in top_words_by_key['idxs']:
                                    pos_tag = get_pos_tag(word)
                                    # print(word, pos_tag)
                                    if key not in top_stats:
                                        top_stats[key] = {pos_tag: 1}
                                    else:
                                        if pos_tag not in top_stats[key]:
                                            top_stats[key][pos_tag] = 1
                                        else:
                                            top_stats[key][pos_tag] += 1

                        elif analysis_type == 'sim':

                            left_words = top_words_by_key['original_text'][0]
                            right_words = top_words_by_key['original_text'][1]
                            top_attn_words = top_words_by_key['text_units']
                            # left_words_attn = [w[0] for w in top_words_by_key['text_units']]
                            # right_words_attn = [w[1] for w in top_words_by_key['text_units']]
                            top_sim_words = get_most_similar_words_from_sent_pair(left_words, right_words, self.topk)
                            #left_top_sim_words = [t[0] for t in top_sim_words]
                            #right_top_sim_words = [t[1] for t in top_sim_words]

                            if len(top_sim_words) == 0 or len(top_attn_words) < self.topk:
                                continue

                            acc = 0
                            for top_attn_word in top_attn_words:
                                if top_attn_word in [(t[0], t[1]) for t in top_sim_words]:
                                    acc += 1
                            acc /= self.topk

                            if key not in top_stats:
                                top_stats[key] = [acc]
                            else:
                                top_stats[key].append(acc)

                        else:
                            raise NotImplementedError()

                top_norm_stats = {}
                if analysis_type != 'sim':
                    for key in top_stats:
                        top_stats_key = top_stats[key]
                        key = '{}{}'.format(key[:2], key[2:].replace("_", "\n"))
                        tot_count = np.sum(list(top_stats_key.values()))
                        top_norm_stats[key] = {k: int(round((v / tot_count) * 100)) for k, v in top_stats_key.items()}
                        for c in stats_cat:
                            if c not in top_norm_stats[key]:
                                top_norm_stats[key][c] = 0
                else:
                    for key in top_stats:
                        if len(top_stats[key]) >= len(top_words_in_cat) - int(0.10 * len(top_words_in_cat)):
                            print(cat, key, len(top_stats[key]),
                                  len(top_words_in_cat) - int(0.10 * len(top_words_in_cat)))
                            top_norm_stats[key] = np.sum(top_stats[key]) / len(top_stats[key])
                        else:
                            print("REMOVED ", cat, key, len(top_stats[key]))

                        # let's exit from the loop in order to consider only the first attribute
                        # note that from Python 3.7 the dict structure has to preserve the key insertion order
                        break

                top_results[cat] = top_norm_stats

            # change the format of the stats
            out_stats = {}
            if analysis_type != 'sim':
                keys = list(list(top_results.values())[0].keys())
                if len(keys) == 1:
                    for key in keys:
                        key_stats = {cat: top_results[cat][key] for cat in top_results}
                        stats = pd.DataFrame(key_stats)
                        stats = stats.rename(columns={'all_pos': 'match', 'all_neg': 'non_match'})
                        stats = stats.T
                        stats = stats[stats_cat].fillna(0)
                        out_stats[key] = stats
                else:
                    for cat in top_results:
                        stats = pd.DataFrame(top_results[cat])
                        stats = stats.T
                        stats = stats[stats_cat]
                        stats = stats.fillna(0)
                        out_stats[cat] = stats
            else:
                stats = pd.DataFrame([top_results[cat] for cat in top_results], index=list(top_results.keys()))
                stats = stats.rename(index={'all_pos': 'match', 'all_neg': 'non_match'})
                out_stats = {self.topk: stats}

            out_data[layer] = out_stats

        return out_data

    @staticmethod
    def plot_top_attn_stats(plot_data: dict, plot_params: dict, ylabel: str, out_plot_name: str = None,
                            legend: bool = True, y_lim: tuple = None, legend_position=None, small_plot=False):
        assert isinstance(plot_data, dict)
        assert isinstance(plot_params, dict)
        assert isinstance(ylabel, str)
        if out_plot_name is not None:
            assert isinstance(out_plot_name, str)

        ncols = 4
        nrows = 3
        figsize = (20, 10)

        if small_plot:
            ncols = 4
            nrows = 1
            figsize = (15, 3)

        if len(plot_data) == 1:
            ncols = 1
            nrows = 1
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, sharey=True)
        if len(plot_data) > 1:
            axes = axes.flat
        for idx, use_case in enumerate(plot_data):

            if len(plot_data) > 1:
                ax = axes[idx]
            else:
                ax = axes

            use_case_stats = plot_data[use_case]
            if len(plot_data) > 1:
                use_case_stats.plot(**plot_params, ax=ax, legend=False, rot=0)
            else:
                use_case_stats.plot(**plot_params, ax=ax, legend=legend, rot=0)

            ax.set_title(use_case, fontsize=18)
            if idx % ncols == 0:
                if ylabel is not None:
                    ax.set_ylabel(ylabel, fontsize=20)
            if not small_plot:
                ax.set_xlabel("Layers", fontsize=20)
            ax.xaxis.set_tick_params(labelsize=16)
            ax.yaxis.set_tick_params(labelsize=20)
            if len(use_case_stats) > 1:
                ax.set_xticks(list(use_case_stats.index)[::2])
            # ax.set_xticks(range(1, len(use_case_stats) + 1))
            # ax.set_xticklabels(range(1, len(use_case_stats) + 1))
            if y_lim is not None:
                assert len(y_lim) == 2
                # ax.set_ylim(y_lim[0], y_lim[1])
                ax.set_yticks(range(y_lim[0], y_lim[1] + 25, 25))
                ax.set_yticklabels(range(y_lim[0], y_lim[1] + 25, 25))

        if len(plot_data) > 1:
            if legend:
                handles, labels = ax.get_legend_handles_labels()
                if legend_position is None:
                    if small_plot:
                        legend_position = (.76, 0)
                    else:
                        legend_position = (0.70, 0.05)
                fig.legend(handles, labels, bbox_to_anchor=legend_position, ncol=4, fontsize=16)
        if small_plot:
            plt.subplots_adjust(wspace=0.05, hspace=0.2)
        else:
            plt.subplots_adjust(wspace=0.1, hspace=0.6)
        if out_plot_name:
            plt.savefig(out_plot_name, bbox_inches='tight')
        # plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_agg_top_attn_stats_bar(plot_data: dict, agg_dim: str, ylabel: str = None, out_plot_name: str = None,
                                    ylim: tuple = None):

        assert isinstance(plot_data, dict)
        assert isinstance(agg_dim, str)
        assert agg_dim in ['layer', 'use_case']
        if ylabel is not None:
            assert isinstance(ylabel, str)
        if out_plot_name is not None:
            assert isinstance(out_plot_name, str)
        if ylim is not None:
            assert isinstance(ylim, tuple)
            assert len(ylim) == 2

        method_map = {'finetune_sentpair': 'ft_sent', 'finetune_attrpair': 'ft_attr',
                      'pretrain_sentpair': 'pt_sent', 'pretrain_attrpair': 'pt_attr'}

        if agg_dim == 'layer':  # average the score across the layers
            first_item = list(plot_data.values())[0]
            columns = first_item.columns
            avg_plot_data = np.zeros((len(plot_data), len(columns)))
            std_plot_data = np.zeros((len(plot_data), len(columns)))
            for idx, uc in enumerate(plot_data):
                avg_plot_data[idx, :] = plot_data[uc].values.mean(axis=0)
                std_plot_data[idx, :] = plot_data[uc].values.std(axis=0)
            avg_plot_data_table = pd.DataFrame(avg_plot_data, index=plot_data.keys(), columns=columns)
            std_plot_data_table = pd.DataFrame(std_plot_data, index=plot_data.keys(), columns=columns)

        else:  # average the scores across the use cases
            assert len(plot_data) > 1

            first_item = list(plot_data.values())[0]
            columns = first_item.columns
            tab_index = first_item.index
            avg_plot_data = np.zeros((len(tab_index), len(columns)))
            std_plot_data = np.zeros((len(tab_index), len(columns)))
            std_layers_plot_data = [np.zeros((len(plot_data), len(columns))) for _ in range(len(tab_index))]
            for idx, uc in enumerate(plot_data):
                uc_plot_data = plot_data[uc]
                avg_plot_data += uc_plot_data.values
                for l in range(len(uc_plot_data)):
                    std_layers_plot_data[l][idx, :] = uc_plot_data.values[l, :]
            avg_plot_data /= len(plot_data)
            for l in range(len(std_layers_plot_data)):
                std_plot_data[l, :] = std_layers_plot_data[l].std(axis=0)
            avg_plot_data_table = pd.DataFrame(avg_plot_data, index=tab_index, columns=columns)
            std_plot_data_table = pd.DataFrame(std_plot_data, index=tab_index, columns=columns)

        avg_plot_data_table = avg_plot_data_table.rename(columns=method_map)
        std_plot_data_table = std_plot_data_table.rename(columns=method_map)
        avg_plot_data_table.plot(kind='bar', figsize=(12, 3), yerr=std_plot_data_table)
        ax = plt.gca()
        if ylim is not None:
            ax.set_ylim(ylim[0], ylim[1])
        plt.xticks(rotation=0)
        plt.legend(ncol=4, loc='upper center')
        if ylabel is not None:
            plt.ylabel(ylabel)
        if agg_dim == 'use_case':
            plt.xlabel('Layers')
        else:
            plt.xlabel('Datasets')
        plt.tight_layout()

        if out_plot_name is not None:
            plt.savefig(out_plot_name, bbox_inches='tight')

        plt.show()

    @staticmethod
    def plot_agg_top_attn_stats(plot_data: dict, agg_dim: str, ylabel: str = None, out_plot_name: str = None,
                                ylim: tuple = None):

        assert isinstance(plot_data, dict)
        assert isinstance(agg_dim, str)
        assert agg_dim in ['layer', 'use_case']
        if ylabel is not None:
            assert isinstance(ylabel, str)
        if out_plot_name is not None:
            assert isinstance(out_plot_name, str)
        if ylim is not None:
            assert isinstance(ylim, tuple)
            assert len(ylim) == 2

        method_map = {'finetune_sentpair': 'ft_sent', 'finetune_attrpair': 'ft_attr',
                      'pretrain_sentpair': 'pt_sent', 'pretrain_attrpair': 'pt_attr'}
        # method_map = {'pretrain_sentpair': 'pre-trained', 'finetune_sentpair': 'fine-tuned'}

        plt.figure(figsize=(6, 2.5))
        # plt.style.use('seaborn-whitegrid')

        if agg_dim == 'layer':
            first_item = list(plot_data.values())[0]
            methods = first_item.columns
            for method in methods:
                method_plot_data = [plot_data[uc][[method]] for uc in plot_data]
                method_plot_data_tab = pd.concat(method_plot_data, axis=1)
                mathod_plot_data_stats = method_plot_data_tab.describe()
                medians = mathod_plot_data_stats.loc['50%', :].values
                percs_25 = mathod_plot_data_stats.loc['25%', :].values
                percs_75 = mathod_plot_data_stats.loc['75%', :].values

                plot_stats = {
                    'x': range(len(plot_data.keys())),
                    'y': medians,
                    'yerr': [medians - percs_25, percs_75 - medians],
                }

                # plt.errorbar(**plot_stats, alpha=.75, fmt=':', capsize=3, capthick=1, label=method_map[method])
                plt.errorbar(**plot_stats, alpha=.75, fmt='.', capsize=5, label=method_map[method])

        else:
            first_item = list(plot_data.values())[0]
            methods = first_item.columns
            layers = first_item.index
            plot_data_by_layers = {l: np.zeros((len(plot_data), len(methods))) for l in layers}
            for uc_idx, uc in enumerate(plot_data):
                uc_plot_data = plot_data[uc][methods].values
                for l, uc_plot_data_layer in enumerate(uc_plot_data, 1):
                    plot_data_by_layers[l][uc_idx] = uc_plot_data_layer

            for method_idx, method in enumerate(methods):
                method_plot_data = [plot_data_by_layers[l][:, method_idx].reshape((-1, 1)) for l in plot_data_by_layers]
                method_plot_data_tab = pd.DataFrame(np.concatenate(method_plot_data, axis=1))
                mathod_plot_data_stats = method_plot_data_tab.describe()
                medians = mathod_plot_data_stats.loc['50%', :].values
                percs_25 = mathod_plot_data_stats.loc['25%', :].values
                percs_75 = mathod_plot_data_stats.loc['75%', :].values

                plot_stats = {
                    'x': range(len(plot_data_by_layers.keys())),
                    'y': medians,
                    'yerr': [medians - percs_25, percs_75 - medians],
                }

                # plt.errorbar(**plot_stats, alpha=.75, fmt=':', capsize=3, capthick=1, label=method_map[method])
                plt.errorbar(**plot_stats, alpha=.75, fmt='.-', capsize=5, label=method_map[method])

        ax = plt.gca()
        if ylim is not None:
            ax.set_ylim(ylim[0], ylim[1])
        plt.xticks(rotation=0)
        # plt.legend(ncol=4, loc='upper center')
        plt.legend(ncol=4, loc='best')
        if ylabel is not None:
            plt.ylabel(ylabel)
        if agg_dim == 'use_case':
            # plt.xlabel('Layers')
            plt.xticks(range(len(plot_data_by_layers.keys())), plot_data_by_layers.keys())
        else:
            # plt.xlabel('Datasets')
            plt.xticks(range(len(plot_data.keys())), plot_data.keys())
        plt.tight_layout()

        if out_plot_name is not None:
            plt.savefig(out_plot_name, bbox_inches='tight')

        plt.show()
