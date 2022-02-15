from core.explanation.gradient.extractors import EntityGradientExtractor
from utils.result_collector import BinaryClassificationResultsAggregator
import numpy as np
import string
import re
import spacy
from spacy.tokenizer import Tokenizer
import pandas as pd
from core.data_models.em_dataset import EMDataset
from utils.nlp import get_pos_tag, get_most_similar_words_from_sent_pair


class TopKGradientAnalyzer(object):
    def __init__(self, grads_data: list, topk: int, metric: str = 'avg', target_entity: str = 'all'):
        EntityGradientExtractor.check_extracted_grad(grads_data)
        assert isinstance(topk, int), "Wrong data type for parameter 'topk'."
        assert isinstance(metric, str), "Wrong data type for parameter 'metric'."
        assert metric in EntityGradientExtractor.grad_agg_fns, "Wrong metric."

        sel_grads_data = []
        for g in grads_data:
            item = {
                'label': g['label'],
                'pred': g['pred'],
                'grad': {'text_units': g['grad'][target_entity], 'values': g['grad'][f'{target_entity}_grad'][metric]},
            }
            sel_grads_data.append(item)
        self.grads_data = sel_grads_data
        self.topk = topk
        self.analysis_types = ['pos', 'str_type', 'sim']
        self.pos_model = spacy.load('en_core_web_sm')
        self.pos_model.tokenizer = Tokenizer(self.pos_model.vocab, token_match=re.compile(r'\S+').match)

    @staticmethod
    def get_topk_text_units(grad_record: dict, topk: int):
        assert isinstance(grad_record, dict), "Wrong data type for parameter 'grad_record'."
        assert all([p in grad_record for p in ['text_units', 'values']])
        assert isinstance(topk, int), "Wrong data type for parameter 'topk'."

        text_units = grad_record['text_units']
        g = grad_record['values']
        top_words = list(sorted(zip(g, text_units), key=lambda x: x[0], reverse=True))[:topk]
        topk_words_idx = np.array(g).argsort()[-topk:][::-1]
        return {'sent': {'text_units': [t[1] for t in top_words], 'values': [t[0] for t in top_words],
                         'idxs': topk_words_idx, 'original_text': text_units}}

    @staticmethod
    def get_topk_text_units_in_entity_pair_by_attr(grad_record, record, topk, pair_mode=False):
        assert isinstance(grad_record, dict), "Wrong data type for parameter 'grad_record'."
        assert all([p in grad_record for p in ['text_units', 'values']])
        assert isinstance(topk, int), "Wrong data type for parameter 'topk'."

        def get_topk_text_units_in_entity_by_attr(grad, grad_tokens, entity, topk, offset=0):
            # check that the input record matches with the grad data
            first_attr = str(entity.iloc[0]).split()
            grad_first_attr = grad_tokens[:len(first_attr)]
            if len(grad_first_attr) < len(first_attr):
                first_attr = first_attr[:len(grad_first_attr)]
            assert grad_first_attr == first_attr

            topk_text_units_by_attr = {}
            cum_idx = 0
            for attr, val in entity.iteritems():
                val = str(val).split()
                grad_attr = grad[cum_idx:cum_idx + len(val)]
                grad_attr_tokens = grad_tokens[cum_idx:cum_idx + len(val)]

                if len(val) == len(grad_attr_tokens):
                    assert val == grad_attr_tokens

                topk_words = list(sorted(zip(grad_attr, grad_attr_tokens), key=lambda x: x[0], reverse=True))[:topk]
                topk_words_idx = np.array(grad_attr).argsort()[-topk:][::-1]
                topk_words_idx += (cum_idx + offset)
                cum_idx += len(val)
                topk_text_units_by_attr[attr] = {'text_units': [w[1] for w in topk_words],
                                                 'values': [w[0] for w in topk_words],
                                                 'idxs': topk_words_idx, 'original_text': val}
                if len(val) > len(grad_attr):
                    break

            return topk_text_units_by_attr

        left_grad_idxs = [i for i, t in enumerate(grad_record['text_units']) if t.startswith('l_')]
        left_grad_tokens = np.array(grad_record['text_units'])[left_grad_idxs]
        left_grad_values = np.array(grad_record['values'])[left_grad_idxs]
        left_topk = get_topk_text_units_in_entity_by_attr(left_grad_values, [t[2:] for t in left_grad_tokens],
                                                          record[0], topk, offset=0)

        right_grad_idxs = [i for i, t in enumerate(grad_record['text_units']) if t.startswith('r_')]
        right_grad_tokens = np.array(grad_record['text_units'])[right_grad_idxs]
        right_grad_values = np.array(grad_record['values'])[right_grad_idxs]
        right_topk = get_topk_text_units_in_entity_by_attr(right_grad_values, [t[2:] for t in right_grad_tokens],
                                                           record[1], topk, offset=right_grad_idxs[0])

        all_topk = {}
        if pair_mode:
            for attr in left_topk:
                if attr in right_topk:
                    all_topk[attr] = (left_topk[attr], right_topk[attr])
        else:
            all_topk.update({f'l_{attr}': left_topk[attr] for attr in left_topk})
            all_topk.update({f'r_{attr}': right_topk[attr] for attr in right_topk})

        return all_topk

    def get_topk(self, by_attr: bool, target_categories: list = None, ignore_special: bool = True,
                entities: EMDataset = None, pair_mode=False):

        aggregator = BinaryClassificationResultsAggregator('grad', target_categories=target_categories)
        agg_grads_data, agg_grads_idxs, _, _ = aggregator.add_batch_data(self.grads_data)

        top_word_by_cat = {}
        new_agg_grads_data = {cat: [] for cat in agg_grads_data}
        for cat in agg_grads_data:
            cat_grads_data = agg_grads_data[cat]
            assert isinstance(agg_grads_data[cat], list)
            assert len(agg_grads_data[cat]) > 0
            cat_grads_idxs = agg_grads_idxs[cat]

            if cat_grads_data is None:
                continue

            for grad_idx, grad_data in enumerate(cat_grads_data):
                # print("--------")

                new_grad_data = grad_data.copy()
                if ignore_special:
                    text_units = grad_data['text_units']
                    g = grad_data['values']
                    sep_idxs = list(np.where(np.array(text_units) == '[SEP]')[0])
                    skip_idxs = [0] + sep_idxs

                    if text_units[0] == '[CLS]':
                        g = [g[i] for i in range(len(text_units)) if i not in skip_idxs]
                        text_units = [text_units[i] for i in range(len(text_units)) if i not in skip_idxs]

                    new_grad_data['text_units'] = text_units
                    new_grad_data['values'] = g
                new_agg_grads_data[cat].append(new_grad_data)

                if by_attr:
                    if cat_grads_idxs[grad_idx] >= len(entities):
                        continue

                    top_words = TopKGradientAnalyzer.get_topk_text_units_in_entity_pair_by_attr(new_grad_data,
                                    entities[cat_grads_idxs[grad_idx]], self.topk, pair_mode=pair_mode)
                else:
                    top_words = TopKGradientAnalyzer.get_topk_text_units(new_grad_data, self.topk)

                if cat not in top_word_by_cat:
                    top_word_by_cat[cat] = [top_words]
                else:
                    top_word_by_cat[cat].append(top_words)

        return new_agg_grads_data, top_word_by_cat

    def analyze(self, analysis_type: str, by_attr: bool, target_categories: list = None, ignore_special: bool = True,
                entities: EMDataset = None):
        assert isinstance(analysis_type, str), "Wrong data type for parameter 'analysis_type'."
        assert analysis_type in self.analysis_types, f"Wrong type: {analysis_type} not in {self.analysis_types}."
        assert isinstance(by_attr, bool), "Wrong data type for parameter 'by_attr'."
        if entities is not None:
            assert isinstance(entities, EMDataset), "Wrong data type for parameter 'entities'."
        if by_attr:
            assert entities is not None, "Entities not provided in by_attr modality."

        # get top-k words at attribute or sentence level
        pair_mode = False
        if analysis_type == 'sim':
            pair_mode = True
        agg_grads_data, top_words_by_cat = self.get_topk(by_attr, target_categories, ignore_special, entities,
                                                         pair_mode=pair_mode)

        out_data = {}
        # loop over the data categories (e.g., match, non-match, fp, etc.)
        for cat in top_words_by_cat:
            top_words_in_cat = top_words_by_cat[cat]
            cat_grads_data = agg_grads_data[cat]

            top_stats = {}
            # loop over the topk words extracted from each record
            for idx, top_words in enumerate(top_words_in_cat):
                grad_data = cat_grads_data[idx]

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
                        sent = ' '.join([t[2:] for t in grad_data['text_units']])
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
                        left_top_grad = top_words_by_key[0]
                        right_top_grad = top_words_by_key[1]
                        left_words = left_top_grad['original_text']
                        right_words = right_top_grad['original_text']
                        top_sim_words = get_most_similar_words_from_sent_pair(left_words, right_words, self.topk)

                        if len(top_sim_words) == 0:
                            continue

                        acc = 0
                        for top_sim_word in top_sim_words:
                            if top_sim_word[0] in left_top_grad['text_units'] and top_sim_word[1] in right_top_grad['text_units']:
                                acc += 1
                        acc /= len(top_sim_words)
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
            else:
                for key in top_stats:
                    if len(top_stats[key]) >= len(top_words_in_cat) - int(0.10 * len(top_words_in_cat)):
                        print(cat, key, len(top_stats[key]), len(top_words_in_cat) - int(0.10 * len(top_words_in_cat)))
                        top_norm_stats[key] = np.sum(top_stats[key]) / len(top_stats[key])
                    else:
                        print("REMOVED ", cat, key, len(top_stats[key]))
                    break   # select only the first attribute
            out_data[cat] = top_norm_stats

        out_stats = {}
        if analysis_type != 'sim':
            keys = list(list(out_data.values())[0].keys())
            if len(keys) == 1:
                for key in keys:
                    key_stats = {cat: out_data[cat][key] for cat in out_data}
                    stats = pd.DataFrame(key_stats)
                    stats = stats.rename(columns={'all_pos': 'match', 'all_neg': 'non_match'})
                    stats = stats.T
                    stats = stats[stats_cat]
                    out_stats[key] = stats
            else:
                for cat in out_data:
                    stats = pd.DataFrame(out_data[cat])
                    stats = stats.T
                    stats = stats[stats_cat]
                    stats = stats.fillna(0)
                    out_stats[cat] = stats
        else:
            stats = pd.DataFrame([out_data[cat] for cat in out_data], index=list(out_data.keys()))
            stats = stats.rename(index={'all_pos': 'match', 'all_neg': 'non_match'})
            out_stats = {self.topk: stats}

        return out_stats
