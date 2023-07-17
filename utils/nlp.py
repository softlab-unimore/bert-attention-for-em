import os.path

import nltk
import numpy as np
from nltk.corpus import wordnet, stopwords
import random
import logging
import pickle
import string
import gensim
import itertools
import pandas as pd
from nltk import ngrams
from tqdm import trange
#nltk.download('wordnet')
#nltk.download('stopwords')


def get_synonyms_from_sent(word, sent):
    synonyms = set([])
    for synset in wordnet.synsets(word):
        for lemma in synset.lemma_names():
            if lemma in sent and lemma != word:
                synonyms.add(lemma)
    return synonyms


def get_synonyms_from_sent_pair(words1, words2, num_words: int = None):
    assert isinstance(words1, list), "Wrong data type for parameter 'words1'."
    assert all([isinstance(w, str) for w in words1]), "Wrong data format for parameter 'words1'."
    assert isinstance(words2, list), "Wrong data type for parameter 'words2'."
    assert all([isinstance(w, str) for w in words2]), "Wrong data format for parameter 'words2'."
    if num_words is not None:
        assert isinstance(num_words, int), "Wrong data type for parameter 'topk'."

    synonyms = []
    for word in words1:
        word_synonyms = get_synonyms_from_sent(word, words2)
        synonyms += [(word, syn) for syn in word_synonyms]

    # for word in words2:
    #     word_synonyms = get_synonyms_from_sent(word, words1)
    #     synonyms += [(syn, word) for syn in word_synonyms]

    synonyms = list(set(synonyms))
    synonyms = [{'left': syn[0], 'right': syn[1]} for syn in synonyms]

    if len(synonyms) == 0:
        return None

    if num_words is None:
        out_data = synonyms
    else:
        out_data = synonyms[:num_words]

    return out_data


def get_random_words_from_sent_pair(words1, words2, num_words: int, exclude_synonyms: bool = False, seed: int = 42):
    assert isinstance(words1, list), "Wrong data type for parameter 'words1'."
    assert all([isinstance(w, str) for w in words1]), "Wrong data format for parameter 'words1'."
    assert isinstance(words2, list), "Wrong data type for parameter 'words2'."
    assert all([isinstance(w, str) for w in words2]), "Wrong data format for parameter 'words2'."
    assert isinstance(num_words, int), "Wrong data type for parameter 'num_words'."
    # assert num_words <= len(words1) * len(words2), f"Too many words requested (max={len(words1) * len(words2)})."
    assert isinstance(exclude_synonyms, bool), "Wrong data type for parameter 'ignore_synonyms'."
    assert isinstance(seed, int), "Wrong data type for parameter 'seed'."

    random.seed(seed)
    words = []
    for i in range(num_words):
        w1 = random.choice(words1)
        w2 = random.choice(words2)

        if exclude_synonyms:
            attempt = 1
            while len(get_synonyms_from_sent(w1, [w2])) > 0 or w1 == w2:
                w1 = random.choice(words1)
                w2 = random.choice(words2)
                attempt += 1

                if attempt == 10:
                    break
            if attempt == 10:
                logging.info("Impossible to select not synonyms words.")

        words.append({'left': w1, 'right': w2})

    return words


def get_common_words_from_sent_pair(words1, words2, num_words: int, seed: int = 42):
    assert isinstance(words1, list), "Wrong data type for parameter 'words1'."
    assert all([isinstance(w, str) for w in words1]), "Wrong data format for parameter 'words1'."
    assert isinstance(words2, list), "Wrong data type for parameter 'words2'."
    assert all([isinstance(w, str) for w in words2]), "Wrong data format for parameter 'words2'."
    assert isinstance(num_words, int), "Wrong data type for parameter 'num_words'."
    assert isinstance(seed, int), "Wrong data type for parameter 'seed'."

    common_words = set(words1).intersection(set(words2))
    common_words = sorted(list(common_words))
    if len(common_words) == 0:
        logging.info("No common words found.")
        return None

    assert num_words <= len(common_words), f"Too many words requested (max={len(common_words)})."

    random.seed(seed)
    words = set([])
    idx = 0
    while len(words) < num_words:
        word = random.choice(list(common_words))
        words.add(word)
        idx += 1

    words = list(words)

    return [{'left': w, 'right': w} for w in words]


def get_synonyms_or_common_words_from_sent_pair(words1, words2, num_words: int, seed: int = 42):
    synonyms = get_synonyms_from_sent_pair(words1, words2, num_words=num_words)
    if synonyms is None:
        return get_common_words_from_sent_pair(words1, words2, num_words, seed=seed)
    if len(synonyms) == num_words:
        return synonyms
    common_words = get_common_words_from_sent_pair(words1, words2, num_words-len(synonyms), seed=seed)
    if common_words is None:
        return synonyms
    return synonyms + common_words


def simple_tokenization_and_clean(text: str):
    assert isinstance(text, str), "Wrong data type for parameter 'text'."

    # remove non-alphabetical and short words
    return [word for word in text.split() if len(word) > 1]# if word.isalpha() and len(word) > 3]


def get_pos_tag(word):
    """Get word POS tag using the Spacy library."""
    pos_tag = word.pos_

    # adjust Spacy's pos tags
    if any(c.isdigit() for c in word.text) and pos_tag != 'NUM':
        pos_tag = 'NUM'
    if not word.text.isalpha() and pos_tag == 'PROPN':
        pos_tag = 'X'
    if word.text == "'" and pos_tag != 'PUNCT':
        pos_tag = 'PUNCT'

    # aggregate Spacy's pos tags
    if pos_tag in ['ADJ', 'ADV', 'AUX', 'NOUN', 'PROPN', 'VERB']:
        pos_tag = 'TEXT'
    elif pos_tag in ['SYM', 'PUNCT']:
        pos_tag = 'PUNCT'
    elif pos_tag in ['NUM']:
        pos_tag = 'NUM&SYM'
    else:
        pos_tag = 'CONN'
    return pos_tag


def get_most_similar_words_from_sent_pair(sent1: list, sent2: list, topk: int):

    assert isinstance(sent1, list), "Wrong data type for parameter 'sent1'."
    assert len(sent1) > 0, "Empty sentence1 tokens."
    assert isinstance(sent2, list), "Wrong data type for parameter 'sent2'."
    assert len(sent2) > 0, "Empty sentence2 tokens."
    assert isinstance(topk, int), "Wrong data type for parameter 'topk'."

    # # filter out non-alphabetic words and stop words
    # stop_words = set(stopwords.words('english'))
    # # sent1 = [w for w in sent1 if w.isalpha() and w not in stop_words]
    # # sent2 = [w for w in sent2 if w.isalpha() and w not in stop_words]
    # sent1 = [w for w in sent1 if w not in stop_words and w not in string.punctuation and len(w) > 2]
    # sent2 = [w for w in sent2 if w not in stop_words and w not in string.punctuation and len(w) > 2]

    if len(sent1) == 0 or len(sent2) == 0:
        return []

    words_by_sim = []
    for w1 in sent1:
        min_dist_idx = np.argmin([nltk.edit_distance(w1, w2) for w2 in sent2])
        min_dist_word = sent2[min_dist_idx]
        min_dist = nltk.edit_distance(w1, min_dist_word)
        if min_dist < 2:
            words_by_sim.append((w1, min_dist_word, min_dist))

    topk_words = sorted(words_by_sim, key=lambda x: x[2])

    if len(topk_words) == 0:
        return topk_words

    # if there are more tokens that occupy the top-k positions in equal merit then return even more than tok-k words
    unique_vals = set([])
    out_words = []
    for topk_word in topk_words:
        unique_vals.add(topk_word[2])
        out_words.append(topk_word)

        if len(unique_vals) > topk:
            break

    if len(unique_vals) > topk:
        out_words = out_words[:-1]

    return out_words


# def get_syntactically_similar_words_from_sent_pair(sent1, sent2, thr, metric, eq=False, return_idxs=False,
#                                                    return_sims=False):
#
#     assert isinstance(sent1, list), "Wrong data type for parameter 'sent1'."
#     assert len(sent1) > 0, "Empty sentence1 tokens."
#     assert isinstance(sent2, list), "Wrong data type for parameter 'sent2'."
#     assert len(sent2) > 0, "Empty sentence2 tokens."
#     assert isinstance(metric, str)
#     assert metric in ['edit', 'jaccard'], "Wrong metric."
#     assert isinstance(eq, bool)
#     if metric == 'edit':
#         assert isinstance(thr, int)
#     elif metric == 'jaccard':
#         assert isinstance(thr, float)
#     else:
#         raise NotImplementedError()
#
#     similar_words = []
#     similar_words_idxs = []
#     similar_words_sims = []
#     all_pairs = list(itertools.product(sent1, sent2))
#     all_pair_idxs = list(itertools.product(range(len(sent1)), range(len(sent2))))
#     for idx, pair in enumerate(all_pairs):
#         left_word, right_word = pair[0], pair[1]
#
#         # remove pairs of words composed by equal words
#         # if left_word == right_word:
#         #     continue
#
#         if len(left_word) < 3 or len(right_word) < 3:
#             continue
#
#         # left_word = left_word.replace('.0', '')
#         # right_word = right_word.replace('.0', '')
#
#         if metric == 'edit':
#             syntax_score = nltk.edit_distance(left_word, right_word)
#         elif metric == 'jaccard':
#             left_char_3grams = list(ngrams(left_word, 3))
#             right_char_3grams = list(ngrams(right_word, 3))
#             intersection = set(left_char_3grams).intersection(set(right_char_3grams))
#             union = set(left_char_3grams).union(set(right_char_3grams))
#             syntax_score = len(intersection) / len(union)
#         else:
#             raise NotImplementedError()
#
#         if eq is True:
#             syntax_cond = syntax_score == thr
#         else:
#             if metric == 'edit':
#                 syntax_cond = syntax_score < thr
#             elif metric == 'jaccard':
#                 syntax_cond = syntax_score > thr
#             else:
#                 raise NotImplementedError()
#
#         if syntax_cond:
#             similar_words.append((left_word, right_word, syntax_score))
#             similar_words_idxs.append(all_pair_idxs[idx])
#             similar_words_sims.append(syntax_score)
#
#     out_dict = {'word_pairs': similar_words}
#     if return_idxs is True:
#         out_dict['word_pair_idxs'] = similar_words_idxs
#     if return_sims is True:
#         out_dict['word_pair_sims'] = similar_words_sims
#
#     return out_dict


# def get_semantically_similar_words_from_sent_pair(sent1, sent2, model, thr, return_idxs=False, return_sims=False):
#     assert isinstance(sent1, list), "Wrong data type for parameter 'sent1'."
#     assert len(sent1) > 0, "Empty sentence1 tokens."
#     assert isinstance(sent2, list), "Wrong data type for parameter 'sent2'."
#     assert len(sent2) > 0, "Empty sentence2 tokens."
#     assert isinstance(thr, float), "Wrong data type for parameter 'thr'."
#
#     similar_words = []
#     similar_words_idxs = []
#     similar_words_sims = []
#     all_pairs = list(itertools.product(sent1, sent2))
#     all_pair_idxs = list(itertools.product(range(len(sent1)), range(len(sent2))))
#     for idx, pair in enumerate(all_pairs):
#         left_word, right_word = pair[0], pair[1]
#
#         # remove pairs of words composed by equal words
#         # if left_word == right_word:
#         #     continue
#
#         if len(left_word) < 3 or len(right_word) < 3:
#             continue
#
#         # left_word = left_word.replace('.0', '')
#         # right_word = right_word.replace('.0', '')
#
#         if left_word in model and right_word in model:
#             sim = model.similarity(left_word, right_word)
#             if sim > thr:
#                 similar_words.append((left_word, right_word, sim))
#                 similar_words_idxs.append(all_pair_idxs[idx])
#                 similar_words_sims.append(sim)
#
#     out_dict = {'word_pairs': similar_words}
#     if return_idxs is True:
#         out_dict['word_pair_idxs'] = similar_words_idxs
#     if return_sims is True:
#         out_dict['word_pair_sims'] = similar_words_sims
#
#     return out_dict

# FIXME: new part
def get_syntactically_similar_words_from_sent_pair(sent1, sent2, thr, metric, eq=False, return_idxs=False,
                                                   return_sims=False, ignore_tokens = None):

    assert isinstance(sent1, list), "Wrong data type for parameter 'sent1'."
    assert len(sent1) > 0, "Empty sentence1 tokens."
    assert isinstance(sent2, list), "Wrong data type for parameter 'sent2'."
    assert len(sent2) > 0, "Empty sentence2 tokens."
    assert isinstance(metric, str)
    assert metric in ['edit', 'jaccard'], "Wrong metric."
    assert isinstance(eq, bool)
    if metric == 'edit':
        assert isinstance(thr, int)
    elif metric == 'jaccard':
        assert isinstance(thr, float)
    else:
        raise NotImplementedError()

    # Remove the indices of the words that have to be ignored
    left_ixs = [ix for ix, word in enumerate(sent1) if word not in ignore_tokens and len(word) >= 3]
    right_ixs = [ix for ix, word in enumerate(sent2) if word not in ignore_tokens and len(word) >= 3]

    all_pair_ixs = list(itertools.product(left_ixs, right_ixs))

    similar_words = []
    similar_words_ixs = []
    similar_words_sims = []
    for pair_ixs in all_pair_ixs:
        left_ix, right_ix = pair_ixs
        left_word = str(sent1[left_ix])
        right_word = str(sent2[right_ix])

        if metric == 'edit':
            syntax_score = nltk.edit_distance(left_word, right_word)
        elif metric == 'jaccard':
            left_char_3grams = list(ngrams(left_word, 3))
            right_char_3grams = list(ngrams(right_word, 3))
            intersection = set(left_char_3grams).intersection(set(right_char_3grams))
            union = set(left_char_3grams).union(set(right_char_3grams))
            syntax_score = len(intersection) / len(union)
        else:
            raise NotImplementedError()

        if eq is True:
            syntax_cond = syntax_score == thr
        else:
            if metric == 'edit':
                syntax_cond = syntax_score < thr
            elif metric == 'jaccard':
                syntax_cond = syntax_score > thr
            else:
                raise NotImplementedError()

        if syntax_cond:
            similar_words.append((left_word, right_word, syntax_score))
            similar_words_ixs.append(pair_ixs)
            similar_words_sims.append(syntax_score)

    out_dict = {'word_pairs': similar_words}
    if return_idxs is True:
        out_dict['word_pair_idxs'] = similar_words_ixs
    if return_sims is True:
        out_dict['word_pair_sims'] = similar_words_sims

    return out_dict


def get_semantically_similar_words_from_sent_pair(sent1, sent2, model, thr, return_idxs=False, return_sims=False,
                                                  ignore_tokens=None):
    assert isinstance(sent1, list), "Wrong data type for parameter 'sent1'."
    assert len(sent1) > 0, "Empty sentence1 tokens."
    assert isinstance(sent2, list), "Wrong data type for parameter 'sent2'."
    assert len(sent2) > 0, "Empty sentence2 tokens."
    assert isinstance(thr, float), "Wrong data type for parameter 'thr'."

    # Remove the indices of the words that have to be ignored
    left_ixs = [ix for ix, word in enumerate(sent1) if word not in ignore_tokens and len(word) >= 3]
    right_ixs = [ix for ix, word in enumerate(sent2) if word not in ignore_tokens and len(word) >= 3]

    all_pair_ixs = list(itertools.product(left_ixs, right_ixs))

    similar_words = []
    similar_words_ixs = []
    similar_words_sims = []
    for pair_ixs in all_pair_ixs:
        left_ix, right_ix = pair_ixs
        left_word = str(sent1[left_ix])
        right_word = str(sent2[right_ix])

        if left_word in model and right_word in model:
            sim = model.similarity(left_word, right_word)
            if sim > thr:
                similar_words.append((left_word, right_word, sim))
                similar_words_ixs.append(pair_ixs)
                similar_words_sims.append(sim)

    out_dict = {'word_pairs': similar_words}
    if return_idxs is True:
        out_dict['word_pair_idxs'] = similar_words_ixs
    if return_sims is True:
        out_dict['word_pair_sims'] = similar_words_sims

    return out_dict


def get_similar_word_pairs(pair_of_entities, sim_type, metric, thrs, op_eq, sem_emb_model=None, continuous_res=False,
                           word_min_len=3):
    assert isinstance(sim_type, str)
    assert sim_type in ['syntax', 'semantic']
    assert isinstance(thrs, list)
    assert len(thrs) > 0
    if sim_type == 'semantic':
        assert sem_emb_model is not None

    word_pairs_map = {thr: {'idxs': [], 'pairs': [], 'sims': [], 'num_all_pairs': 0} for thr in thrs}

    # loop over the entity pairs
    for idx in trange(len(pair_of_entities)):
        pair = pair_of_entities[idx]
        left_entity = pair[0]
        right_entity = pair[1]

        # Get textual entity representation
        left_sent = ' '.join([str(val) for val in left_entity if not pd.isnull(val)]).split()
        if word_min_len is not None:
            left_sent = [token.replace('.0', '') for token in left_sent if len(token) > word_min_len]
        right_sent = ' '.join([str(val) for val in right_entity if not pd.isnull(val)]).split()
        if word_min_len is not None:
            right_sent = [token.replace('.0', '') for token in right_sent if len(token) > word_min_len]

        # Get word pairs with a distance/similarity smaller/greater than the input thresholds
        for thr in thrs:

            if sim_type == 'syntax':
                word_pair_res = get_syntactically_similar_words_from_sent_pair(sent1=left_sent, sent2=right_sent,
                                                                               thr=thr, metric=metric, eq=op_eq,
                                                                               return_sims=continuous_res)
                word_pairs = word_pair_res['word_pairs']
                word_pair_sims = []
                if continuous_res is True:
                    word_pair_sims = word_pair_res['word_pair_sims']
            else:
                word_pair_res = get_semantically_similar_words_from_sent_pair(sent1=left_sent, sent2=right_sent,
                                                                              model=sem_emb_model,
                                                                              thr=thr, return_sims=continuous_res)
                word_pairs = word_pair_res['word_pairs']
                word_pair_sims = []
                if continuous_res is True:
                    word_pair_sims = word_pair_res['word_pair_sims']

            if len(word_pairs) > 0:
                word_pairs_map[thr]['idxs'].append(idx)
                word_pairs_map[thr]['pairs'].append(word_pairs)
                word_pairs_map[thr]['sims'].append(word_pair_sims)
                word_pairs_map[thr]['num_all_pairs'] += len(left_sent) * len(right_sent)

    return word_pairs_map


class FastTextModel:
    def __init__(self):
        data_dir = os.path.join(os.path.abspath('../..'), 'data')
        model_path = os.path.join(data_dir, 'wiki-news-300d-1M', 'wiki-news-300d-1M.vec')
        self.model_cache_path = os.path.join(data_dir, "fast_text_model_with_cache.pkl")
        self.cache = {}

        if os.path.isfile(self.model_cache_path):
            model_with_cache = self.load_model_with_cache()
            self.model = model_with_cache['model']
            self.cache = model_with_cache['cache']
        else:
            self.model = gensim.models.KeyedVectors.load_word2vec_format(model_path, binary=False, encoding='utf8')
            self.save_model_with_cache()

    def __contains__(self, key):
        return key in self.model

    def load_model_with_cache(self):
        return pickle.load(open(self.model_cache_path, "rb"))

    def save_model_with_cache(self):
        model_with_cache = {
            'model': self.model,
            'cache': self.cache
        }
        pickle.dump(model_with_cache, open(self.model_cache_path, "wb"))

    def similarity(self, w1, w2):
        key = tuple(sorted((w1, w2)))
        if key in self.cache:
            return self.cache[key]
        sim = self.model.similarity(w1, w2)
        self.cache[key] = sim
        return sim


