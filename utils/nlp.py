import nltk
import numpy as np
from nltk.corpus import wordnet, stopwords
import random
import logging
import string
nltk.download('wordnet')
nltk.download('stopwords')


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
        if min_dist < 0.2:
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
