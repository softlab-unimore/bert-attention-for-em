import os
import nltk
from nltk.corpus import wordnet
import pandas as pd
from utils.general import get_use_case
import numpy as np


def get_df(use_case, data_type):
    use_case_data_dir = get_use_case(use_case)

    if data_type == 'train':
        dataset_path = os.path.join(use_case_data_dir, "train.csv")
    elif data_type == 'test':
        dataset_path = os.path.join(use_case_data_dir, "test.csv")
    else:
        dataset_path = os.path.join(use_case_data_dir, "valid.csv")

    data = pd.read_csv(dataset_path)
    out_data = []
    for idx, row in data.iterrows():
        left_cols = [col for col in row.index if col.startswith('left') and '_id' not in col]
        right_cols = [col for col in row.index if col.startswith('right') and '_id' not in col]
        left_entity = row[left_cols]
        right_entity = row[right_cols]
        out_data.append((left_entity, right_entity))

    return out_data


def get_synonyms_from_sent(word, sent):
    synonyms = set([])
    for synset in wordnet.synsets(word):
        for lemma in synset.lemma_names():
            if lemma in sent and lemma != word:
                synonyms.add(lemma)
    return synonyms


def get_synonyms_from_sent_pair(sent1, sent2):
    synonyms = []
    for word in sent1:
        word_synonyms = get_synonyms_from_sent(word, sent2)
        synonyms += [(word, syn) for syn in word_synonyms]

    # for word in sent2:
    #     word_synonyms = get_synonyms_from_sent(word, sent1)
    #     synonyms += [(syn, word) for syn in word_synonyms]

    return set(synonyms)


def get_use_case_synonyms(pair_of_entities):

    synonyms = {}
    for idx, pair in enumerate(pair_of_entities):
        left_entity = pair[0]
        right_entity = pair[1]
        left_sent = ' '.join([str(val) for val in left_entity if not pd.isnull(val)]).split()
        left_sent = [token for token in left_sent if token.isalpha() and len(token) > 2]
        right_sent = ' '.join([str(val) for val in right_entity if not pd.isnull(val)]).split()
        right_sent = [token for token in right_sent if token.isalpha() and len(token) > 2]
        pair_synonyms = get_synonyms_from_sent_pair(left_sent, right_sent)
        if len(pair_synonyms) > 0:
            for pair_syn in pair_synonyms:
                if pair_syn not in synonyms:
                    synonyms[pair_syn] = [idx]
                else:
                    synonyms[pair_syn].append(idx)

    assert all([len(set(synonyms[syn])) == len(synonyms[syn]) for syn in synonyms])

    num_synonyms = len({tuple(sorted(syn)) for syn in synonyms})
    print(f"Num. synonyms: {num_synonyms}")
    num_rows_with_syn = np.sum([len(syn) for syn in synonyms])
    print(f"Num. rows with a synonym: {num_rows_with_syn} ({(num_rows_with_syn / len(pair_of_entities)) * 100})")

    return synonyms


def get_benchmark_synonyms(use_cases, data_type):
    dfs = [get_df(use_case=uc, data_type=data_type) for uc in use_cases]

    synonyms = {}
    for uc_idx in range(len(use_cases)):
        uc = use_cases[uc_idx]
        print("\n\n", uc)
        pair_of_entities = dfs[uc_idx]
        uc_synonyms = get_use_case_synonyms(pair_of_entities)
        if len(uc_synonyms) == 0:
            uc_synonyms = None
        synonyms[uc] = uc_synonyms

    return synonyms


def get_use_case_matching_spans(pair_of_entities):
    spans = {}
    for idx, pair in enumerate(pair_of_entities):
        left_entity = pair[0]
        right_entity = pair[1]
        left_sent = ' '.join([str(val) for val in left_entity if not pd.isnull(val)]).split()
        left_sent = [token for token in left_sent if token.isalpha() and len(token) > 3]
        right_sent = ' '.join([str(val) for val in right_entity if not pd.isnull(val)]).split()
        right_sent = [token for token in right_sent if token.isalpha() and len(token) > 3]
        common_tokens = set(left_sent).intersection(set(right_sent))
        spans[idx] = common_tokens

    assert len(spans) == len(pair_of_entities)

    return spans


def get_benchmark_matching_spans(use_cases, data_type):
    dfs = [get_df(use_case=uc, data_type=data_type) for uc in use_cases]

    spans = {}
    for uc_idx in range(len(use_cases)):
        uc = use_cases[uc_idx]
        print("\n\n", uc)
        pair_of_entities = dfs[uc_idx]
        uc_spans = get_use_case_matching_spans(pair_of_entities)
        spans[uc] = uc_spans

    return spans


if __name__ == '__main__':
    nltk.download('wordnet')

    use_cases = ["Structured_Fodors-Zagats", "Structured_DBLP-GoogleScholar", "Structured_DBLP-ACM",
                 "Structured_Amazon-Google", "Structured_Walmart-Amazon", "Structured_Beer",
                 "Structured_iTunes-Amazon", "Textual_Abt-Buy", "Dirty_iTunes-Amazon", "Dirty_DBLP-ACM",
                 "Dirty_DBLP-GoogleScholar", "Dirty_Walmart-Amazon"]

    # benchmark_synonyms = get_benchmark_synonyms(use_cases, data_type='train')
    benchmark_spans = get_benchmark_matching_spans(use_cases, data_type='train')
