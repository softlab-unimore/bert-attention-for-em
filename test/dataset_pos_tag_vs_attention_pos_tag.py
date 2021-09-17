import pandas as pd
from utils.general import get_benchmark_pos_tag_distr
import matplotlib.pyplot as plt
import spacy
from spacy.tokenizer import Tokenizer
import re
from utils.test_utils import ConfCreator
from pathlib import Path
import os
import pickle
from scipy.stats import entropy as kl_divergence


PROJECT_DIR = Path(__file__).parent.parent
DATASET_POS_TAG_RES_DIR = os.path.join(PROJECT_DIR, 'results')
ATTN_POS_TAG_RES_DIR = os.path.join(PROJECT_DIR, 'results', 'attention')


def compare_dataset_vs_attn_pos_tag_freq(use_cases: list, conf: dict, sampler_conf: dict, fine_tune: str,
                                         attn_params: dict):
    assert isinstance(use_cases, list)
    assert len(use_cases) > 0
    assert isinstance(conf, dict)
    assert isinstance(sampler_conf, dict)
    assert isinstance(fine_tune, str)
    assert isinstance(attn_params, dict)

    use_case_map = ConfCreator().use_case_map

    extr_name = attn_params['attn_extractor']
    extr_params = attn_params["attn_extr_params"]["agg_metric"]

    # load attention pos tag results
    attn_res_file_name = f"RES_POS_{conf['tok']}_{sampler_conf['size']}_{fine_tune}_{extr_name}_{extr_params}.pkl"
    attn_res_path = os.path.join(ATTN_POS_TAG_RES_DIR, attn_res_file_name)
    attn_res = pickle.load(open(attn_res_path, "rb"))

    dataset_pos_tag_path = os.path.join(DATASET_POS_TAG_RES_DIR, 'RES_pos_tag_distr.pkl')
    dataset_pos_tag_res = pickle.load(open(dataset_pos_tag_path, "rb"))

    divergences = {}
    for use_case in use_cases:
        uc = use_case_map[use_case]
        attn_uc_res = attn_res[use_case].mean(axis=0).apply(lambda x: x / 100)
        dataset_pos_tag_uc_res = dataset_pos_tag_res.loc[uc]

        divergence = kl_divergence(attn_uc_res, qk=dataset_pos_tag_uc_res)
        divergences[uc] = divergence
        print(uc)
        print("\t", attn_uc_res)
        print("\t", dataset_pos_tag_uc_res)

    return divergences


if __name__ == '__main__':

    use_cases = ["Structured_Fodors-Zagats", "Structured_DBLP-GoogleScholar", "Structured_DBLP-ACM",
                 "Structured_Amazon-Google", "Structured_Walmart-Amazon", "Structured_Beer",
                 "Structured_iTunes-Amazon", "Textual_Abt-Buy", "Dirty_iTunes-Amazon", "Dirty_DBLP-ACM",
                 "Dirty_DBLP-GoogleScholar", "Dirty_Walmart-Amazon"]
    # use_cases = ["Structured_Fodors-Zagats", "Structured_Walmart-Amazon", "Structured_iTunes-Amazon", "Textual_Abt-Buy"]

    # [BEGIN] INPUT PARAMS ---------------------------------------------------------------------------------------------
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

    attn_params = {
        'attn_extractor': 'word_extractor',  # 'attr_extractor', 'word_extractor', 'token_extractor'
        'attn_extr_params': {'special_tokens': True, 'agg_metric': 'mean'},
    }

    divergences = compare_dataset_vs_attn_pos_tag_freq(use_cases, conf, sampler_conf, fine_tune, attn_params)
    pd.DataFrame([divergences]).T.plot.bar()
    plt.show()

