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


PROJECT_DIR = Path(__file__).parent.parent
RESULTS_DIR = os.path.join(PROJECT_DIR, 'results')


def plot_benchmark_pos_tag_distr(pos_tag_distr: pd.DataFrame, save_path: str = None):
    assert isinstance(pos_tag_distr, pd.DataFrame)
    if save_path is not None:
        assert isinstance(save_path, str)

    figsize = (12, 4)
    # figsize = (8, 3)
    ax = pos_tag_distr.plot.bar(stacked=True, legend=False, figsize=figsize)
    handles, labels = ax.get_legend_handles_labels()
    # plt.legend(handles, labels, bbox_to_anchor=(1.05, -0.1), ncol=4, fontsize=16)
    plt.legend(handles, labels, bbox_to_anchor=(.9, -0.1), ncol=4, fontsize=16)
    plt.ylim(0, 100)
    plt.tick_params(axis='both', labelsize=18)
    plt.ylabel('Freq. (%)', fontsize=18)
    plt.xticks(rotation=0)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')

    plt.show()


if __name__ == '__main__':

    use_cases = ["Structured_Fodors-Zagats", "Structured_DBLP-GoogleScholar", "Structured_DBLP-ACM",
                 "Structured_Amazon-Google", "Structured_Walmart-Amazon", "Structured_Beer",
                 "Structured_iTunes-Amazon", "Textual_Abt-Buy", "Dirty_iTunes-Amazon", "Dirty_DBLP-ACM",
                 "Dirty_DBLP-GoogleScholar", "Dirty_Walmart-Amazon"]
    # use_cases = ["Structured_Fodors-Zagats", "Structured_Walmart-Amazon", "Structured_iTunes-Amazon", "Textual_Abt-Buy"]

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
        'size': 5,
        'target_class': 'both',  # 'both', 0, 1
        'seeds': [42, 42],  # [42 -> class 0, 42 -> class 1]
    }

    pos_model = spacy.load('en_core_web_sm')
    pos_model.tokenizer = Tokenizer(pos_model.vocab, token_match=re.compile(r'\S+').match)
    use_case_map = ConfCreator().use_case_map

    res_save_path = os.path.join(RESULTS_DIR, 'RES_pos_tag_distr.pkl')
    plot_save_path = os.path.join(RESULTS_DIR, 'PLOT_pos_tag_distr.pdf')
    precomputed = True

    if precomputed:
        pos_tag_distr = pickle.load(open(res_save_path, "rb"))
        target_use_cases = [use_case_map[uc] for uc in use_cases]
        pos_tag_distr = pos_tag_distr.loc[target_use_cases]
    else:
        pos_tag_distr = get_benchmark_pos_tag_distr(use_cases, conf, sampler_conf, pos_model)
        pos_tag_distr = pos_tag_distr.rename(index=use_case_map)

        with open(res_save_path, 'wb') as f:
            pickle.dump(pos_tag_distr, f)

    pos_tag_distr = pos_tag_distr.applymap(lambda x: x * 100)
    plot_benchmark_pos_tag_distr(pos_tag_distr, save_path=plot_save_path)
