import pickle
import os
import pandas as pd
from pathlib import Path

from utils.data_collector import DM_USE_CASES
from utils.test_utils import ConfCreator


PROJECT_DIR = Path(__file__).parent.parent.parent
RESULTS_DIR = os.path.join(PROJECT_DIR, 'results', 'sent_sim')


def load_results(path):
    with open(path, "rb") as fp:
        data = pickle.load(fp)

    return data


def get_sim_distributions(data):
    labels = data['labels']
    jac_sims = data['jac_sims']
    cos_sims = data['cos_sims']

    match_distr = pd.DataFrame({
        'jac': jac_sims[labels == 1],
        'cos': cos_sims[labels == 1]
    }).describe().loc[['mean']]
    non_match_distr = pd.DataFrame({
        'jac': jac_sims[labels == 0],
        'cos': cos_sims[labels == 0]
    }).describe().loc[['mean']]
    all_distr = pd.DataFrame({
        'jac': jac_sims,
        'cos': cos_sims
    }).describe().loc[['mean']]

    return match_distr, non_match_distr, all_distr


def get_match_non_match_cosine_sims(data: pd.DataFrame):
    labels = data['labels']
    cos_sims = data['cos_sims']

    out_data = [
        {'data type': 'match', 'cosine sim': cos_sims[labels == 1].mean()},
        {'data type': 'non-match', 'cosine sim': cos_sims[labels == 0].mean()}
    ]
    out = pd.DataFrame(out_data)

    return out


if __name__ == '__main__':

    # res_bert_ft = load_results(os.path.join(RESULTS_DIR, 'bert_ft_sent_corr.pickle'))
    # res_sbert_ft = load_results(os.path.join(RESULTS_DIR, 'sbert_ft_sent_corr.pickle'))
    # res_bert_pt = load_results(os.path.join(RESULTS_DIR, 'bert_sent_corr.pickle'))
    # res_sbert_pt = load_results(os.path.join(RESULTS_DIR, 'sbert_sent_corr.pickle'))
    res_ditto = load_results(os.path.join(RESULTS_DIR, 'ditto_sent_corr.pickle'))
    res_supcon = load_results(os.path.join(RESULTS_DIR, 'supcon_sent_corr.pickle'))

    in_results = [
        ('ditto', res_ditto),
        ('supcon', res_supcon)
    ]

    out_results = []
    for uc in DM_USE_CASES:
        print(f"USE CASE: {uc}")

        # Loop over the results of the multiple models on the current use case
        cos_results = []
        for model_name, res in in_results:
            if uc not in res:
                break
            cos_res = get_match_non_match_cosine_sims(res[uc]['raw'])
            cos_res['model'] = model_name
            cos_results.append(cos_res)

        if len(cos_results) > 0:
            uc_cos_res = pd.concat(cos_results)
            uc_cos_res['data'] = uc
            out_results.append(uc_cos_res)

    out_results = pd.concat(out_results)
    out_results = out_results.reset_index(drop=True)
    out_results['data'] = out_results['data'].map(ConfCreator().use_case_map)

    a = []
