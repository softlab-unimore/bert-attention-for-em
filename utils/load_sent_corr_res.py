import pickle
import os
import pandas as pd

from utils.data_collector import DM_USE_CASES


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


if __name__ == '__main__':
    res_dir = 'C:\\Users\\matte\\PycharmProjects\\bert-attention-for-em\\utils'

    res_bert_ft = load_results(os.path.join(res_dir, 'bert_ft_sent_corr.pickle'))
    res_sbert_ft = load_results(os.path.join(res_dir, 'sbert_ft_sent_corr.pickle'))
    res_bert_pt = load_results(os.path.join(res_dir, 'bert_sent_corr.pickle'))
    res_sbert_pt = load_results(os.path.join(res_dir, 'sbert_sent_corr.pickle'))

    match_distr_list = []
    non_match_distr_list = []
    all_distr_list = []
    for uc in DM_USE_CASES:
        print(f"USE CASE: {uc}")

        match_bert_ft_distr, non_match_bert_ft_distr, all_bert_ft_distr = get_sim_distributions(res_bert_ft[uc]['raw'])
        match_sbert_ft_distr, non_match_sbert_ft_distr, all_sbert_ft_distr = get_sim_distributions(
            res_sbert_ft[uc]['raw']
        )
        match_bert_pt_distr, non_match_bert_pt_distr, all_bert_pt_distr = get_sim_distributions(res_bert_pt[uc]['raw'])
        match_sbert_pt_distr, non_match_sbert_pt_distr, all_sbert_pt_distr = get_sim_distributions(
            res_sbert_pt[uc]['raw']
        )

        match_distr = pd.concat(
            (match_bert_ft_distr, match_sbert_ft_distr, match_bert_pt_distr, match_sbert_pt_distr),
            keys=['bert_ft', 'sbert_ft', 'bert_pt', 'sbert_pt'], axis=1
        )
        non_match_distr = pd.concat(
            (non_match_bert_ft_distr, non_match_sbert_ft_distr, non_match_bert_pt_distr, non_match_sbert_pt_distr),
            keys=['bert_ft', 'sbert_ft', 'bert_pt', 'sbert_pt'], axis=1
        )
        all_distr = pd.concat(
            (all_bert_ft_distr, all_sbert_ft_distr, all_bert_pt_distr, all_sbert_pt_distr),
            keys=['bert_ft', 'sbert_ft', 'bert_pt', 'sbert_pt'], axis=1
        )

        match_distr_list.append(match_distr)
        non_match_distr_list.append(non_match_distr)
        all_distr_list.append(all_distr)

    bench_match = pd.concat(match_distr_list, keys=DM_USE_CASES)
    bench_non_match = pd.concat(non_match_distr_list, keys=DM_USE_CASES)
    bench_all = pd.concat(all_distr_list, keys=DM_USE_CASES)

    bench_match_descr = bench_match.describe()
    bench_non_match_descr = bench_non_match.describe()
    bench_all_descr = bench_all.describe()

    out_bench_match = pd.concat((bench_match, bench_match_descr))
    out_bench_non_match = pd.concat((bench_non_match, bench_non_match_descr))
    out_bench_all = pd.concat((bench_all, bench_all_descr))

    a = []
