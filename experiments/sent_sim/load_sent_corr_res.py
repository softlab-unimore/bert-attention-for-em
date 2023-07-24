import pickle
import os
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

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


def prepare_data_for_plot(df: pd.DataFrame):
    # Change data format: index=datasets, columns=models
    df = pd.pivot_table(df, index='data', columns='model')
    # Remove the cosine similarity column level
    df = df.droplevel(0, axis=1)
    # Re-order the columns by forcing that the pre-trained model are displayed first
    col_names = [name for name in df.columns if 'pt' in name]
    col_names += [name for name in df.columns if name not in col_names]
    df = df[col_names]
    # Index the rows with the selected dataset order
    df = df.loc[list(ConfCreator().use_case_map.values())]

    return df


def plot_sentence_sim_res(data: pd.DataFrame):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8.75, 3.5), sharey=True)
    axes = axes.flat

    df_match = data[data['data type'] == 'match']
    del df_match['data type']
    df_match = prepare_data_for_plot(df_match)
    sns.boxplot(data=df_match, ax=axes[0], linewidth=0.7)
    axes[0].set_title('Match', fontsize=14)
    # axes[0].get_legend().remove()
    axes[0].tick_params(axis='both', which='major', labelsize=14)
    axes[0].tick_params(axis='both', which='minor', labelsize=14)
    axes[0].set_xlabel('model', fontsize=14)
    axes[0].set_ylabel('cosine sim', fontsize=14)

    df_non_match = data[data['data type'] == 'non-match']
    df_non_match = prepare_data_for_plot(df_non_match)
    sns.boxplot(data=df_non_match, ax=axes[1], linewidth=0.7)
    axes[1].set_title('Non-match', fontsize=14)
    # axes[1].get_legend().remove()
    axes[1].get_yaxis().set_visible(False)
    axes[1].tick_params(axis='both', which='major', labelsize=14, left=False)
    axes[1].tick_params(axis='both', which='minor', labelsize=14)
    axes[1].set_xlabel('model', fontsize=14)

    # handles, labels = axes[1].get_legend_handles_labels()
    # fig.legend(handles, labels, loc='lower center', ncol=2)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "sent_sim_plot.pdf"))


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

    # Collect some intermediate results to be integrated with the current ones
    tmp_res_path = os.path.join(RESULTS_DIR, 'bert_sbert_sent_sim.csv')
    if os.path.exists(tmp_res_path):
        tmp_res = pd.read_csv(tmp_res_path)
        tmp_res['model type'] = tmp_res['model type'].map({'pre-trained': 'pt', 'fine-tuned': 'ft'})
        tmp_res['model'] = tmp_res.apply(lambda x: f"{x['model']}\n{x['model type']}", axis=1)
        del tmp_res['model type']
        out_results = pd.concat((out_results, tmp_res))

    plot_sentence_sim_res(out_results)
