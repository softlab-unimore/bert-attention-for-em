import pickle
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from utils.data_collector import DM_USE_CASES
from utils.test_utils import ConfCreator

RESULTS_DIR = Path(__file__).parent.parent.parent


def load_results(results_dir, repeat):
    out_results = {}
    for uc in DM_USE_CASES:
        res_file = f'INJECTION_{uc}_repeat{repeat}.pickle'
        full_res_file = os.path.join(results_dir, res_file)
        if os.path.exists(full_res_file):
            with open(full_res_file, "rb") as fp:
                uc_res = pickle.load(fp)
            out_results[uc] = uc_res

    return out_results


def add_total_stats(res):
    out_res = {}
    for uc in res:

        if uc not in res:
            continue

        uc_res = res[uc]['random']

        num_samples = None
        idxs = None
        for word, word_stats in uc_res.items():
            if word_stats['perc'] > 0:
                if num_samples is None:
                    num_samples = int((100 * word_stats['num']) // word_stats['perc'])
                if idxs is None:
                    idxs = set(word_stats['idxs'])
                else:
                    idxs = set(idxs).union(set(word_stats['idxs']))

            if uc not in out_res:
                out_res[uc] = {word: word_stats}
            else:
                out_res[uc].update({word: word_stats})

        tot_stats = {'perc': (len(idxs) / num_samples) * 100, 'num': len(idxs), 'idxs': sorted(list(idxs))}
        avg_stats = {'perc': np.mean([x['perc'] for x in out_res[uc].values()]), 'num': None, 'idxs': None}
        std_stats = {'perc': np.std([x['perc'] for x in out_res[uc].values()]), 'num': None, 'idxs': None}
        if uc not in out_res:
            out_res[uc] = {'tot': tot_stats, 'avg': avg_stats, 'std': std_stats}
        else:
            out_res[uc].update({'tot': tot_stats, 'avg': avg_stats, 'std': std_stats})

    return out_res


def plot_percentage_change(data: pd.DataFrame):
    sns.boxplot(data=data, linewidth=0.7)
    # axes[0].set_title('Match', fontsize=14)
    # # axes[0].get_legend().remove()
    # axes[0].tick_params(axis='both', which='major', labelsize=14)
    # axes[0].tick_params(axis='both', which='minor', labelsize=14)
    # axes[0].set_xlabel('model', fontsize=14)
    # axes[0].set_ylabel('cosine sim', fontsize=14)
    plt.show()


def plot_multiple_percentage_changes(data):
    fig, ax = plt.subplots(figsize=(8.75, 3.5))
    sns.boxplot(x="Dataset", hue="repeat", y="Prediction change", data=data, ax=ax)
    # ax.tick_params(axis='both', which='major', labelsize=14, left=False)
    # ax.tick_params(axis='both', which='minor', labelsize=14)
    ax.get_legend().remove()
    # ax.set_xlabel('masking', fontsize=14)
    # ax.set_ylabel('F1', fontsize=14)
    # fig = plt.gcf()
    # fig.set_size_inches(6, 4)
    # plt.tight_layout()
    # plt.savefig("em_masking.pdf")

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=3)
    # fig.legend(handles, labels, bbox_to_anchor=(.6, 0.95), ncol=3, title='Repeat', fontsize=16)

    # plt.tight_layout()
    # plt.show()
    plt.savefig(os.path.join(RESULTS_DIR, 'results', 'robustness', 'robustness_word_repeat.pdf'))


if __name__ == '__main__':

    repeats = [3, 5, 10]
    use_case_map = ConfCreator().use_case_map

    plot_results = []
    for repeat in repeats:
        raw_results = load_results(os.path.join(RESULTS_DIR, 'results', 'robustness'), repeat=repeat)
        results = pd.DataFrame({uc: [x['perc'] for x in res['random'].values()] for uc, res in raw_results.items()})
        results = results.unstack().reset_index()
        results.columns = ['Dataset', 'index', 'Prediction change']
        results['repeat'] = repeat
        results['Dataset'] = results['Dataset'].map(use_case_map)
        plot_results.append(results)
    plot_results = pd.concat(plot_results)

    plot_multiple_percentage_changes(plot_results)

    # results = add_total_stats(results)
    # plot_res = pd.DataFrame({k: [x['perc'] for x in v['random'].values()] for k, v in results.items()})
    # plot_res = plot_res.rename(columns=use_case_map)
    # plot_percentage_change(plot_res)
