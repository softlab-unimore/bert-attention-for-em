import pickle
import os
from pathlib import Path

from utils.data_collector import DM_USE_CASES

RESULTS_DIR = Path(__file__).parent


def load_results(results_dir):
    out_results = {}
    for uc in DM_USE_CASES:
        res_file = f'word_occ_hacking_{uc}_repeat5.pickle'
        full_res_file = os.path.join(results_dir, res_file)
        if os.path.exists(full_res_file):
            with open(full_res_file, "rb") as fp:
                uc_res = pickle.load(fp)
            out_results[uc] = uc_res

    return out_results


def add_total_stats(res):
    out_res = {}
    for uc in res:
        uc_res = res[uc]

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
        if uc not in out_res:
            out_res[uc] = {'tot': tot_stats}
        else:
            out_res[uc].update({'tot': tot_stats})

    return out_res


if __name__ == '__main__':

    results = load_results(RESULTS_DIR)

    out = add_total_stats(results)
    a = []