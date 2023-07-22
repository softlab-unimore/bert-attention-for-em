import os
import pickle
from pathlib import Path
import pandas as pd

from utils.test_utils import ConfCreator

PROJECT_DIR = Path(__file__).parent.parent.parent
RESULTS_DIR = os.path.join(PROJECT_DIR, 'results', 'robustness')


def load_pickle(path: str):
    with open(path, 'rb') as fp:
        out = pickle.load(fp)
    return out


def load_results(dir_path: str):
    res_files = [os.path.join(dir_path, x) for x in os.listdir(dir_path) if x.startswith('ROB')]

    data = []
    for f in res_files:
        file_name = f.split(os.sep)[-1]
        file_name_items = file_name.split('_')
        model = file_name_items[1]
        res = load_pickle(f)
        res['model'] = model
        data.append(res)

    return pd.DataFrame(data)


SCENARIOS = {
    # Same domain
    ('Structured_Walmart-Amazon', 'Textual_Abt-Buy'): 'same',
    ('Textual_Abt-Buy', 'Structured_Walmart-Amazon'): 'same',
    ('Structured_DBLP-GoogleScholar', 'Structured_DBLP-ACM'): 'same',
    ('Structured_DBLP-ACM', 'Structured_DBLP-GoogleScholar'): 'same',
    ('Dirty_DBLP-GoogleScholar', 'Structured_DBLP-ACM'): 'same',
    ('Dirty_DBLP-GoogleScholar', 'Dirty_DBLP-ACM'): 'same',
    # Different domain
    ('Structured_iTunes-Amazon', 'Structured_DBLP-ACM'): 'different',
    ('Structured_iTunes-Amazon', 'Structured_DBLP-GoogleScholar'): 'different',
    ('Structured_DBLP-ACM', 'Structured_iTunes-Amazon'): 'different',
    ('Structured_DBLP-GoogleScholar', 'Structured_iTunes-Amazon'): 'different',
    ('Dirty_iTunes-Amazon', 'Dirty_DBLP-ACM'): 'different',
    ('Dirty_DBLP-ACM', 'Dirty_iTunes-Amazon'): 'different',
}


if __name__ == '__main__':
    results = load_results(RESULTS_DIR)

    use_case_map = ConfCreator().use_case_map
    reverse_uc_map = {v: k for k, v in use_case_map.items()}
    results['source'] = results['source'].map(use_case_map)
    results['target'] = results['target'].map(use_case_map)
    results = pd.pivot_table(results, values='f1', index=['source', 'target'], columns='model')
    results['category'] = results.index.map(lambda x: SCENARIOS[(reverse_uc_map[x[0]], reverse_uc_map[x[1]])])
    results = results.loc[[(use_case_map[x[0]], use_case_map[x[1]]) for x in SCENARIOS.keys()]]


    same_stats = results[results['category'] == 'same'].describe()
    different_stats = results[results['category'] == 'different'].describe()

    results = pd.concat((results, results.describe()))
