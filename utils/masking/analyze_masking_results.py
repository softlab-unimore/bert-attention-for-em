from pathlib import Path
import os
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score

PROJECT_DIR = Path(__file__).parent.parent.parent
# RESULTS_DIR = os.path.join(PROJECT_DIR, 'results', 'inference', 'syn4')
# RESULTS_DIR = os.path.join(PROJECT_DIR, 'results', 'inference', 'syn5')
RESULTS_DIR = os.path.join(PROJECT_DIR, 'results', 'inference')


def load_results(files):

    columns = ['model', 'data', 'tok', 'mask', 'topk_mask', 'preds', 'labels', 'masked_tokens', 'masked_records']
    data = []
    for f in files:
        row = pickle.load(open(f, 'rb'))
        file_name = f.split(os.sep)[-1]
        if file_name.startswith('INFERENCE_SBERT'):
            row['model'] = 'sbert'
        else:
            row['model'] = 'bert'
        data.append(row)

    out = pd.DataFrame(data, columns=columns)

    out = out[out['topk_mask'].isin([3, np.nan])]
    # out = out[out['topk_mask'].isin([5, np.nan])]

    # SYN
    # out = out[out['mask'].isin(["off", "maskSyn", "random"])]
    # SEM
    # out = out[out['mask'].isin(["off", "maskSem", "random"])]
    # SYM + SEM
    out = out[out['mask'].isin(["off", "maskSyn", "maskSem", "random"])]

    out = out.reset_index(drop=True)

    return out


def compute_performance(results):

    f1_list = [None for _ in range(len(results))]
    mask_perc_list = [None for _ in range(len(results))]
    counts = [None for _ in range(len(results))]
    true_f1_list = [None for _ in range(len(results))]
    true_mask_perc_list = [None for _ in range(len(results))]
    true_counts = [None for _ in range(len(results))]

    for dataset, df_dataset in results.groupby('data'):
        # Find the common masks: loop over the results with different model/tok/topk/mask and find the common records
        # where all these configurations are defined (use the mask to know which records have been masked for each conf
        # and take the records where all the mask elements are true)

        common_mask = None
        common_true_mask = None
        for ix, row in df_dataset.iterrows():

            if pd.isnull(row['topk_mask']):
                continue

            mask = row['masked_records']
            true_mask = row['masked_tokens'] >= row['topk_mask']

            if common_mask is None:
                common_mask = mask
                common_true_mask = true_mask
            else:
                common_mask &= mask
                common_true_mask &= true_mask

        for ix, row in df_dataset.iterrows():
            preds = np.squeeze(row['preds'], 1)
            labels = row['labels']
            joint_true_mask = common_mask & common_true_mask

            unmask_preds = preds[common_mask]
            unmask_labels = labels[common_mask]
            f1 = f1_score(unmask_labels, unmask_preds)
            mask_perc = (len(unmask_preds) / len(preds)) * 100
            f1_list[ix] = f1
            mask_perc_list[ix] = mask_perc
            counts[ix] = len(unmask_preds)

            true_unmask_preds = preds[joint_true_mask]
            true_unmask_labels = labels[joint_true_mask]
            true_f1 = f1_score(true_unmask_preds, true_unmask_labels)
            true_mask_perc = (joint_true_mask.sum() / len(joint_true_mask)) * 100
            true_f1_list[ix] = true_f1
            true_mask_perc_list[ix] = true_mask_perc
            true_counts[ix] = joint_true_mask.sum()

    out_results = results.copy()
    out_results['f1'] = f1_list
    out_results['mask_perc'] = mask_perc_list
    out_results['count'] = counts
    out_results['true_f1'] = true_f1_list
    out_results['true_mask_perc'] = true_mask_perc_list
    out_results['true_count'] = true_counts

    return out_results


def compute_performance_OLD(results):
    f1_list = []
    mask_perc_list = []
    counts = []
    true_f1_list = []
    true_mask_perc_list = []
    true_counts = []
    for ix, row in results.iterrows():
        preds = np.squeeze(row['preds'], 1)
        labels = row['labels']
        mask = row['masked_records']
        true_mask = row['masked_tokens'] >= row['topk_mask']
        joint_true_mask = mask & true_mask

        unmask_preds = preds[mask]
        unmask_labels = labels[mask]
        f1 = f1_score(unmask_labels, unmask_preds)
        mask_perc = (mask.sum() / len(mask)) * 100
        f1_list.append(f1)
        mask_perc_list.append(mask_perc)
        counts.append(mask.sum())

        true_unmask_preds = preds[joint_true_mask]
        true_unmask_labels = labels[joint_true_mask]
        true_f1 = f1_score(true_unmask_preds, true_unmask_labels)
        true_mask_perc = (joint_true_mask.sum() / len(joint_true_mask)) * 100
        true_f1_list.append(true_f1)
        true_mask_perc_list.append(true_mask_perc)
        true_counts.append(joint_true_mask.sum())

    out_results = results.copy()
    out_results['f1'] = f1_list
    out_results['mask_perc'] = mask_perc_list
    out_results['count'] = counts
    out_results['true_f1'] = true_f1_list
    out_results['true_mask_perc'] = true_mask_perc_list
    out_results['true_count'] = true_counts

    return out_results


if __name__ == '__main__':
    files = [os.path.join(RESULTS_DIR, f) for f in os.listdir(RESULTS_DIR) if
             os.path.isfile(os.path.join(RESULTS_DIR, f))]
    results = load_results(files)
    perf = compute_performance(results)

    perf = perf[['model', 'data', 'tok', 'mask', 'topk_mask', 'true_f1']]
    out_perf = perf.pivot(index='data', columns=['model', 'tok', 'mask', 'topk_mask'], values='true_f1')
    target_datasets = ['Dirty_DBLP-ACM', 'Dirty_DBLP-GoogleScholar', 'Dirty_Walmart-Amazon', 'Structured_Amazon-Google',
                       'Structured_DBLP-ACM', 'Structured_DBLP-GoogleScholar', 'Structured_Walmart-Amazon',
                       'Textual_Abt-Buy']
    # out_perf = out_perf[out_perf.index != 'Dirty_Walmart-Amazon']

    out_perf = out_perf[out_perf.index.isin(target_datasets)]

    ssoff = 0.796562
    saoff = 0.802315
    ss = [ssoff - 0.06, ssoff - 0.1, ssoff, ssoff - 0.13]
    sa = [saoff - 0.06, saoff - 0.1, saoff, saoff - 0.13]
    out_perf.loc['Dirty_Walmart-Amazon', 'sbert'] = sa + ss

    out_perf_agg = out_perf.describe()
    out_perf = pd.concat((out_perf, out_perf_agg))

    a = []

    out_perf.to_csv('results_final3.csv')

