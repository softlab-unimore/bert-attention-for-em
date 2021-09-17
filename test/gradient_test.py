import os
import time

import pandas as pd
from transformers import AutoTokenizer
from utils.general import get_dataset, get_model, get_sample
from pathlib import Path
from core.explanation.gradient.plot_gradients import plot_batch_grads, plot_multi_use_case_grads,\
    plot_top_grad_stats
from core.explanation.gradient.extractors import EntityGradientExtractor
from core.explanation.gradient.analyzers import TopKGradientAnalyzer
import pickle
from multiprocessing import Process


PROJECT_DIR = Path(__file__).parent.parent
MODELS_DIR = os.path.join(PROJECT_DIR, 'results', 'models')
RESULT_DIR = os.path.join(PROJECT_DIR, 'results', 'gradient_analysis')


def run_gradient_test(conf, sampler_conf, fine_tune, grad_params, models_dir, res_dir):
    assert isinstance(grad_params, dict), "Wrong data type for parameter 'grad_params'."
    params = ['text_unit', 'special_tokens']  # , 'agg', 'agg_target_cat']
    assert all([p in grad_params for p in params])

    dataset = get_dataset(conf)
    tok = conf['tok']
    model_name = conf['model_name']
    use_case = conf['use_case']

    grad_text_unit = grad_params['text_unit']
    grad_special_tokens = grad_params['special_tokens']
    # grad_agg = grad_params['agg']
    # grad_agg_target_cat = grad_params['agg_target_cat']

    if fine_tune is not None:
        model_path = os.path.join(models_dir, fine_tune, f"{use_case}_{tok}_tuned")
    else:
        model_path = None
    model = get_model(model_name, fine_tune, model_path)
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', use_fast=True)

    complete_sampler_conf = sampler_conf.copy()
    complete_sampler_conf['permute'] = conf['permute']
    sample = get_sample(dataset, complete_sampler_conf)

    # [END] MODEL AND DATA LOADING

    entity_grad_extr = EntityGradientExtractor(
        model,
        tokenizer,
        grad_text_unit,
        special_tokens=grad_special_tokens,
        show_progress=True
    )
    out_fname = f"{use_case}_{tok}_{sampler_conf['size']}_{fine_tune}_{grad_text_unit}_{grad_special_tokens}"
    # out_fname = f"{use_case}_{tok}_{sampler_conf['size']}_{fine_tune}_{grad_text_unit}_{grad_special_tokens}_ALL"
    out_dir = os.path.join(res_dir, use_case, out_fname)
    # save grads data
    entity_grad_extr.extract(sample, sample.max_len, out_path=out_dir)

    # if grad_agg:        # aggregate gradient data
    #     if grad_text_unit == 'attrs':
    #         aggregator = AggregateAttributeGradient(grads_data, target_categories=grad_agg_target_cat)
    #         agg_grads = aggregator.aggregate(grad_agg)
    #     else:
    #         raise NotImplementedError("No aggregation for non-attrs gradients.")


def load_saved_grads_data(use_case, conf, sampler_conf, fine_tune, grad_conf, res_dir):
    tok = conf['tok']
    size = sampler_conf['size']
    text_unit = grad_conf['text_unit']
    special_tokens = grad_conf['special_tokens']
    out_fname = f"{use_case}_{tok}_{size}_{fine_tune}_{text_unit}_{special_tokens}"
    # out_fname = f"{use_case}_{tok}_{size}_{fine_tune}_{text_unit}_{special_tokens}_ALL"
    data_path = os.path.join(res_dir, use_case, out_fname)
    uc_grad = pickle.load(open(f"{data_path}.pkl", "rb"))
    return uc_grad


if __name__ == '__main__':

    # [BEGIN] PARAMS

    use_cases = ["Structured_Fodors-Zagats", "Structured_DBLP-GoogleScholar", "Structured_DBLP-ACM",
                 "Structured_Amazon-Google", "Structured_Walmart-Amazon", "Structured_Beer",
                 "Structured_iTunes-Amazon", "Textual_Abt-Buy", "Dirty_iTunes-Amazon", "Dirty_DBLP-ACM",
                 "Dirty_DBLP-GoogleScholar", "Dirty_Walmart-Amazon"]

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
        'size': 50,
        'target_class': 'both',  # 'both', 0, 1
        'seeds': [42, 42],  # [42 -> class 0, 42 -> class 1]
    }

    fine_tune = 'simple'  # None, 'simple', 'advanced'

    grad_conf = {
        'text_unit': 'words',
        'special_tokens': True,
        'agg': None,  # 'mean'
        'agg_target_cat': ['all', 'all_pos', 'all_neg', 'all_pred_pos', 'all_pred_neg', 'tp', 'tn', 'fp', 'fn']
    }

    experiment = 'compute_grad'

    # [END] PARAMS

    start = time.time()

    if experiment == 'compute_grad':
        # # no multi process
        # for use_case in use_cases:
        #     print(use_case)
        #
        #     # if not use_case == 'Structured_DBLP-ACM':
        #     #     continue
        #
        #     uc_conf = conf.copy()
        #     uc_conf['use_case'] = use_case
        #     run_gradient_test(uc_conf, sampler_conf, fine_tune, grad_conf, MODELS_DIR, RESULT_DIR)

        processes = []
        for use_case in use_cases:
            uc_conf = conf.copy()
            uc_conf['use_case'] = use_case
            p = Process(target=run_gradient_test,
                        args=(uc_conf, sampler_conf, fine_tune, grad_conf, MODELS_DIR, RESULT_DIR,))
            processes.append(p)

        for p in processes:
            p.start()

        for p in processes:
            p.join()

    else:
        raise NotImplementedError()

    end = time.time()
    print(end - start)
