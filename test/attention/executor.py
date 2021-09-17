import os
import pickle
import copy
import itertools
from pathlib import Path

from utils.general import get_pipeline


PROJECT_DIR = Path(__file__).parent.parent.parent
MODELS_DIR = os.path.join(PROJECT_DIR, 'results', 'models')
RESULTS_DIR = os.path.join(PROJECT_DIR, 'results', 'attention')


def run(confs: list, num_attempts: int, save: bool):
    assert isinstance(confs, list), "Wrong data type for parameter 'confs'."
    assert len(confs) > 0, "Empty configuration list."
    assert isinstance(num_attempts, int), "Wrong data type for parameter 'num_attempts'."
    assert num_attempts > 0, "Wrong value for parameter 'num_attempts'."
    assert isinstance(save, bool), "Wrong data type for parameter 'save'."

    def update_conf_with_pre_computed_attn_path(conf):
        assert 'analyzer_params' in conf
        assert isinstance(conf['analyzer_params'], dict)
        analyzer_params = conf['analyzer_params']
        assert 'pre_computed_attns' in analyzer_params
        pre_computed_attns = analyzer_params['pre_computed_attns']
        if pre_computed_attns:
            assert 'extractor' in conf
            assert isinstance(conf['extractor'], dict)
            assert all([p in ['attn_extractor', 'attn_extr_params'] for p in conf['extractor']])
            attn_extractor = conf['extractor']['attn_extractor']
            attn_extr_params = conf['extractor']['attn_extr_params']
            assert isinstance(attn_extractor, str)
            assert isinstance(attn_extr_params, dict)

            params = '_'.join([f'{x[0]}={x[1]}' for x in attn_extr_params.items()])
            out_fname = f"ATTN_{conf['use_case']}_{conf['tok']}_{conf['size']}_{conf['fine_tune_method']}_{attn_extractor}_{params}"
            out_file = os.path.join(RESULTS_DIR, conf['use_case'], out_fname)
            pre_computed_attns = f'{out_file}.pkl'

        conf['analyzer_params']['pre_computed_attns'] = pre_computed_attns

        return conf

    confs_by_use_case = {}
    for conf in confs:
        use_case = conf['use_case']
        if use_case not in confs_by_use_case:
            confs_by_use_case[use_case] = [conf]
        else:
            confs_by_use_case[use_case].append(conf)

    for use_case in confs_by_use_case:
        print(use_case)
        use_case_confs = confs_by_use_case[use_case]

        out_path = os.path.join(RESULTS_DIR, use_case)
        Path(out_path).mkdir(parents=True, exist_ok=True)

        all_results = {}
        all_results_counts = {}
        for idx, conf in enumerate(use_case_confs):

            print(conf)
            conf = update_conf_with_pre_computed_attn_path(conf)
            _, _, analyzers = get_pipeline(conf)
            analyzer = analyzers[0]

            extractor_name = conf['extractor']['attn_extractor']
            extractor_params = '_'.join([f'{x[0]}={x[1]}' for x in conf['extractor']['attn_extr_params'].items()])
            tester_name = conf['tester']['tester']
            tester_params = '_'.join([f'{x[0]}={x[1]}' for x in conf['tester']['tester_params'].items()])
            if tester_name == 'attr_pattern_tester':
                tester_name = 'attr_patt_tester'
            template_file_name = 'ANALYSIS_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(use_case, conf['data_type'],
                                                                                    extractor_name, tester_name,
                                                                                    conf['fine_tune_method'],
                                                                                    conf['permute'], conf['tok'],
                                                                                    conf['size'], extractor_params,
                                                                                    tester_params, idx % num_attempts)

            res_out_file_name = os.path.join(out_path, '{}.pickle'.format(template_file_name))

            # run the tests
            testers_res = analyzer.analyze_all()

            # append the current run results to the total results
            res_key = '_'.join(template_file_name.split('_')[:-1])
            if res_key not in all_results:
                res_copy = {}
                res = testers_res[0]
                all_res_cat_counts = {}
                for cat, cat_res in res.items():
                    res_copy[cat] = copy.deepcopy(cat_res)
                    if cat_res is not None:
                        all_res_cat_counts[cat] = 1
                    else:
                        all_res_cat_counts[cat] = 0
                all_results[res_key] = res_copy
                all_results_counts[res_key] = all_res_cat_counts
            else:
                res = testers_res[0]
                for cat, cat_res in res.items():
                    if cat_res is not None:
                        if all_results[res_key][cat] is not None:
                            # all_results[res_key][cat].add_collector(cat_res)
                            all_results[res_key][cat].transform_collector(cat_res, transform_fn=lambda x, y: x + y)
                        else:
                            all_results[res_key][cat] = copy.deepcopy(cat_res)
                        all_results_counts[res_key][cat] += 1

            # save the results into file
            if save:
                with open(res_out_file_name, 'wb') as f:
                    pickle.dump(testers_res, f)

                # # save some stats
                # size = len(sample)

                # y_true, y_pred = analyzer.get_labels_and_preds()
                # f1 = f1_score(y_true, y_pred)
                # print("F1 Match class: {}".format(f1))

                # discarded_rows = attn_extractor.get_num_invalid_attr_attn_maps()
                # print("Num. discarded rows: {}".format(discarded_rows))

                # df = pd.DataFrame([{'size': size, 'f1': f1, 'skip': discarded_rows, 'data_type': data_type}])
                # df.to_csv(os.path.join(drive_results_out_dir, "stats_{}.csv".format(template_file_name)), index=False)

        # average the results
        avg_results = {}
        for res_key in all_results:

            all_res = all_results[res_key]

            avg_res = {}
            for cat, all_cat_res in all_res.items():

                if all_cat_res is not None:
                    assert all_results_counts[res_key][cat] > 0
                    all_cat_res.transform_all(lambda x: x / all_results_counts[res_key][cat])
                    avg_res[cat] = copy.deepcopy(all_cat_res)

            avg_results[res_key] = avg_res

            if save:
                out_avg_file = os.path.join(out_path, '{}_AVG.pickle'.format(res_key))
                with open(out_avg_file, 'wb') as f:
                    pickle.dump(avg_res, f)


if __name__ == '__main__':
    use_cases = ["Structured_Fodors-Zagats", "Structured_DBLP-GoogleScholar", "Structured_DBLP-ACM",
                 "Structured_Amazon-Google", "Structured_Walmart-Amazon", "Structured_Beer",
                 "Structured_iTunes-Amazon", "Textual_Abt-Buy", "Dirty_iTunes-Amazon", "Dirty_DBLP-ACM",
                 "Dirty_DBLP-GoogleScholar", "Dirty_Walmart-Amazon"]

    fixed_params = {
        'label_col': 'label',
        'left_prefix': 'left_',
        'right_prefix': 'right_',
        'max_len': 128,
        'verbose': False,
        'analyzer_params': {'pre_computed_attns': True},
    }

    variable_params = {
        'use_case': use_cases,
        'data_type': ['train'],  #['train', 'test'],
        'permute': [False],
        'model_name': ['bert-base-uncased'],
        'tok': ['sent_pair'],  # 'sent_pair', 'attr', 'attr_pair'
        'size': [None],
        'target_class': ['both'],  # 'both', 0, 1
        'fine_tune_method': [None],  # None, 'simple', 'advanced'
        'extractor': [
            {
                'attn_extractor': 'attr_extractor',     # 'word_extractor'
                'attn_extr_params': {'special_tokens': True, 'agg_metric': 'mean'},
            }
        ],
        'tester': [
            {
                'tester': 'attr_tester',    # 'attr_tester', 'attr_pattern_tester'
                'tester_params': {'ignore_special': True}
            }
        ],
        'seeds': [[42, 42]]#, [42, 24], [42, 12]]
    }

    confs_vals = list(itertools.product(*variable_params.values()))
    confs = [{key: val for (key, val) in zip(list(variable_params), vals)} for vals in confs_vals]
    for conf in confs:
        conf.update(fixed_params)

    save = True
    num_attempts = len(variable_params['seeds'])

    run(confs, num_attempts, save)
