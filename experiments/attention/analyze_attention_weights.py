import os
import pickle
import copy
import itertools
from pathlib import Path
from utils.data_collector import DM_USE_CASES
import argparse
import distutils.util
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
    parser = argparse.ArgumentParser(description='Analysis of attention weights')

    # General parameters
    parser.add_argument('-use_cases', '--use_cases', nargs='+', required=True, choices=DM_USE_CASES + ['all'],
                        help='the names of the datasets')
    parser.add_argument('-data_type', '--data_type', type=str, default='train', choices=['train', 'test', 'valid'],
                        help='dataset types: train, test or valid')
    parser.add_argument('-bert_model', '--bert_model', default='bert-base-uncased', type=str,
                        help='the version of the BERT model')
    parser.add_argument('-tok', '--tok', default='sent_pair', type=str, choices=['sent_pair', 'attr_pair'],
                        help='the tokenizer for the EM entries')
    parser.add_argument('-label', '--label_col', default='label', type=str,
                        help='the name of the column in the EM dataset that contains the label')
    parser.add_argument('-left', '--left_prefix', default='left_', type=str,
                        help='the prefix used to identify the columns related to the left entity')
    parser.add_argument('-right', '--right_prefix', default='right_', type=str,
                        help='the prefix used to identify the columns related to the right entity')
    parser.add_argument('-max_len', '--max_len', default=128, type=int,
                        help='the maximum BERT sequence length')
    parser.add_argument('-permute', '--permute', default=False, type=lambda x: bool(distutils.util.strtobool(x)),
                        help='boolean flag for permuting dataset attributes')
    parser.add_argument('-v', '--verbose', default=False, type=lambda x: bool(distutils.util.strtobool(x)),
                        help='boolean flag for the dataset verbose modality')
    parser.add_argument('-return_offset', '--return_offset', default=False,
                        type=lambda x: bool(distutils.util.strtobool(x)),
                        help='boolean flag for extracting EM entry word indexes')

    # Parameters for data sampling
    parser.add_argument('-sample_size', '--sample_size', type=int,
                        help='size of the sample')
    parser.add_argument('-sample_target_class', '--sample_target_class', default='both', choices=['both', 0, 1],
                        help='classes to sample: match, non-match or both')
    parser.add_argument('-sample_seeds', '--sample_seeds', nargs='+', default=[42, 42],
                        help='seeds for each class sample. <seed non match> <seed match>')

    # Parameters for attention computation
    parser.add_argument('-pre_computed_attns', '--pre_computed_attns', default=False,
                        type=lambda x: bool(distutils.util.strtobool(x)),
                        help='avoid extracting attention weights by loading already extracted attention weights')
    parser.add_argument('-attn_extractor', '--attn_extractor', required=True,
                        choices=['attr_extractor', 'word_extractor', 'token_extractor'],
                        help='type of attention to extract: 1) "attr_extractor": the attention weights are aggregated \
                        by attributes, 2) "word_extractor": the attention weights are aggregated by words, \
                        3) "token_extractor": the original attention weights are retrieved')
    parser.add_argument('-special_tokens', '--special_tokens', default=True,
                        type=lambda x: bool(distutils.util.strtobool(x)),
                        help='consider or ignore special tokens (e.g., [SEP], [CLS])')
    parser.add_argument('-agg_metric', '--agg_metric', required=True, choices=['mean', 'max'],
                        help='method for aggregating the attention weights')
    parser.add_argument('-ft', '--fine_tune', default=False,
                        type=lambda x: bool(distutils.util.strtobool(x)),
                        help='boolean flag for selecting fine-tuned or pre-trained model')

    # Parameters for attention analysis
    parser.add_argument('-attn_tester', '--attn_tester', required=True, choices=['attr_tester', 'attr_pattern_tester'],
                        help='method for analysing the extracted attention weights')
    parser.add_argument('-attn_tester_ignore_special', '--attn_tester_ignore_special', default=True,
                        type=lambda x: bool(distutils.util.strtobool(x)),
                        help='whether the attention weights analyzer has to ignore special tokens')
    parser.add_argument('-save', '--save', default=True,
                        type=lambda x: bool(distutils.util.strtobool(x)),
                        help='whether to save the results of the analysis')

    args = parser.parse_args()

    use_cases = args.use_cases
    if use_cases == ['all']:
        use_cases = DM_USE_CASES

    fixed_params = {
        'label_col': args.label_col,
        'left_prefix': args.left_prefix,
        'right_prefix': args.right_prefix,
        'max_len': args.max_len,
        'verbose': args.verbose,
        'return_offset': args.return_offset,
        'analyzer_params': {'pre_computed_attns': args.pre_computed_attns},
    }

    variable_params = {
        'use_case': use_cases,
        'data_type': [args.data_type],
        'permute': [args.permute],
        'model_name': [args.bert_model],
        'tok': [args.tok],
        'size': [args.sample_size],
        'target_class': [args.sample_target_class],
        'fine_tune_method': [args.fine_tune],
        'extractor': [
            {
                'attn_extractor': args.attn_extractor,
                'attn_extr_params': {'special_tokens': args.special_tokens, 'agg_metric': args.agg_metric},
            }
        ],
        'tester': [
            {
                'tester': args.attn_tester,
                'tester_params': {'ignore_special': args.attn_tester_ignore_special}
            }
        ],
        'seeds': [args.sample_seeds],
    }

    save = args.save

    confs_vals = list(itertools.product(*variable_params.values()))
    confs = [{key: val for (key, val) in zip(list(variable_params), vals)} for vals in confs_vals]
    for conf in confs:
        conf.update(fixed_params)

    num_attempts = len(variable_params['seeds'])

    run(confs, num_attempts, save)
