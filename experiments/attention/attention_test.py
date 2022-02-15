import os
import copy
import numpy as np
from pathlib import Path
import matplotlib.image as mpimg
from utils.result_collector import TestResultCollector, BinaryClassificationResultsAggregator
from utils.plot import plot_results, plot_benchmark_results, plot_agg_results, plot_comparison, plot_images_grid
from utils.test_utils import ConfCreator
from utils.attention_utils import get_analysis_results
from utils.data_collector import DM_USE_CASES
import argparse
import distutils.util

PROJECT_DIR = Path(__file__).parent.parent.parent
RESULTS_DIR = os.path.join(PROJECT_DIR, 'results', 'attention')


def cmp_results(res1: dict, res2: dict):
    def check_data(res):
        assert isinstance(res, dict)
        for cat in res:
            if res[cat] is not None:
                assert isinstance(res[cat], TestResultCollector)

    check_data(res1)
    check_data(res2)
    res1 = copy.deepcopy(res1)
    res2 = copy.deepcopy(res2)

    cmp_res = {}
    for cat in res1:
        cat_res1 = None
        if cat in res1:
            cat_res1 = res1[cat]

        cat_res2 = None
        if cat in res2:
            cat_res2 = res2[cat]

        if cat_res1 is None or cat_res2 is None:
            print(f"Skip {cat} results.")
            continue

        out_cat_res = copy.deepcopy(cat_res1)
        out_cat_res.transform_collector(cat_res2, lambda x, y: x - y)
        cmp_res[cat] = out_cat_res

    return cmp_res


def cmp_benchmark_results(results1, results2):
    bench_cmp_res = {}
    for use_case in list(results1):
        res1 = results1[use_case]
        res2 = results2[use_case]

        cmp_res = cmp_results(res1, res2)
        bench_cmp_res[use_case] = cmp_res

    return bench_cmp_res


def aggregate_results(results: dict, agg_fns: list, result_ids: list):
    assert isinstance(results, dict)
    assert isinstance(agg_fns, list)
    assert len(agg_fns) > 0
    assert isinstance(result_ids, list)
    assert len(results) > 0

    new_agg_fns = []
    for agg_fn in agg_fns:
        if agg_fn == 'row_mean':
            agg_fn = lambda x: x.mean(1)
        elif agg_fn == 'row_std':
            agg_fn = lambda x: x.std(1)
        else:
            raise ValueError("Wrong value for the aggregate function.")
        new_agg_fns.append(agg_fn)

    agg_cat_results = {}
    out_use_cases = []

    # aggregate the results
    for use_case in results:  # loop over the results
        use_case_res = results[use_case]
        assert isinstance(use_case_res, dict)
        out_use_cases.append(use_case)

        for cat in use_case_res:  # loop over category results

            target_res = {}
            for idx, agg_fn in enumerate(new_agg_fns):
                cat_res = copy.deepcopy(use_case_res[cat])
                assert isinstance(cat_res, TestResultCollector)

                agg_target_res = {}
                for result_id in result_ids:  # aggregate the result ids
                    res = cat_res.get_result(result_id)
                    agg_target_res[result_id] = agg_fn(res).reshape((-1, 1))
                target_res[agg_fns[idx]] = agg_target_res

            if cat not in agg_cat_results:
                agg_cat_results[cat] = [target_res]
            else:
                agg_cat_results[cat].append(target_res)

    agg_results = {}  # concat aggregated results
    for cat in agg_cat_results:
        agg_cat_res = agg_cat_results[cat]
        assert isinstance(agg_cat_res, list)
        assert len(agg_cat_res) > 0

        current_cat_res = {}
        for use_case_res in agg_cat_res:
            if len(current_cat_res) == 0:
                current_cat_res = copy.deepcopy(use_case_res)
            else:
                for agg_metric in current_cat_res:

                    for result_id in current_cat_res[agg_metric]:
                        current_cat_res[agg_metric][result_id] = np.concatenate([current_cat_res[agg_metric][result_id],
                                                                                 use_case_res[agg_metric][result_id]],
                                                                                axis=1)

        agg_results[cat] = current_cat_res

    return agg_results


def cmp_agg_results(res1, res2, target_cats):
    cmp_res = {}
    for cat in res1:

        if cat not in target_cats:
            continue

        cat_res1 = None
        if cat in res1:
            cat_res1 = res1[cat]

        cat_res2 = None
        if cat in res2:
            cat_res2 = res2[cat]

        if cat_res1 is None or cat_res2 is None:
            print(f"Skip {cat} results.")
            continue

        out_cat_res = copy.deepcopy(cat_res1)
        for metric in cat_res2:
            for res_id in cat_res2[metric]:
                out_cat_res[metric][res_id] -= cat_res2[metric][res_id]
        cmp_res[cat] = out_cat_res

    return cmp_res


def use_case_analysis(conf: dict, plot_params: list, categories: list, agg_fns: list = None,
                      target_agg_result_ids: list = None, plot_type: str = 'simple', save_path: str = None):
    assert isinstance(conf, dict), "Wrong data type for parameter 'conf'."
    assert isinstance(plot_params, list), "Wrong data type for parameter 'plot_params'."
    assert len(plot_params) > 0, "Wrong value for parameter 'plot_params'."
    assert isinstance(categories, list), "Wrong data type for parameter 'categories'."
    assert len(categories) > 0, "Wrong value for parameter 'categories'."
    if agg_fns is not None:
        assert isinstance(agg_fns, list), "Wrong data type for parameter 'agg_fns'."
        assert len(agg_fns) > 0, "Empty aggregation functions."
    if target_agg_result_ids is not None:
        assert isinstance(target_agg_result_ids, list), "Wrong data type for parameter 'target_agg_result_ids'."
        assert len(target_agg_result_ids) > 0, "Empty target aggregated results."

    res, tester = get_analysis_results(conf, [conf['use_case']], RESULTS_DIR)

    if agg_fns is not None:
        res = aggregate_results(res, agg_fns, target_agg_result_ids)
        display_uc = [conf_creator.use_case_map[conf['use_case']]]
        plot_agg_results(res, target_cats=categories, xticks=display_uc, vmin=-0.5, vmax=0.5, agg=False,
                         plot_type=plot_type, save_path=save_path)

    else:
        plot_results(res, tester, target_cats=categories, plot_params=plot_params, plot_type=plot_type,
                     save_path=save_path)


def use_case_comparison_analysis(confs: list, plot_params: list, categories: list, compared_methods: list,
                                 agg_fns: list = None, target_agg_result_ids: list = None, only_diff: bool = True):
    assert isinstance(confs, list), "Wrong data type for parameter 'confs'."
    assert len(confs) > 1, "Wrong value for parameter 'confs'."

    for conf_idx1 in range(len(confs) - 1):
        conf1 = confs[conf_idx1]
        res1, tester = get_analysis_results(conf1, [conf1['use_case']], RESULTS_DIR)

        for conf_idx2 in range(conf_idx1 + 1, len(confs)):
            conf2 = confs[conf_idx2]
            res2, tester = get_analysis_results(conf2, [conf2['use_case']], RESULTS_DIR)

            cmp_vals = []
            for param in conf1:
                if conf1[param] != conf2[param]:
                    cmp_vals.append(conf1[param])
                    cmp_vals.append(conf2[param])
                    break

            assert list(res1.keys()) == list(res2.keys())
            assert len(res1) == 1

            if agg_fns is not None:
                res1 = aggregate_results(res1, agg_fns, target_agg_result_ids)
                res2 = aggregate_results(res2, agg_fns, target_agg_result_ids)

                cmp_res = cmp_agg_results(res1, res2, target_cats=categories)
                display_uc = [conf_creator.use_case_map[conf2['use_case']]]

                if only_diff:
                    res1 = res2 = res1_name = res2_name = None
                else:
                    res1_name = compared_methods[0]
                    res2_name = compared_methods[1]

                plot_agg_results(cmp_res, target_cats=categories, title_prefix=f'{cmp_vals[0]} vs {cmp_vals[1]}',
                                 xticks=display_uc, vmin=-0.5, vmax=0.5, res1=res1, res2=res2, res1_name=res1_name,
                                 res2_name=res2_name)

            else:
                res1 = list(res1.values())[0]
                res2 = list(res2.values())[0]
                cmp_res = cmp_results(res1, res2)
                plot_comparison(res1, res2, cmp_res, tester, cmp_vals, target_cats=categories, plot_params=plot_params)


def benchmark_analysis(conf: dict, plot_params: list, categories: list, agg_fns: list = None,
                       target_agg_result_ids: list = None):
    assert isinstance(conf, dict), "Wrong data type for parameter 'conf'."
    assert isinstance(plot_params, list), "Wrong data type for parameter 'plot_params'."
    assert len(plot_params) > 0, "Empty plot params."
    assert isinstance(categories, list), "Wrong data type for parameter 'categories'."
    assert len(categories) > 0, "Empty categories."
    if agg_fns is not None:
        assert isinstance(agg_fns, list), "Wrong data type for parameter 'agg_fns'."
        assert len(agg_fns) > 0, "Empty aggregation functions."
    if target_agg_result_ids is not None:
        assert isinstance(target_agg_result_ids, list), "Wrong data type for parameter 'target_agg_result_ids'."
        assert len(target_agg_result_ids) > 0, "Empty target aggregated results."

    use_cases = conf['use_case']
    assert isinstance(use_cases, list), "Wrong type for configuration use_case param."
    assert len(use_cases) > 0, "Empty use case list."
    res, tester = get_analysis_results(conf, use_cases, RESULTS_DIR)

    if agg_fns is not None:
        res = aggregate_results(res, agg_fns, target_agg_result_ids)
        display_uc = [conf_creator.use_case_map[uc] for uc in conf_creator.conf_template['use_case']]
        plot_agg_results(res, target_cats=categories, xticks=display_uc, vmin=-0.5, vmax=0.5, agg=True)
    else:
        plot_benchmark_results(res, tester, use_cases, target_cats=categories, plot_params=plot_params)


def benchmark_comparison_analysis(confs: list, plot_params: list, categories: list, compared_methods: list,
                                  agg_fns: list = None, target_agg_result_ids: list = None, only_diff: bool = True,
                                  save_path: str = None):
    assert isinstance(confs, list), "Wrong data type for parameter 'confs'."
    assert len(confs) > 1, "Wrong value for parameter 'confs'."

    for conf in confs:
        assert isinstance(conf['use_case'], list), "Wrong type for configuration use_case param."
        assert len(conf['use_case']) > 0, "Empty use case list."
    for idx in range(len(confs) - 1):
        assert confs[idx]['use_case'] == confs[idx + 1]['use_case'], "Use cases not equal."
    use_cases = confs[0]['use_case']

    assert isinstance(compared_methods, list)
    assert len(compared_methods) == 2
    assert all([isinstance(p, str) for p in compared_methods])

    for conf_idx1 in range(len(confs) - 1):
        conf1 = confs[conf_idx1]
        res1, tester = get_analysis_results(conf1, use_cases, RESULTS_DIR)

        for conf_idx2 in range(conf_idx1 + 1, len(confs)):
            conf2 = confs[conf_idx2]
            res2, tester = get_analysis_results(conf2, use_cases, RESULTS_DIR)

            cmp_vals = []
            for param in conf1:
                if conf1[param] != conf2[param]:
                    cmp_vals.append(conf1[param])
                    cmp_vals.append(conf2[param])
                    break

            print(conf1)
            print(conf2)

            if agg_fns is not None:
                res1 = aggregate_results(res1, agg_fns, target_agg_result_ids)
                res2 = aggregate_results(res2, agg_fns, target_agg_result_ids)

                cmp_res = cmp_agg_results(res1, res2, target_cats=categories)
                display_uc = [conf_creator.use_case_map[uc] for uc in conf_creator.conf_template['use_case']]

                if only_diff:
                    res1 = res2 = res1_name = res2_name = None
                else:
                    res1_name = compared_methods[0]
                    res2_name = compared_methods[1]

                plot_agg_results(cmp_res, target_cats=categories, title_prefix=f'{cmp_vals[0]} vs {cmp_vals[1]}',
                                 xticks=display_uc, agg=True, vmin=-0.5, vmax=0.5, res1=res1, res2=res2,
                                 res1_name=res1_name, res2_name=res2_name, save_path=save_path)

            else:
                cmp_res = cmp_benchmark_results(res1, res2)
                plot_benchmark_results(cmp_res, tester, use_cases, target_cats=categories,
                                       title_prefix=f'{cmp_vals[0]} {cmp_vals[1]}', plot_params=plot_params, vmin=-0.5,
                                       vmax=0.5, save_path=save_path)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Elaborate the results of the analysis of the attention weights')

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
    analysis_results = ['match_attr_attn_loc', 'match_attr_attn_over_mean',
                        'avg_attr_attn', 'attr_attn_3_last', 'attr_attn_last_1',
                        'attr_attn_last_2', 'attr_attn_last_3',
                        'avg_attr_attn_3_last', 'avg_attr_attn_last_1',
                        'avg_attr_attn_last_2', 'avg_attr_attn_last_3']
    available_categories = BinaryClassificationResultsAggregator.categories
    parser.add_argument('-attn_tester', '--attn_tester', required=True, choices=['attr_tester', 'attr_pattern_tester'],
                        help='method for analysing the extracted attention weights')
    parser.add_argument('-attn_tester_ignore_special', '--attn_tester_ignore_special', default=True,
                        type=lambda x: bool(distutils.util.strtobool(x)),
                        help='whether the attention weights analyzer has to ignore special tokens')
    parser.add_argument('-analysis_target', '--analysis_target', required=True, choices=['use_case', 'benchmark'],
                        help='whether apply the analysis to all the dataset of the benchmark or a single use case')
    parser.add_argument('-analysis_type', '--analysis_type', required=True, choices=['simple', 'comparison', 'multi'],
                        help='whether to compute attention weights analysis (i.e., the "simple" option) or compare \
                             previous analysis')
    parser.add_argument('-comparison_param', '--comparison_param', choices=['tok', 'fine_tune_method'],
                        help='the dimension where to compare previous analysis')
    parser.add_argument('-agg_fns', '--agg_fns', nargs='+', choices=['row_mean', 'row_std'], default=None,
                        help='optional aggregation function to apply to the results of the analysis')
    parser.add_argument('-target_agg_result_ids', '--target_agg_result_ids', nargs='+', default=None,
                        choices=analysis_results, help='optional name of the result variable to which apply the \
                        aggregations indicated by the --agg_funs option')
    parser.add_argument('-plot_params', '--plot_params', required=True, nargs='+', choices=analysis_results,
                        help='the name of the analysis result to plot')
    parser.add_argument('-data_categories', '--data_categories', default=['all'], nargs='+',
                        choices=available_categories,
                        help='the categories of records where to apply the attention analysis')

    args = parser.parse_args()

    use_cases = args.use_cases
    if use_cases == ['all']:
        use_cases = DM_USE_CASES

    conf = {
        'use_case': use_cases,
        'data_type': args.data_type,
        'model_name': args.bert_model,
        'tok': args.tok,
        'size': args.sample_size,
        'label_col': args.label_col,
        'left_prefix': args.left_prefix,
        'right_prefix': args.right_prefix,
        'max_len': args.max_len,
        'permute': args.permute,
        'verbose': args.verbose,
        'return_offset': args.return_offset,
        'fine_tune_method': args.fine_tune,
        'extractor': {
            'attn_extractor': args.attn_extractor,
            'attn_extr_params': {'special_tokens': args.special_tokens, 'agg_metric': args.agg_metric},
        },
        'tester': {
            'tester': args.attn_tester,
            'tester_params': {'ignore_special': args.attn_tester_ignore_special}
        },
    }

    extractor_name = conf['extractor']['attn_extractor']
    tester_name = conf['tester']['tester']
    agg_metric = conf['extractor']['attn_extr_params']['agg_metric']

    analysis_target = args.analysis_target
    analysis_type = args.analysis_type
    comparison_param = args.comparison_param
    agg_fns = args.agg_fns
    target_agg_result_ids = args.target_agg_result_ids
    plot_params = args.plot_params
    categories = args.data_categories
    conf_creator = ConfCreator()
    use_case_map = conf_creator.use_case_map

    if analysis_target == 'use_case':

        assert len(conf['use_case']) == 1
        assert conf['use_case'] != ['all']

        if analysis_type == 'simple':
            use_case_analysis(conf, plot_params, categories, agg_fns, target_agg_result_ids, plot_type='advanced')

        elif analysis_type == 'comparison':

            if comparison_param == 'fine_tune_method':
                compared_methods = ['Pre-training', 'Fine-tuning']
            elif comparison_param == 'tok':
                compared_methods = ['Attr-pair', 'Sent-pair']
            else:
                raise ValueError("Wrong comparison param.")

            confs = conf_creator.get_confs(conf, [comparison_param])
            use_case_comparison_analysis(confs, plot_params, categories, compared_methods, agg_fns,
                                         target_agg_result_ids, only_diff=False)

        else:
            raise NotImplementedError()

    elif analysis_target == 'benchmark':

        bench_conf = conf.copy()
        bench_conf['use_case'] = conf_creator.conf_template['use_case']

        if analysis_type == 'simple':

            benchmark_analysis(bench_conf, plot_params, categories, agg_fns, target_agg_result_ids)

        elif analysis_type == 'multi':

            assert agg_fns is None
            assert target_agg_result_ids is None
            assert len(categories) == 1

            template_file_name = '{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(conf['data_type'], extractor_name,
                                                                           tester_name,
                                                                           conf['fine_tune_method'], conf['permute'],
                                                                           conf['tok'],
                                                                           conf['size'], analysis_target, analysis_type,
                                                                           agg_metric, categories[0])

            confs = conf_creator.get_confs(conf, ['use_case'])
            imgs = []
            for conf in confs:
                uc = conf['use_case']

                out_file = os.path.join(RESULTS_DIR, uc, f'PLOT_{uc}_{template_file_name}')
                Path(os.path.join(RESULTS_DIR, uc)).mkdir(parents=True, exist_ok=True)
                use_case_analysis(conf, plot_params, categories, agg_fns, target_agg_result_ids, plot_type='advanced',
                                  save_path=out_file)
                imgs.append(mpimg.imread(f'{out_file}_{plot_params[0]}.png'))

            save_path = os.path.join(RESULTS_DIR, f'GRID_PLOT_{template_file_name}_{plot_params[0]}.pdf')
            plot_images_grid(imgs, nrows=3, ncols=4, save_path=save_path)

        elif analysis_type == 'comparison':

            template_file_name = '{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(conf['data_type'], extractor_name, tester_name,
                                                                     conf['permute'], conf['size'], agg_metric,
                                                                     analysis_target, analysis_type, comparison_param)

            if comparison_param == 'fine_tune_method':
                compared_methods = ['Pre-training', 'Fine-tuning']
                template_file_name = f'{template_file_name}_{conf["tok"]}'
            elif comparison_param == 'tok':
                compared_methods = ['Attr-pair', 'Sent-pair']
                template_file_name = f'{template_file_name}_{conf["fine_tune_method"]}'
            else:
                raise ValueError("Wrong comparison param.")

            bench_confs = conf_creator.get_confs(bench_conf, [comparison_param])

            save_path = os.path.join(RESULTS_DIR, f'PLOT_LOC_{template_file_name}_{plot_params[0]}.pdf')
            benchmark_comparison_analysis(bench_confs, plot_params, categories, compared_methods, agg_fns,
                                          target_agg_result_ids, only_diff=False, save_path=save_path)

        else:
            raise NotImplementedError()

    else:
        raise NotImplementedError()
