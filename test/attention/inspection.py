import os
from pathlib import Path

from utils.result_collector import TestResultCollector
from utils.plot import plot_layers_heads_attention, plot_left_to_right_heatmap
from utils.general import get_pipeline


PROJECT_DIR = Path(__file__).parent.parent.parent
MODELS_DIR = os.path.join(PROJECT_DIR, 'results', 'models')
RESULTS_DIR = os.path.join(PROJECT_DIR, 'results', 'attention')


def run_inspection(conf: dict, inspect_row_idx: int, save: bool):
    assert isinstance(conf, dict), "Wrong data type for parameter 'conf'."
    assert isinstance(inspect_row_idx, int), "Wrong data type for parameter 'inspect_row_idx'."
    assert isinstance(save, bool), "Wrong data type for parameter 'save'."

    use_case = conf['use_case']
    out_path = os.path.join(RESULTS_DIR, use_case)
    Path(out_path).mkdir(parents=True, exist_ok=True)

    extractor_name = conf['extractor']['attn_extractor']
    extractor_params = '_'.join([f'{x[0]}={x[1]}' for x in conf['extractor']['attn_extr_params'].items()])
    tester_name = conf['tester']['tester']
    tester_params = '_'.join([f'{x[0]}={x[1]}' for x in conf['tester']['tester_params'].items()])
    template_file_name = 'INSPECTION_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(use_case, conf['data_type'],
                                                                           extractor_name, tester_name,
                                                                           conf['fine_tune_method'],
                                                                           conf['permute'], conf['tok'],
                                                                           conf['size'], extractor_params,
                                                                           tester_params)

    extractors, testers, analyzers = get_pipeline(conf)
    attn_extractor = extractors[0]
    analyzer = analyzers[0]

    _, _, inspect_row_attns = attn_extractor.extract(inspect_row_idx)
    inspect_row_attns = inspect_row_attns['attns']

    inspect_row_results, label, pred, category, text_units = analyzer.analyze(inspect_row_idx)
    print("LABEL: {}".format(label))
    print("PRED: {}".format(pred))

    template_file_name += "_{}_{}_{}".format(inspect_row_idx, label, pred)

    params_to_inspect = {
        'attr_tester': ['match_attr_attn_loc'],
    }
    original_testers = [conf['tester']]
    for idx, tester in enumerate(testers):

        tester_name = original_testers[idx]['tester']
        test_params_to_inspect = params_to_inspect[tester_name]
        inspect_row_test_results = inspect_row_results[idx]

        if save:
            out_dir = out_path
            out_file_name_prefix = template_file_name
        else:
            out_dir = None
            out_file_name_prefix = None
        tester.plot(inspect_row_test_results, out_dir=out_dir, out_file_name_prefix=out_file_name_prefix)

        if isinstance(inspect_row_test_results, dict):
            for key in inspect_row_test_results:
                print(key)
                test_collector = inspect_row_test_results[key]
                res = test_collector.get_results()
                for param in test_params_to_inspect:
                    mask = res[param] > 0
                    print(param)

                    if save:
                        out_file_name = os.path.join(out_path, '{}_{}'.format(template_file_name, "lxh_attns.pdf"))
                    else:
                        out_file_name = None
                    # plot_layers_heads_attention(inspect_row_attns, mask=mask, out_file_name=out_file_name)
                    plot_left_to_right_heatmap(inspect_row_attns[3][0], vmin=0, vmax=1, is_annot=True,
                                               out_file_name=f'{use_case}_matching_pattern.pdf')

        elif isinstance(inspect_row_test_results, TestResultCollector):
            res = inspect_row_test_results.get_results()
            for param in test_params_to_inspect:
                print(param)
                mask = res[param] > 0

                if save:
                    out_file_name = os.path.join(out_path, '{}_{}'.format(template_file_name, "lxh_attns.pdf"))
                else:
                    out_file_name = None
                # plot_layers_heads_attention(inspect_row_attns, mask=mask, out_file_name=out_file_name)
                plot_left_to_right_heatmap(inspect_row_attns[3][0], vmin=0, vmax=1, is_annot=True,
                                           out_file_name=f'{use_case}_matching_pattern.pdf')


if __name__ == '__main__':
    conf = {
        'use_case': "Structured_Fodors-Zagats",
        'data_type': 'train',  # 'train', 'test', 'valid'
        'permute': False,
        'model_name': 'bert-base-uncased',
        'tok': 'sent_pair',  # 'sent_pair', 'attr', 'attr_pair'
        'label_col': 'label',
        'left_prefix': 'left_',
        'right_prefix': 'right_',
        'max_len': 128,
        'verbose': False,
        'size': None,
        'target_class': 'both',  # 'both', 0, 1
        'fine_tune_method': 'simple',  # None, 'simple', 'advanced'
        'extractor': {
            'attn_extractor': 'attr_extractor',  # word_extractor
            'attn_extr_params': {'special_tokens': False, 'agg_metric': 'mean'},
        },
        'tester': {
            'tester': 'attr_tester',
            'tester_params': {'ignore_special': True}
        },
        'analyzer_params': {'pre_computed_attns': None},
        'seeds': [42, 42]
    }

    if conf['fine_tune_method'] is None:
        assert conf['extractor']['att_extr_params']['special_tokens'] is False

    save = False
    inspect_row_idx = 0

    run_inspection(conf, inspect_row_idx, save)
