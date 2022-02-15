import os
import pickle
from utils.general import get_dataset, get_model, get_sample, get_extractors, get_testers
from core.attention.extractors import AttributeAttentionExtractor, WordAttentionExtractor, AttentionExtractor


def get_attn_extractor(conf: dict, sampler_conf: dict, fine_tune: bool, attn_params: dict, models_dir: str):
    assert isinstance(attn_params, dict)
    assert 'attn_extractor' in attn_params
    assert 'attn_extr_params' in attn_params
    assert attn_params['attn_extractor'] in ['attr_extractor', 'word_extractor', 'token_extractor']
    assert isinstance(attn_params['attn_extr_params'], dict)

    dataset = get_dataset(conf)
    use_case = conf['use_case']
    tok = conf['tok']
    model_name = conf['model_name']

    if fine_tune is True:
        model_path = os.path.join(models_dir, f"{use_case}_{tok}_tuned")
    else:
        model_path = None
    model = get_model(model_name, fine_tune, model_path)

    complete_sampler_conf = sampler_conf.copy()
    complete_sampler_conf['permute'] = conf['permute']
    sample = get_sample(dataset, complete_sampler_conf)

    extractor_name = attn_params['attn_extractor']
    extractor_params = {
        'dataset': sample,
        'model': model,
    }
    extractor_params.update(attn_params['attn_extr_params'])
    attn_extractors = get_extractors({extractor_name: extractor_params})
    attn_extractor = attn_extractors[0]

    return attn_extractor


def load_saved_attn_data(use_case, conf, sampler_conf, fine_tune, attn_params, res_dir):
    tok = conf['tok']
    size = sampler_conf['size']
    extractor_name = attn_params['attn_extractor']
    params = '_'.join([f'{x[0]}={x[1]}' for x in attn_params['attn_extr_params'].items()])
    out_fname = f"ATTN_{use_case}_{tok}_{size}_{fine_tune}_{extractor_name}_{params}"
    data_path = os.path.join(res_dir, use_case, out_fname)
    uc_attn = pickle.load(open(f"{data_path}.pkl", "rb"))
    if extractor_name == 'attr_extractor':
        AttributeAttentionExtractor.check_batch_attn_features(uc_attn)
    elif extractor_name == 'word_extractor':
        WordAttentionExtractor.check_batch_attn_features(uc_attn)
    elif extractor_name == 'token_extractor':
        AttentionExtractor.check_batch_attn_features(uc_attn)
    else:
        raise NotImplementedError()
    return uc_attn


def get_analysis_results(conf: dict, use_cases: list, results_dir: str):
    assert isinstance(conf, dict), "Wrong data type for parameter 'conf'."
    assert isinstance(use_cases, list), "Wrong data type for parameter 'use_cases'."
    assert len(use_cases) > 0, "Empty use case list."

    tester_params = {}
    tester_name = conf['tester']['tester']
    if tester_name == 'attr_tester':
        tester_param = {
            'permute': conf['permute'],
            'model_attention_grid': (12, 12),
        }
        tester_param.update(conf['tester']['tester_params'])
        tester_params[tester_name] = tester_param

    elif tester_name == 'attr_pattern_tester':
        tester_params[tester_name] = conf['tester']['tester_params']

    else:
        raise ValueError("Wrong tester name.")

    testers = get_testers(tester_params)

    assert len(testers) == 1
    tester = testers[0]

    results = {}
    for use_case in use_cases:

        out_path = os.path.join(results_dir, use_case)

        extractor_name = conf['extractor']['attn_extractor']
        extractor_params = '_'.join([f'{x[0]}={x[1]}' for x in conf['extractor']['attn_extr_params'].items()])
        if tester_name == 'attr_pattern_tester':
            tester_name = 'attr_patt_tester'
        tester_params = '_'.join([f'{x[0]}={x[1]}' for x in conf['tester']['tester_params'].items()])
        template_file_name = 'ANALYSIS_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_AVG.pickle'.format(use_case, conf['data_type'],
                                                                                        extractor_name, tester_name,
                                                                                        conf['fine_tune_method'],
                                                                                        conf['permute'], conf['tok'],
                                                                                        conf['size'], extractor_params,
                                                                                        tester_params)

        res_file = os.path.join(out_path, template_file_name)
        with open(res_file, 'rb') as f:
            res = pickle.load(f)
        results[use_case] = res

    # if len(results) > 0:
    #     if len(results) == 1:
    #         results = list(results.values())[0]

    return results, tester
