from utils.general import get_dataset, get_model, get_sample, get_extractors, get_testers, get_analyzers
from core.attention.extractors import AttributeAttentionExtractor
import os
from pathlib import Path


PROJECT_DIR = Path(__file__).parent.parent.parent
MODELS_DIR = os.path.join(PROJECT_DIR, 'results', 'models')
RESULTS_DIR = os.path.join(PROJECT_DIR, 'results', 'attention')


def test_dataset(conf: dict):

    dataset = get_dataset(conf)
    verbose = conf['verbose']

    row = dataset[0]
    left_entity = None
    right_entity = None
    features = None

    if verbose:
        left_entity = row[0]
        right_entity = row[1]
        features = row[2]
    else:
        features = row

    assert features is not None

    if left_entity is not None:
        print(left_entity)

    if right_entity is not None:
        print(right_entity)

    row_text = dataset.tokenizer.convert_ids_to_tokens(features['input_ids'])
    row_label = features['labels']
    print(row_text)
    print(row_label)
    print("Num. sentences: {}".format(len(features['token_type_ids'].unique())))


def test_sampler(conf: dict, sampler_conf: dict):

    dataset = get_dataset(conf)

    complete_sampler_conf = sampler_conf.copy()
    complete_sampler_conf['permute'] = conf['permute']
    sample = get_sample(dataset, complete_sampler_conf)

    print("Num. samples: {}".format(len(sample)))

    first_row = sample[0]
    first_left_entity = first_row[0]
    first_right_entity = first_row[1]
    first_input_ids = first_row[2]['input_ids']
    first_row_text = dataset.tokenizer.convert_ids_to_tokens(first_input_ids)
    first_label = first_row[2]['labels']
    print("\nFIRST ROW")
    print(first_left_entity)
    print(first_right_entity)
    print(first_row_text)
    print(first_label)
    print("Num. sentences: {}".format(len(first_row[2]['token_type_ids'].unique())))

    last_row = sample[-1]
    last_left_entity = last_row[0]
    last_right_entity = last_row[1]
    last_input_ids = last_row[2]['input_ids']
    last_row_text = dataset.tokenizer.convert_ids_to_tokens(last_input_ids)
    last_label = last_row[2]['labels']
    print("\nLAST ROW")
    print(last_left_entity)
    print(last_right_entity)
    print(last_row_text)
    print(last_label)
    print("Num. sentences: {}".format(len(last_row[2]['token_type_ids'].unique())))


def test_attn_extractor(conf: dict, sampler_conf: dict, fine_tune: str, attn_extr_params: list):

    dataset = get_dataset(conf)
    use_case = conf['use_case']
    tok = conf['tok']
    model_name = conf['model_name']

    if fine_tune is not None:
        model_path = os.path.join(MODELS_DIR, fine_tune, f"{use_case}_{tok}_tuned")
    else:
        model_path = None
    model = get_model(model_name, fine_tune, model_path)

    complete_sampler_conf = sampler_conf.copy()
    complete_sampler_conf['permute'] = conf['permute']
    sample = get_sample(dataset, complete_sampler_conf)

    extractor_params = {}
    for attn_extr_param in attn_extr_params:

        extractor_name = attn_extr_param['attn_extractor']
        if extractor_name in ['attr_extractor', 'word_extractor', 'token_extractor']:
            extractor_param = {
                'dataset': sample,
                'model': model,
            }
            extractor_param.update(attn_extr_param['attn_extr_params'])
            extractor_params[extractor_name] = extractor_param
        else:
            raise ValueError("Wrong value for parameter 'extractor_names'.")

    attn_extractors = get_extractors(extractor_params)

    for attn_extractor in attn_extractors:
        print(type(attn_extractor))
        results = attn_extractor.extract_all()
        if isinstance(attn_extractor, AttributeAttentionExtractor):
            invalid_attn_maps = attn_extractor.get_num_invalid_attr_attn_maps()

        for idx, res in enumerate(results):
            print(f"Record#{idx}")
            left = res[0]
            right = res[1]
            features = res[2]
            if features['attns'] is None:
                print("Skip.")
                continue
            print(left)
            print(right)
            print(features.keys())
            print(features['tokens'])
            print(features['text_units'])
            print(features['attns'][0].shape[1:])
            print("-" * 10)

        if isinstance(attn_extractor, AttributeAttentionExtractor):
            print(f"Invalid attention maps: {invalid_attn_maps}/{len(results)}={(invalid_attn_maps/len(results))*100}")


def test_attn_tester(conf: dict, sampler_conf: dict, fine_tune: str, attn_extr_params: list, attn_tester_params: list):

    dataset = get_dataset(conf)
    use_case = conf['use_case']
    tok = conf['tok']
    model_name = conf['model_name']

    if fine_tune is not None:
        model_path = os.path.join(MODELS_DIR, fine_tune, f"{use_case}_{tok}_tuned")
    else:
        model_path = None
    model = get_model(model_name, fine_tune, model_path)

    complete_sampler_conf = sampler_conf.copy()
    complete_sampler_conf['permute'] = conf['permute']
    sample = get_sample(dataset, complete_sampler_conf)

    extractor_params = {}
    for attn_extr_param in attn_extr_params:

        extractor_name = attn_extr_param['attn_extractor']
        if extractor_name in ['attr_extractor', 'word_extractor', 'token_extractor']:
            extractor_param = {
                'dataset': sample,
                'model': model,
            }
            extractor_param.update(attn_extr_param['attn_extr_params'])
            extractor_params[extractor_name] = extractor_param
        else:
            raise ValueError("Wrong extractor name.")

    attn_extractors = get_extractors(extractor_params)

    tester_params = {}
    for attn_tester_param in attn_tester_params:

        tester_name = attn_tester_param['tester']
        if tester_name == 'attr_tester':
            tester_param = {
                'permute': conf['permute'],
                'model_attention_grid': (12, 12),
            }
            tester_param.update(attn_tester_param['tester_params'])
            tester_params[tester_name] = tester_param
        else:
            raise ValueError("Wrong tester name.")

    attn_testers = get_testers(tester_params)

    for attn_extractor in attn_extractors:

        print(type(attn_extractor))

        for idx, (left_entity, right_entity, attn_params) in enumerate(attn_extractor.extract_all()):

            print("\t", f"Row#{idx}")

            for tester_id, tester in enumerate(attn_testers):

                print("\t", "\t", type(tester))

                result = tester.test(left_entity, right_entity, attn_params)
                print(result.get_results().keys())
                
                
def test_attn_analyzer(conf: dict, sampler_conf: dict, fine_tune: str, attn_extr_params: list, attn_tester_params: list,
                       analyzer_params: dict = None):

    dataset = get_dataset(conf)
    use_case = conf['use_case']
    tok = conf['tok']
    model_name = conf['model_name']

    if fine_tune is not None:
        model_path = os.path.join(MODELS_DIR, fine_tune, f"{use_case}_{tok}_tuned")
    else:
        model_path = None
    model = get_model(model_name, fine_tune, model_path)

    complete_sampler_conf = sampler_conf.copy()
    complete_sampler_conf['permute'] = conf['permute']
    sample = get_sample(dataset, complete_sampler_conf)

    extractor_params = {}
    for attn_extr_param in attn_extr_params:

        extractor_name = attn_extr_param['attn_extractor']
        if extractor_name in ['attr_extractor', 'word_extractor', 'token_extractor']:
            extractor_param = {
                'dataset': sample,
                'model': model,
            }
            extractor_param.update(attn_extr_param['attn_extr_params'])
            extractor_params[extractor_name] = extractor_param
        else:
            raise ValueError("Wrong extractor name.")

    tester_params = {}
    for attn_tester_param in attn_tester_params:

        tester_name = attn_tester_param['tester']
        if tester_name == 'attr_tester':
            tester_param = {
                'permute': conf['permute'],
                'model_attention_grid': (12, 12),
            }
            tester_param.update(attn_tester_param['tester_params'])
            tester_params[tester_name] = tester_param
        else:
            raise ValueError("Wrong tester name.")

    _, _, analyzers = get_analyzers(extractor_params, tester_params, analyzer_params)

    for analyzer in analyzers:

        results = analyzer.analyze_all()

        print(results)


if __name__ == '__main__':
    use_cases = ["Structured_Fodors-Zagats", "Structured_DBLP-GoogleScholar", "Structured_DBLP-ACM",
                 "Structured_Amazon-Google", "Structured_Walmart-Amazon", "Structured_Beer",
                 "Structured_iTunes-Amazon", "Textual_Abt-Buy", "Dirty_iTunes-Amazon", "Dirty_DBLP-ACM",
                 "Dirty_DBLP-GoogleScholar", "Dirty_Walmart-Amazon"]

    # [BEGIN] INPUT PARAMS ---------------------------------------------------------------------------------------------
    conf = {
        'use_case': "Structured_Beer",
        'data_type': 'train',                       # 'train', 'test', 'valid'
        'model_name': 'bert-base-uncased',
        'tok': 'sent_pair',                         # 'sent_pair', 'attr', 'attr_pair'
        'label_col': 'label',
        'left_prefix': 'left_',
        'right_prefix': 'right_',
        'max_len': 128,
        'permute': False,
        'verbose': False,
    }

    sampler_conf = {
        'size': None,
        'target_class': 'both',                     # 'both', 0, 1
        'seeds': [42, 42],                          # [42 -> class 0, 42 -> class 1]
    }

    fine_tune = 'simple'  # None, 'simple', 'advanced'

    # attention extractor parameters
    attn_params = {
        'attn_extractor': 'attr_extractor',   # 'attr_extractor', 'word_extractor'
        'attn_extr_params': {'special_tokens': True, 'agg_metric': 'mean'}
    }

    # attention tester parameters
    attn_tester_params = {
        'tester': 'attr_tester',
        'tester_params': {'ignore_special': True},
    }

    # attention analyzer parameters
    params = '_'.join([f'{x[0]}={x[1]}' for x in attn_params['attn_extr_params'].items()])
    out_fname = f"ATTN_{conf['use_case']}_{conf['tok']}_{sampler_conf['size']}_{fine_tune}_{attn_params['attn_extractor']}_{params}"
    out_file = os.path.join(RESULTS_DIR, conf['use_case'], out_fname)
    attn_analyzer_params = {
        'pre_computed_attns': f'{out_file}.pkl'
    }
    # attn_analyzer_params = {
    #     'pre_computed_attns': None
    # }
    # [END] INPUT PARAMS -----------------------------------------------------------------------------------------------

    # TEST 1
    # test_dataset(conf)

    # TEST 2
    # test_sampler(conf, sampler_conf)

    # TEST 3
    # test_attn_extractor(conf, sampler_conf, fine_tune, [attn_params])

    # TEST 4
    # test_attn_tester(conf, sampler_conf, fine_tune, [attn_params], [attn_tester_params])
    
    # TEST 5
    test_attn_analyzer(conf, sampler_conf, fine_tune, [attn_params], [attn_tester_params], attn_analyzer_params)
