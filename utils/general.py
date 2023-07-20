import os
import pandas as pd
from transformers import AutoModel, AutoModelForSequenceClassification
from pathlib import Path
from collections import OrderedDict
from tqdm import tqdm

from utils.data_collector import DataCollector, DataCollectorWDC
from core.data_models.em_dataset import EMDataset
from utils.data_selection import Sampler
from core.attention.extractors import AttributeAttentionExtractor, WordAttentionExtractor, AttentionExtractor
from core.attention.testers import GenericAttributeAttentionTest, AttributeAttentionPatternFreqTest
from core.attention.analyzers import AttentionMapAnalyzer
from utils.nlp import get_pos_tag


PROJECT_DIR = Path(__file__).parent.parent
MODELS_DIR = os.path.join(PROJECT_DIR, 'results', 'models')


def get_use_case(use_case: str, bench='dm'):

    if bench == 'dm':
        data_collector = DataCollector()
    elif bench == 'wdc':
        data_collector = DataCollectorWDC()
    else:
        raise ValueError("Benchmark not found!")

    use_case_dir = data_collector.get_data(use_case)

    return use_case_dir


def get_dataset(conf: dict):
    assert isinstance(conf, dict), "Wrong data type for parameter 'conf'."
    params = ['use_case', 'data_type', 'model_name', 'tok', 'label_col', 'left_prefix', 'right_prefix', 'max_len',
              'verbose', 'permute', 'typeMask', 'columnMask']
    assert all([p in conf for p in params])
    assert isinstance(conf['data_type'], str), "Wrong data type for parameter 'data_type'."
    assert conf['data_type'] in ['train', 'test', 'valid'], "Wrong value for parameter 'data_type'."

    use_case = conf['use_case']
    data_type = conf['data_type']
    model_name = conf['model_name']
    tok = conf['tok']
    label_col = conf['label_col']
    left_prefix = conf['left_prefix']
    right_prefix = conf['right_prefix']
    max_len = conf['max_len']
    verbose = conf['verbose']
    permute = conf['permute']
    typeMask = conf['typeMask']
    columnMask = conf['columnMask']
    return_offset = False
    if 'return_offset' in conf:
        return_offset = conf['return_offset']
    bench = conf.get('bench')
    sem_emb_model = conf.get('sem_emb_model')

    if bench is None:
        use_case_data_dir = get_use_case(use_case)
    else:
        use_case_data_dir = get_use_case(use_case, bench=bench)

    if data_type == 'train':
        dataset_path = os.path.join(use_case_data_dir, "train.csv")
    elif data_type == 'test':
        dataset_path = os.path.join(use_case_data_dir, "test.csv")
    else:
        dataset_path = os.path.join(use_case_data_dir, "valid.csv")

    data = pd.read_csv(dataset_path)
    dataset = EMDataset(data, model_name, tokenization=tok, label_col=label_col, left_prefix=left_prefix,
                        right_prefix=right_prefix, max_len=max_len, verbose=verbose, permute=permute, typeMask=typeMask,
                        columnMask=columnMask, return_offset=return_offset, sem_emb_model=sem_emb_model)

    return dataset


def get_sample(dataset: EMDataset, sampler_conf: dict):
    assert isinstance(sampler_conf, dict), "Wrong data type for parameter 'sampler_conf'."
    params = ['size', 'target_class', 'permute', 'seeds']
    assert all([p in sampler_conf for p in params]), "Wrong value for parameter 'sampler_conf'."

    size = sampler_conf['size']
    target_class = sampler_conf['target_class']
    permute = sampler_conf['permute']
    seeds = sampler_conf['seeds']
    assert isinstance(target_class, (str, int)), "Wrong data type for parameter 'target_class'."
    assert target_class in ['both', 0, 1], "Wrong value for parameter 'target_class'."
    assert isinstance(seeds, list), "Wrong data type for parameter 'seeds'."
    assert len(seeds) == 2, "Wrong value for parameter 'seeds'."

    sampler = Sampler(dataset, permute=permute)

    if target_class == 'both':
        sample = sampler.get_balanced_data(size=size, seeds=seeds)
    elif target_class == 0:
        sample = sampler.get_non_match_data(size=size, seed=seeds[0])
    else:  # target_class = 1
        sample = sampler.get_match_data(size=size, seed=seeds[1])

    return sample


def get_model(model_name: str, fine_tune: bool = False, model_path: str = None):
    assert isinstance(model_name, str), "Wrong data type for parameter 'model_name'."
    assert isinstance(fine_tune, bool), "Wrong data type for parameter 'fine_tune'."
    if fine_tune:
        assert model_path is not None, "If 'fine_tune' is not null, provide a model path."
    if model_path is not None:
        assert isinstance(model_path, str), "Wrong data type for parameter 'model_path'."
        found = False
        if os.path.exists(model_path):
            found = True
        assert found, "Wrong value for parameter 'model_path'."

    if not fine_tune:
        model = AutoModel.from_pretrained(model_name, output_hidden_states=True, output_attentions=True)
    else:
        # if fine_tune == 'simple':
        #     model = AutoModelForSequenceClassification.from_pretrained(model_path)
        # else:  # fine_tune = 'advanced':
        #     model = MatcherTransformer.load_from_checkpoint(checkpoint_path=model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)

    return model


def get_extractors(extractor_params: dict):
    assert isinstance(extractor_params, dict), "Wrong data type for parameter 'extractor_params'."
    available_extractors = ['attr_extractor', 'word_extractor', 'token_extractor']
    for ex in extractor_params:
        assert ex in available_extractors, f"Wrong value for parameter 'extractor_params' ({available_extractors})."
        assert isinstance(extractor_params[ex], dict), "Wrong value for parameter 'extractor_params'."

    extractors = []
    for extractor_name in extractor_params:

        extractor_param = extractor_params[extractor_name]
        params = ['dataset', 'model']
        assert all([p in extractor_param for p in params]), "Wrong value for attr_extractor."

        dataset = extractor_param['dataset']
        model = extractor_param['model']
        other_params = {}
        for p in extractor_param:
            if p not in params:
                other_params[p] = extractor_param[p]

        if extractor_name in 'attr_extractor':
            attn_extractor = AttributeAttentionExtractor(dataset, model, **other_params)
        elif extractor_name in 'word_extractor':
            attn_extractor = WordAttentionExtractor(dataset, model, **other_params)
        else:   # token extractor
            attn_extractor = AttentionExtractor(dataset, model, **other_params)

        extractors.append(attn_extractor)

    return extractors


def get_testers(tester_params: dict):
    assert isinstance(tester_params, dict), "Wrong data type for parameter 'tester_params'."
    available_testers = ['attr_tester', 'attr_pattern_tester']
    for t in tester_params:
        assert t in available_testers, f"Wrong value for parameter 'tester_params' ({available_testers})."
        assert isinstance(tester_params[t], dict), "Wrong value for parameter 'tester_params'."

    testers = []
    for tester_name in tester_params:

        if tester_name == 'attr_tester':

            tester_param = tester_params[tester_name]
            params = ['permute', 'model_attention_grid', 'ignore_special']
            assert all([p in params for p in tester_param]), "Wrong value for attr_tester."

            attn_tester = GenericAttributeAttentionTest(**tester_param)

        elif tester_name == 'attr_pattern_tester':

            tester_param = tester_params[tester_name]
            params = ['ignore_special']
            assert all([p in params for p in tester_param]), "Wrong value for attr_pattern_tester."

            attn_tester = AttributeAttentionPatternFreqTest(**tester_param)

        else:
            raise NotImplementedError()

        testers.append(attn_tester)

    return testers


def get_analyzers(extractor_params: dict, tester_params: dict, analyzer_params: dict):
    extractors = get_extractors(extractor_params)
    testers = get_testers(tester_params)

    analyzers = [AttentionMapAnalyzer(extractor, testers, **analyzer_params) for extractor in extractors]

    return extractors, testers, analyzers


def get_pipeline(conf):
    assert isinstance(conf, dict), "Wrong data type for parameter 'conf'."
    params = ['use_case', 'data_type', 'permute', 'model_name', 'tok', 'label_col', 'left_prefix', 'right_prefix',
              'max_len', 'verbose', 'size', 'target_class', 'seeds', 'fine_tune_method', 'extractor', 'tester',
              'analyzer_params']
    assert all([p in conf for p in params]), "Wrong value for parameter 'conf'."

    # get dataset
    dataset = get_dataset(conf)

    # get sample
    sample = get_sample(dataset, conf)

    # get model
    if conf['fine_tune_method']:
        model_path = os.path.join(MODELS_DIR, f"{conf['use_case']}_{conf['tok']}_tuned")
    else:
        model_path = None
    model = get_model(conf['model_name'], conf['fine_tune_method'], model_path)

    # prepare extractor params
    extractor_params = {}
    extractor_name = conf['extractor']['attn_extractor']
    if extractor_name == 'attr_extractor':
        extractor_param = {
            'dataset': sample,
            'model': model,
        }
        extractor_param.update(conf['extractor']['attn_extr_params'])
        extractor_params[extractor_name] = extractor_param
    else:
        raise ValueError("Wrong value for parameter 'extractor_names'.")

    # prepare testers params
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
        tester_params[tester_name] = conf['tester']['tester_params'].copy()
    else:
        raise ValueError("Wrong tester name.")

    return get_analyzers(extractor_params, tester_params, conf['analyzer_params'])


def get_use_case_avg_attr_len(df, text_unit='char', pair_mode=False, left_prefix='left_', right_prefix='right_'):
    if pair_mode:
        left_df = df[[c for c in df.columns if c.startswith(left_prefix)]]
        left_df.columns = [c.replace(left_prefix, "") for c in left_df.columns]
        right_df = df[[c for c in df.columns if c.startswith(right_prefix)]]
        right_df.columns = [c.replace(right_prefix, "") for c in right_df.columns]
        df = pd.concat([left_df, right_df], axis=0)

    # ignore non relevant attribute
    drop_attrs = []
    for attr in df.columns:
        if 'id' in attr or 'label' in attr:
            drop_attrs.append(attr)
    df.drop(drop_attrs, axis=1, inplace=True)
    if text_unit == 'char':
        stats = df.applymap(lambda x: 0 if pd.isnull(x) else len(str(x))).mean(axis=0)
    elif text_unit == 'word':
        stats = df.applymap(lambda x: 0 if pd.isnull(x) else len(str(x).split())).mean(axis=0)
    else:
        raise ValueError("Wrong text unit.")

    stats = pd.Series(stats.map(lambda x: int(x)).to_dict(OrderedDict))

    return stats


def get_benchmark_avg_attr_len(use_cases, conf, sampler_conf, pair_mode=False, text_unit='char'):
    dfs = []
    for use_case in use_cases:
        uc_conf = conf.copy()
        uc_conf['use_case'] = use_case
        dataset = get_dataset(uc_conf)
        uc_sampler_conf = sampler_conf.copy()
        uc_sampler_conf['permute'] = uc_conf['permute']
        sample = get_sample(dataset, uc_sampler_conf).get_complete_data()
        dfs.append(sample)

    avg_attr_len = {}
    for uc_idx in range(len(use_cases)):
        uc = use_cases[uc_idx]
        df = dfs[uc_idx]
        avg_attr_len[uc] = get_use_case_avg_attr_len(df, text_unit=text_unit, pair_mode=pair_mode,
                                                     left_prefix=conf['left_prefix'], right_prefix=conf['right_prefix'])

    return avg_attr_len


def get_use_case_pos_tag_distr(df, pos_model):
    pos_tags_distr = {}
    num_words = 0

    for idx in tqdm(range(len(df))):
        row = df.iloc[idx]
        row_text = ''
        for attr, val in row.iteritems():
            if not pd.isnull(val):
                row_text += f'{str(val)}'

        row_text = pos_model(row_text)

        num_words += len(row_text)
        for word_idx, word in enumerate(row_text):
            pos_tag = get_pos_tag(word)
            if pos_tag not in pos_tags_distr:
                pos_tags_distr[pos_tag] = 1
            else:
                pos_tags_distr[pos_tag] += 1

    norm_pos_tags = {}
    for k in pos_tags_distr:
        norm_pos_tags[k] = pos_tags_distr[k] / num_words

    return pd.Series(norm_pos_tags)


def get_benchmark_pos_tag_distr(use_cases, conf, sampler_conf, pos_model):
    dfs = []
    for use_case in use_cases:
        uc_conf = conf.copy()
        uc_conf['use_case'] = use_case
        dataset = get_dataset(uc_conf)
        uc_sampler_conf = sampler_conf.copy()
        uc_sampler_conf['permute'] = uc_conf['permute']
        sample = get_sample(dataset, uc_sampler_conf).get_complete_data()
        dfs.append(sample)

    pos_tag_distr = {}
    tags = ['TEXT', 'PUNCT', 'NUM&SYM', 'CONN']
    for uc_idx in range(len(use_cases)):
        uc = use_cases[uc_idx]
        df = dfs[uc_idx]
        pos_tag_distr[uc] = get_use_case_pos_tag_distr(df, pos_model)

    return pd.DataFrame.from_dict(pos_tag_distr, orient='index')[tags]
