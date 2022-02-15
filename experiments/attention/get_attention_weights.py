from multiprocessing import Process
from utils.test_utils import ConfCreator
from utils.data_collector import DM_USE_CASES
import argparse
import distutils.util
from pathlib import Path
import os
import pickle
from utils.attention_utils import get_attn_extractor


PROJECT_DIR = Path(__file__).parent.parent.parent
MODELS_DIR = os.path.join(PROJECT_DIR, 'results', 'models')
RESULTS_DIR = os.path.join(PROJECT_DIR, 'results', 'attention')


def run_attn_extractor(conf: dict, sampler_conf: dict, fine_tune: bool, attn_params: dict, models_dir: str,
                       res_dir: str):
    use_case = conf['use_case']
    tok = conf['tok']
    extractor_name = attn_params['attn_extractor']

    attn_extractor = get_attn_extractor(conf, sampler_conf, fine_tune, attn_params, models_dir)
    results = attn_extractor.extract_all()

    params = '_'.join([f'{x[0]}={x[1]}' for x in attn_params['attn_extr_params'].items()])
    out_fname = f"ATTN_{use_case}_{tok}_{sampler_conf['size']}_{fine_tune}_{extractor_name}_{params}"
    out_file = os.path.join(res_dir, use_case, out_fname)
    out_dir_path = out_file.split(os.sep)
    out_dir = os.sep.join(out_dir_path[:-1])
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    with open(f'{out_file}.pkl', 'wb') as f:
        pickle.dump(results, f)

    return results


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Retrieve attention weights')

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
    parser.add_argument('-special_tokens', '--special_tokens', required=True, default=False,
                        type=lambda x: bool(distutils.util.strtobool(x)),
                        help='consider or ignore special tokens (e.g., [SEP], [CLS])')
    parser.add_argument('-agg_metric', '--agg_metric', required=True,  choices=['mean', 'max'],
                        help='method for aggregating the attention weights')
    parser.add_argument('-ft', '--fine_tune', default=False,
                        type=lambda x: bool(distutils.util.strtobool(x)),
                        help='boolean flag for selecting fine-tuned or pre-trained model')
    parser.add_argument('-multi_process', '--multi_process', default=False,
                        type=lambda x: bool(distutils.util.strtobool(x)),
                        help='multi process modality')

    args = parser.parse_args()

    use_cases = args.use_cases
    if use_cases == ['all']:
        use_cases = DM_USE_CASES

    conf = {
        'data_type': args.data_type,
        'model_name': args.bert_model,
        'tok': args.tok,
        'label_col': args.label_col,
        'left_prefix': args.left_prefix,
        'right_prefix': args.right_prefix,
        'max_len': args.max_len,
        'permute': args.permute,
        'verbose': args.verbose,
        'return_offset': args.return_offset,
    }

    sampler_conf = {
        'size': args.sample_size,
        'target_class': args.sample_target_class,
        'seeds': args.sample_seeds,
    }

    fine_tune = args.fine_tune

    attn_params = {
        'attn_extractor': args.attn_extractor,
        'attn_extr_params': {'special_tokens': args.special_tokens, 'agg_metric': args.agg_metric},
    }

    multi_process = args.multi_process

    use_case_map = ConfCreator().use_case_map

    if args.multi_process is True:
        processes = []
        for use_case in use_cases:
            uc_conf = conf.copy()
            uc_conf['use_case'] = use_case
            p = Process(target=run_attn_extractor,
                        args=(uc_conf, sampler_conf, fine_tune, attn_params, MODELS_DIR, RESULTS_DIR,))
            processes.append(p)

        for p in processes:
            p.start()

        for p in processes:
            p.join()

    else:
        for use_case in use_cases:
            print(use_case)

            uc_conf = conf.copy()
            uc_conf['use_case'] = use_case
            run_attn_extractor(uc_conf, sampler_conf, fine_tune, attn_params, MODELS_DIR, RESULTS_DIR)


# "Structured_DBLP-GoogleScholar", "Structured_DBLP-ACM", "Dirty_DBLP-ACM", "Dirty_DBLP-GoogleScholar", "Dirty_Walmart-Amazon", "Structured_Amazon-Google", "Structured_Walmart-Amazon", "Textual_Abt-Buy"