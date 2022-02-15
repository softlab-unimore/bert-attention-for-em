import os
import time
from transformers import AutoTokenizer
from utils.general import get_dataset, get_model, get_sample
from pathlib import Path
from core.explanation.gradient.extractors import EntityGradientExtractor
from multiprocessing import Process
from utils.test_utils import ConfCreator
from utils.data_collector import DM_USE_CASES
import argparse
import distutils.util


PROJECT_DIR = Path(__file__).parent.parent.parent
MODELS_DIR = os.path.join(PROJECT_DIR, 'results', 'models')
RESULT_DIR = os.path.join(PROJECT_DIR, 'results', 'grads')


def run_gradient_test(conf, sampler_conf, fine_tune, grad_params, models_dir, res_dir):
    assert isinstance(grad_params, dict), "Wrong data type for parameter 'grad_params'."
    params = ['text_unit', 'special_tokens']
    assert all([p in grad_params for p in params])

    dataset = get_dataset(conf)
    tok = conf['tok']
    model_name = conf['model_name']
    use_case = conf['use_case']

    grad_text_unit = grad_params['text_unit']
    grad_special_tokens = grad_params['special_tokens']

    if fine_tune is True:
        model_path = os.path.join(models_dir, f"{use_case}_{tok}_tuned")
    else:
        model_path = None
    model = get_model(model_name, fine_tune, model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

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
    out_dir = os.path.join(res_dir, use_case, out_fname)
    # save grads data
    entity_grad_extr.extract(sample, sample.max_len, out_path=out_dir)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Compute gradients for text units processed by BERT-based models')

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

    # Parameters for gradient computation
    parser.add_argument('-grad_text_units', '--grad_text_units', default='words', choices=['tokens', 'words', 'attrs'],
                        help='the typology of text unit where to compute gradients')
    parser.add_argument('-grad_special_tokens', '--grad_special_tokens', default=True,
                        type=lambda x: bool(distutils.util.strtobool(x)),
                        help='whether to consider or ignore special tokens in the gradient computation')
    parser.add_argument('-ft', '--fine_tune', default=True,
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
    assert fine_tune is True

    grad_conf = {
        'text_unit': args.grad_text_units,
        'special_tokens': args.grad_special_tokens,
    }

    multi_process = args.multi_process

    use_case_map = ConfCreator().use_case_map
    start = time.time()

    if args.multi_process is True:
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
        for use_case in use_cases:
            print(use_case)
            uc_conf = conf.copy()
            uc_conf['use_case'] = use_case
            run_gradient_test(uc_conf, sampler_conf, fine_tune, grad_conf, MODELS_DIR, RESULT_DIR)

    end = time.time()
    print(end - start)
