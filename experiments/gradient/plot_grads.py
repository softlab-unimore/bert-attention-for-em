import os
import time
from transformers import AutoTokenizer
from utils.general import get_dataset, get_model, get_sample
from pathlib import Path
from core.explanation.gradient.extractors import EntityGradientExtractor
from multiprocessing import Process
from utils.test_utils import ConfCreator
from utils.data_collector import DM_USE_CASES
from utils.grad_utils import load_saved_grads_data, plot_multi_use_case_grads, plot_batch_grads
import argparse
import distutils.util


PROJECT_DIR = Path(__file__).parent.parent.parent
MODELS_DIR = os.path.join(PROJECT_DIR, 'results', 'models')
RESULT_DIR = os.path.join(PROJECT_DIR, 'results', 'grads')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Plot gradients computed over BERT-based models')

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

    # Parameters for gradient plot
    parser.add_argument('-grad_text_units', '--grad_text_units', default='words', choices=['tokens', 'words', 'attrs'],
                        help='the typology of text unit where to compute gradients')
    parser.add_argument('-grad_special_tokens', '--grad_special_tokens', default=True,
                        type=lambda x: bool(distutils.util.strtobool(x)),
                        help='whether to consider or ignore special tokens in the gradient computation')
    parser.add_argument('-ft', '--fine_tune', default=True,
                        type=lambda x: bool(distutils.util.strtobool(x)),
                        help='boolean flag for selecting fine-tuned or pre-trained model')
    parser.add_argument('-plot_grad_agg_metrics', '--plot_grad_agg_metrics', default=['max'], nargs='+',
                        choices=['sum', 'avg', 'median', 'max'],
                        help='the method for aggregating the gradients when plotting')
    parser.add_argument('-plot_ignore_special', '--plot_ignore_special', default=True,
                        type=lambda x: bool(distutils.util.strtobool(x)),
                        help='boolean flag for ignoring gradients associated to special tokens when plotting')
    parser.add_argument('-plot_type', '--plot_type', default='box', choices=['box', 'error'],
                        help='type of chart used for plotting the gradients')

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

    # plot grad conf
    plot_grad_agg_metrics = args.plot_grad_agg_metrics
    plot_ignore_special = args.plot_ignore_special
    plot_type = args.plot_type

    use_case_map = ConfCreator().use_case_map

    if grad_conf['text_unit'] == 'attrs':
        assert len(plot_grad_agg_metrics) == 1
        out_plot_name = os.path.join(RESULT_DIR,
                                     f"grad_{conf['tok']}_{sampler_conf['size']}_{grad_conf['text_unit']}_{plot_grad_agg_metrics[0]}.pdf")
        plot_multi_use_case_grads(conf, sampler_conf, fine_tune, grad_conf, use_cases, RESULT_DIR,
                                  grad_agg_metrics=plot_grad_agg_metrics, plot_type=plot_type,
                                  ignore_special=plot_ignore_special, out_plot_name=out_plot_name,
                                  use_case_map=use_case_map)
    else:
        for use_case in use_cases:
            uc_grad = load_saved_grads_data(use_case, conf, sampler_conf, fine_tune, grad_conf, RESULT_DIR)
            out_plot_name = os.path.join(RESULT_DIR, use_case,
                                         f"grad_{grad_conf['text_unit']}_{grad_conf['special_tokens']}")
            plot_batch_grads(uc_grad, 'all', title_prefix=use_case, out_plot_name=out_plot_name,
                             ignore_special=plot_ignore_special)

