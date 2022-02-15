import os
from pathlib import Path
from utils.result_collector import BinaryClassificationResultsAggregator
from utils.test_utils import ConfCreator
from core.attention.analyzers import AttrToClsAttentionAnalyzer
from utils.attention_utils import load_saved_attn_data
from utils.data_collector import DM_USE_CASES
import argparse
import distutils.util

PROJECT_DIR = Path(__file__).parent.parent.parent
RESULTS_DIR = os.path.join(PROJECT_DIR, 'results', 'attention')


def compute_attn_to_cls(use_cases, conf, sampler_conf, fine_tune, attn_params, res_dir, target_categories):
    uc_grouped_attn = {}
    for use_case in use_cases:
        uc_attn = load_saved_attn_data(use_case, conf, sampler_conf, fine_tune, attn_params, res_dir)

        grouped_attn_res = AttrToClsAttentionAnalyzer.group_or_aggregate(uc_attn, target_categories=target_categories)
        uc_grouped_attn[use_case] = grouped_attn_res

    return uc_grouped_attn


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Analyse attention between the [CLS] token and dataset attributes')

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
    available_categories = BinaryClassificationResultsAggregator.categories
    parser.add_argument('-experiment', '--experiment', required=True, choices=['simple', 'comparison'],
                        help='whether to compute attention between the special token [CLS] and the dataset attributes \
                        (i.e., the "simple" option) or compare previous analysis')
    parser.add_argument('-comparison', '--comparison', choices=['tune', 'tok', 'tune_tok'],
                        help='the dimension where to compare previous analysis')
    parser.add_argument('-data_categories', '--data_categories', default=['all'], nargs='+',
                        choices=available_categories,
                        help='the categories of records where to apply the attention analysis')
    parser.add_argument('-small_plot', '--small_plot', default=False,
                        type=lambda x: bool(distutils.util.strtobool(x)),
                        help='boolean flag for generating small plot')

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

    experiment = args.experiment
    comparison = args.comparison
    target_categories = args.data_categories
    small_plot = args.small_plot
    conf_creator = ConfCreator()
    use_case_map = conf_creator.use_case_map

    assert attn_params['attn_extractor'] == 'attr_extractor'
    assert attn_params['attn_extr_params']["special_tokens"] is True
    extractor_name = attn_params['attn_extractor']
    extr_params = attn_params["attn_extr_params"]["agg_metric"]

    if experiment == 'simple':

        attn2cls_res = compute_attn_to_cls(use_cases, conf, sampler_conf, fine_tune, attn_params, RESULTS_DIR,
                                           target_categories)

        out_fname = f"PLOT_ATT2CLS_{conf['tok']}_{sampler_conf['size']}_{fine_tune}_{extractor_name}_{extr_params}.pdf"

    elif experiment == 'comparison':
        print("Target categories set to 'all'")
        assert target_categories == ['all']

        if comparison == 'tune':
            new_fine_tune = False
            print("Load pre-trained data")
            pretrain_att2cls_res = compute_attn_to_cls(use_cases, conf, sampler_conf, new_fine_tune, attn_params,
                                                       RESULTS_DIR, target_categories)

            new_fine_tune = True
            print("Load fine-tuned data")
            finetune_att2cls_res = compute_attn_to_cls(use_cases, conf, sampler_conf, new_fine_tune, attn_params,
                                                       RESULTS_DIR, target_categories)

            attn2cls_res = {}
            for k in pretrain_att2cls_res:
                attn2cls_res[k] = {'pre-trained': pretrain_att2cls_res[k]['all']}
            for k in finetune_att2cls_res:
                attn2cls_res[k]['fine-tuned'] = finetune_att2cls_res[k]['all']

            out_fname = f"PLOT_ATT2CLS_CMP_{comparison}_{conf['tok']}_{sampler_conf['size']}_{extractor_name}_{extr_params}.pdf"

        elif comparison == 'tok':
            new_conf = conf.copy()
            new_conf['tok'] = 'sent_pair'
            print("Load sent_pair data")
            sentpair_att2cls_res = compute_attn_to_cls(use_cases, new_conf, sampler_conf, fine_tune, attn_params,
                                                       RESULTS_DIR, target_categories)

            new_conf['tok'] = 'attr_pair'
            print("Load attr_pair data")
            attrpair_att2cls_res = compute_attn_to_cls(use_cases, new_conf, sampler_conf, fine_tune, attn_params,
                                                       RESULTS_DIR, target_categories)

            attn2cls_res = {}
            for k in sentpair_att2cls_res:
                attn2cls_res[k] = {'sent_pair': sentpair_att2cls_res[k]['all']}
            for k in attrpair_att2cls_res:
                attn2cls_res[k]['attr_pair'] = attrpair_att2cls_res[k]['all']
            out_fname = f"PLOT_ATT2CLS_CMP_{comparison}_{sampler_conf['size']}_{fine_tune}_{extractor_name}_{extr_params}.pdf"

        elif comparison == 'tune_tok':
            new_conf = conf.copy()
            new_conf['tok'] = 'sent_pair'
            new_fine_tune = False
            print("Load pretrain sent_pair data")
            pretrain_sent_att2cls_res = compute_attn_to_cls(use_cases, new_conf, sampler_conf, new_fine_tune,
                                                            attn_params, RESULTS_DIR, target_categories)

            new_conf = conf.copy()
            new_conf['tok'] = 'attr_pair'
            print("Load pretrain attr_pair data")
            pretrain_attr_att2cls_res = compute_attn_to_cls(use_cases, new_conf, sampler_conf, new_fine_tune,
                                                            attn_params, RESULTS_DIR, target_categories)

            new_fine_tune = True
            new_conf['tok'] = 'sent_pair'
            print("Load finetuned sent_pair data")
            finetuned_sent_att2cls_res = compute_attn_to_cls(use_cases, new_conf, sampler_conf, new_fine_tune,
                                                             attn_params, RESULTS_DIR, target_categories)

            new_conf['tok'] = 'attr_pair'
            print("Load finetuned attr_pair data")
            finetuned_attr_att2cls_res = compute_attn_to_cls(use_cases, new_conf, sampler_conf, new_fine_tune,
                                                             attn_params, RESULTS_DIR, target_categories)

            attn2cls_res = {}
            for k in pretrain_sent_att2cls_res:
                attn2cls_res[k] = {'pretrain_sent_pair': pretrain_sent_att2cls_res[k]['all']}
            for k in pretrain_attr_att2cls_res:
                attn2cls_res[k]['pretrain_attr_pair'] = pretrain_attr_att2cls_res[k]['all']
            for k in finetuned_sent_att2cls_res:
                attn2cls_res[k]['finetune_sent_pair'] = finetuned_sent_att2cls_res[k]['all']
            for k in finetuned_attr_att2cls_res:
                attn2cls_res[k]['finetune_attr_pair'] = finetuned_attr_att2cls_res[k]['all']
            out_fname = f"PLOT_ATT2CLS_CMP_{comparison}_{sampler_conf['size']}_{extractor_name}_{extr_params}.pdf"

        else:
            raise ValueError("Wrong comparison variable.")

    else:
        raise ValueError("Wrong experiment.")

    out_file = os.path.join(RESULTS_DIR, out_fname)

    attn2cls_res = {use_case_map[uc]: attn2cls_res[uc] for uc in attn2cls_res}
    AttrToClsAttentionAnalyzer.plot_multi_attr_to_cls_attn(attn2cls_res, small_plot=small_plot, save_path=out_file)
