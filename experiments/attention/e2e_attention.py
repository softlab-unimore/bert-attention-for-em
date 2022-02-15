import os
from pathlib import Path
import pickle
from core.attention.analyzers import EntityToEntityAttentionAnalyzer
from utils.test_utils import ConfCreator
from utils.data_collector import DM_USE_CASES
import argparse
import distutils.util
from utils.attention_utils import load_saved_attn_data, get_attn_extractor


PROJECT_DIR = Path(__file__).parent.parent.parent
MODELS_DIR = os.path.join(PROJECT_DIR, 'results', 'models')
RESULTS_DIR = os.path.join(PROJECT_DIR, 'results', 'attention')


def compute_entity_to_entity_attention(use_cases, conf, sampler_conf, fine_tune, attn_params, res_dir,
                                       analysis_type='cross_entity', ignore_special=True, target_categories=None,
                                       extract_attention=False, precomputed=False, save=False):
    text_unit = attn_params['attn_extractor'].split('_')[0]
    tok = conf['tok']
    assert analysis_type in ['cross_entity', 'same_entity']

    uc_e2e_res = {}
    for use_case in use_cases:
        print(use_case)
        uc_e2e_res_path = f"E2E_{use_case}_{tok}_{sampler_conf['size']}_{fine_tune}_{attn_params['attn_extractor']}.pkl"
        uc_e2e_res_path = os.path.join(res_dir, use_case, uc_e2e_res_path)

        force_computation = not precomputed
        if precomputed:
            try:
                e2e_res = pickle.load(open(uc_e2e_res_path, "rb"))
            except Exception:
                print(f"No precomputed result found in {uc_e2e_res_path}.")
                force_computation = True

        if not precomputed or force_computation is True:
            if not extract_attention:
                uc_attn = load_saved_attn_data(use_case, conf, sampler_conf, fine_tune, attn_params, res_dir)
            else:
                uc_conf = conf.copy()
                uc_conf['use_case'] = use_case
                uc_attn = get_attn_extractor(uc_conf, sampler_conf, fine_tune, attn_params, MODELS_DIR)

            analyzer = EntityToEntityAttentionAnalyzer(uc_attn, text_unit=text_unit, tokenization=tok,
                                                       analysis_type=analysis_type, ignore_special=ignore_special,
                                                       target_categories=target_categories)
            e2e_res = analyzer.analyze_all()

            # free some memory
            uc_attn = None
            analyzer = None

            # merged_e2e_res = {}
            # cat_map = {'all': 'all', 'all_pred_pos': 'match', 'all_pred_neg': 'non_match'}
            # for cat in same_e2e_res:
            #     merged_e2e_res[f'same_{cat_map[cat]}'] = same_e2e_res[cat]
            # for cat in cross_e2e_res:
            #     merged_e2e_res[f'cross_{cat_map[cat]}'] = cross_e2e_res[cat]

        uc_e2e_res[use_case] = e2e_res

        if save:
            Path(os.path.join(res_dir, use_case)).mkdir(parents=True, exist_ok=True)
            with open(uc_e2e_res_path, 'wb') as f:
                pickle.dump(e2e_res, f)

    return uc_e2e_res


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Entity to entity attention analyzer')

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
    parser.add_argument('-ft', '--fine_tune', default=False,
                        type=lambda x: bool(distutils.util.strtobool(x)),
                        help='boolean flag for selecting fine-tuned or pre-trained model')

    # Parameters for data sampling
    parser.add_argument('-sample_size', '--sample_size', type=int,
                        help='size of the sample')
    parser.add_argument('-sample_target_class', '--sample_target_class', default='both', choices=['both', 0, 1],
                        help='classes to sample: match, non-match or both')
    parser.add_argument('-sample_seeds', '--sample_seeds', nargs='+', default=[42, 42],
                        help='seeds for each class sample. <seed non match> <seed match>')

    # Parameters for configuring the experiment
    parser.add_argument('-experiment', '--experiment', default='simple', choices=['simple', 'comparison'],
                        help='type of the experiment: 1) "simple": run a e2e experiment for a specific configuration, \
                             2) "comparison": compare two e2e experiments')
    parser.add_argument('-comparison', '--comparison', default=None, choices=['tune', 'tok', 'tune_tok'],
                        help='e2e configurations to be compared if --experiment is set to "comparison"')
    parser.add_argument('-precomputed', '--precomputed', default=False,
                        type=lambda x: bool(distutils.util.strtobool(x)),
                        help='avoid re-computing the e2e results')
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
        'attn_extractor': 'token_extractor',
        'attn_extr_params': {'special_tokens': True, 'agg_metric': 'mean'},
    }
    # target_categories = ['all_pred_pos', 'all_pred_neg']
    target_categories = ['all_pos', 'all_neg']

    experiment = args.experiment
    precomputed = args.precomputed
    comparison = args.comparison
    if experiment == 'comparison':
        assert comparison is not None
    small_plot = args.small_plot

    use_case_map = ConfCreator().use_case_map

    if experiment == 'simple':
        e2e_res = compute_entity_to_entity_attention(use_cases, conf, sampler_conf, fine_tune, attn_params,
                                                     RESULTS_DIR, analysis_type='cross_entity', ignore_special=True,
                                                     target_categories=target_categories)

        out_fname = f"PLOT_E2E_{conf['tok']}_{sampler_conf['size']}_{fine_tune}_{attn_params['attn_extractor']}.pdf"

    elif experiment == 'comparison':

        # print("Target categories set to 'all'")
        # target_categories = ['all']
        # comparison = 'tune_tok'  # 'tune', 'tok', 'tune_tok'

        if comparison == 'tune':
            new_fine_tune = False
            print("Load pre-trained data")
            pretrain_e2e_res = compute_entity_to_entity_attention(use_cases, conf, sampler_conf, new_fine_tune,
                                                                  attn_params, RESULTS_DIR,
                                                                  analysis_type='cross_entity', ignore_special=True,
                                                                  target_categories=['all', 'all_pos', 'all_neg'],
                                                                  precomputed=True)

            new_fine_tune = True
            print("Load fine-tuned data")
            finetune_e2e_res = compute_entity_to_entity_attention(use_cases, conf, sampler_conf, new_fine_tune,
                                                                  attn_params, RESULTS_DIR,
                                                                  analysis_type='cross_entity', ignore_special=True,
                                                                  target_categories=['all', 'all_pos', 'all_neg'],
                                                                  precomputed=True)
            # target_categories=['all_pos', 'all_neg'])

            e2e_res = {}
            for k in pretrain_e2e_res:
                e2e_res[k] = {'pre-trained': pretrain_e2e_res[k]['all']}
            for k in finetune_e2e_res:
                e2e_res[k]['fine-tuned'] = finetune_e2e_res[k]['all']
                # e2e_res[k]['finetune_match'] = finetune_e2e_res[k]['all_pos']
                # e2e_res[k]['finetune_non_match'] = finetune_e2e_res[k]['all_neg']

            out_fname = f"PLOT_E2E_CMP_{comparison}_{conf['tok']}_{sampler_conf['size']}_{attn_params['attn_extractor']}.pdf"

        elif comparison == 'tok':
            new_conf = conf.copy()
            new_conf['tok'] = 'sent_pair'
            print("Load sent_pair data")
            sentpair_e2e_res = compute_entity_to_entity_attention(use_cases, new_conf, sampler_conf, fine_tune,
                                                                  attn_params, RESULTS_DIR,
                                                                  analysis_type='cross_entity', ignore_special=True,
                                                                  target_categories=['all_pred_pos',
                                                                                     'all_pred_neg'],
                                                                  precomputed=False)

            new_conf['tok'] = 'attr_pair'
            print("Load attr_pair data")
            attrpair_e2e_res = compute_entity_to_entity_attention(use_cases, new_conf, sampler_conf, fine_tune,
                                                                  attn_params, RESULTS_DIR,
                                                                  analysis_type='cross_entity', ignore_special=True,
                                                                  target_categories=['all_pred_pos',
                                                                                     'all_pred_neg'],
                                                                  precomputed=False)

            e2e_res = {}
            for k in sentpair_e2e_res:
                e2e_res[k] = {'sentpair_match': sentpair_e2e_res[k]['all_pred_pos']}
                e2e_res[k]['sentpair_non_match'] = sentpair_e2e_res[k]['all_pred_neg']
            for k in attrpair_e2e_res:
                e2e_res[k]['attrpair_match'] = attrpair_e2e_res[k]['all_pred_pos']
                e2e_res[k]['attrpair_non_match'] = attrpair_e2e_res[k]['all_pred_neg']
            out_fname = f"PLOT_E2E_CMP_{comparison}_{sampler_conf['size']}_{fine_tune}_{attn_params['attn_extractor']}.pdf"

        elif comparison == 'tune_tok':
            new_conf = conf.copy()
            new_conf['tok'] = 'sent_pair'
            new_fine_tune = False
            print("Load pretrain sent_pair data")
            pretrain_sent_e2e_res = compute_entity_to_entity_attention(use_cases, new_conf, sampler_conf,
                                                                       new_fine_tune, attn_params, RESULTS_DIR,
                                                                       analysis_type='cross_entity',
                                                                       ignore_special=True,
                                                                       target_categories=['all', 'all_pos',
                                                                                          'all_neg'],
                                                                       precomputed=True)
            # extract_attention=True, save=True)
            # target_categories=['all'])

            new_conf = conf.copy()
            new_conf['tok'] = 'attr_pair'
            print("Load pretrain attr_pair data")
            pretrain_attr_e2e_res = compute_entity_to_entity_attention(use_cases, new_conf, sampler_conf,
                                                                       new_fine_tune, attn_params, RESULTS_DIR,
                                                                       analysis_type='cross_entity',
                                                                       ignore_special=True,
                                                                       target_categories=['all', 'all_pos',
                                                                                          'all_neg'],
                                                                       precomputed=True)
            # extract_attention=True, save=True)
            # target_categories=['all'])

            new_fine_tune = True
            new_conf['tok'] = 'sent_pair'
            print("Load finetuned sent_pair data")
            finetune_sent_e2e_res = compute_entity_to_entity_attention(use_cases, new_conf, sampler_conf,
                                                                       new_fine_tune, attn_params, RESULTS_DIR,
                                                                       analysis_type='cross_entity',
                                                                       ignore_special=True,
                                                                       target_categories=['all', 'all_pos',
                                                                                          'all_neg'],
                                                                       precomputed=True)
            # extract_attention=True, save=True)
            # target_categories=['all'])

            new_conf['tok'] = 'attr_pair'
            print("Load finetuned attr_pair data")
            finetune_attr_e2e_res = compute_entity_to_entity_attention(use_cases, new_conf, sampler_conf,
                                                                       new_fine_tune, attn_params, RESULTS_DIR,
                                                                       analysis_type='cross_entity',
                                                                       ignore_special=True,
                                                                       target_categories=['all', 'all_pos',
                                                                                          'all_neg'],
                                                                       precomputed=True)
            # extract_attention=True, save=True)
            # target_categories=['all'])

            e2e_res = {}
            for k in pretrain_sent_e2e_res:
                e2e_res[k] = {'pt_sent': pretrain_sent_e2e_res[k]['all']}
            for k in pretrain_attr_e2e_res:
                e2e_res[k]['pt_attr'] = pretrain_attr_e2e_res[k]['all']
            for k in finetune_sent_e2e_res:
                e2e_res[k]['ft_sent'] = finetune_sent_e2e_res[k]['all']
            for k in finetune_attr_e2e_res:
                e2e_res[k]['ft_attr'] = finetune_attr_e2e_res[k]['all']
            out_fname = f"PLOT_E2E_CMP_{comparison}_{sampler_conf['size']}_{attn_params['attn_extractor']}.pdf"

        else:
            raise ValueError("Wrong comparison variable.")

    else:
        raise ValueError("Wrong sub experiment.")

    out_file = os.path.join(RESULTS_DIR, out_fname)
    e2e_res = {use_case_map[k]: e2e_res[k] for k in e2e_res}
    EntityToEntityAttentionAnalyzer.plot_multi_entity_to_entity_attn(e2e_res, small_plot=small_plot,
                                                                     save_path=out_file)
