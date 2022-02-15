from utils.general import get_dataset, get_model, get_sample
from pathlib import Path
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoModel, AutoModelForSequenceClassification
from tqdm import tqdm
from utils.test_utils import ConfCreator
import seaborn as sns
from utils.data_collector import DM_USE_CASES
import argparse
import distutils.util

"""
Code adapted from https://github.com/text-machine-lab/dark-secrets-of-BERT/blob/master/visualize_attention.ipynb
"""

PROJECT_DIR = Path(__file__).parent.parent
MODELS_DIR = os.path.join(PROJECT_DIR, 'results', 'models')
RESULTS_DIR = os.path.join(PROJECT_DIR, 'results', 'attention')


def get_use_case_entity_pairs(conf, sampler_conf):
    dataset = get_dataset(conf)

    complete_sampler_conf = sampler_conf.copy()
    complete_sampler_conf['permute'] = conf['permute']
    sample = get_sample(dataset, complete_sampler_conf)

    return sample


def get_pt_ft_attention_sim(conf, sampler_conf, precomputed=False, save=False):

    uc_sim_maps = {}
    for uc in conf['use_case']:
        print("\n\n", uc)
        uc_conf = conf.copy()
        uc_conf['use_case'] = uc

        save_path = os.path.join(RESULTS_DIR, uc, 'attn_pt_ft_similarity.npy')
        compute = True
        if precomputed:
            if os.path.exists(save_path):
                print("Loading precomputed similarity map.")
                avg_uc_sim = np.load(save_path)
                compute = False
            else:
                print("No precomputed results are available.")
                compute = True

        if compute is True:

            print("Computing similarity map...")
            # Get data
            encoded_dataset = get_use_case_entity_pairs(conf=uc_conf, sampler_conf=sampler_conf)

            # Get pre-trained model
            pt_model = AutoModel.from_pretrained(conf['model_name'], output_attentions=True)

            # Get fine-tuned model
            ft_model_path = os.path.join(MODELS_DIR, f"{uc}_{conf['tok']}_tuned")
            ft_model = AutoModelForSequenceClassification.from_pretrained(ft_model_path, output_attentions=True)

            n_layers, n_heads = pt_model.config.num_hidden_layers, pt_model.config.num_attention_heads

            uc_sims = []
            for encoded_row in tqdm(encoded_dataset):
                features = encoded_row[2]
                input_ids = features['input_ids'].unsqueeze(0)
                attention_mask = features['attention_mask'].unsqueeze(0)
                token_type_ids = features['token_type_ids'].unsqueeze(0)
                with torch.no_grad():
                    pt_attn = pt_model(input_ids=input_ids, attention_mask=attention_mask,
                                       token_type_ids=token_type_ids)["attentions"]
                    ft_attn = ft_model(input_ids=input_ids, attention_mask=attention_mask,
                                       token_type_ids=token_type_ids)["attentions"]

                    pt_attn = torch.cat(pt_attn).view(n_layers, n_heads, -1)
                    ft_attn = torch.cat(ft_attn).view(n_layers, n_heads, -1)

                    sim_map = torch.nn.functional.cosine_similarity(pt_attn, ft_attn, dim=-1).detach().numpy()
                    uc_sims.append(sim_map)

            uc_sims = np.stack(uc_sims, axis=-1)
            avg_uc_sim = np.mean(uc_sims, axis=-1)
            if save:
                Path(os.path.join(RESULTS_DIR, uc)).mkdir(parents=True, exist_ok=True)
                np.save(save_path, avg_uc_sim)

        uc_sim_maps[uc] = avg_uc_sim

    sim_maps = {'all': np.mean(np.stack(list(uc_sim_maps.values()), axis=-1), axis=-1)}
    sim_maps.update(uc_sim_maps)

    return sim_maps


def plot_attention_sim(att_sim, ax=None, title=None, show_xlabel=True, show_ylabel=True, cbar=False, cbar_ax=None):

    show = False
    if ax is None:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 9))
        show = True

    sns.heatmap(att_sim, cmap='Blues_r', vmin=0, vmax=1, ax=ax, cbar=cbar, cbar_ax=cbar_ax)
    ax.grid(False)
    if title:
        ax.set_title(title, fontsize=16)
    if show_xlabel:
        ax.set_xlabel('Head', fontsize=14)
    if show_ylabel:
        ax.set_ylabel('Layer', fontsize=14)
    ax.yaxis.set_tick_params(rotation=0, labelsize=14)
    ax.xaxis.set_tick_params(rotation=0, labelsize=14)
    ax.set_xticklabels(range(1, att_sim.shape[0] + 1))
    ax.set_yticklabels(range(1, att_sim.shape[1] + 1))

    if show:
        plt.show()


def plot_attention_sim_maps(sim_maps, save_path=None):

    use_case_map = ConfCreator().use_case_map

    ncols = 6
    nrows = 2
    figsize = (18, 6)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, sharey=True, sharex=True)
    axes = axes.flat
    # colorbar_ax = fig.add_axes([1.02, .377, .03, .27])
    colorbar_ax = fig.add_axes([1.01, .113, .025, .82])

    # loop over the use cases
    for idx, use_case in enumerate(sim_maps):
        ax = axes[idx]
        uc_sim_map = sim_maps[use_case]

        if idx % ncols == 0:
            show_ylabel = True
        else:
            show_ylabel = False

        if idx // ncols == nrows - 1:
            show_xlabel = True
        else:
            show_xlabel = False

        cbar = False
        cbar_ax = None
        # if idx // ncols == 1 and idx % ncols == ncols - 1:
        if idx == len(sim_maps) - 1:
            cbar = True
            cbar_ax = colorbar_ax

        plot_attention_sim(uc_sim_map, ax=ax, title=use_case_map[use_case], show_xlabel=show_xlabel,
                           show_ylabel=show_ylabel, cbar=cbar, cbar_ax=cbar_ax)

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.02, hspace=0.15)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')

    plt.show()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Impact of the EM fine-tuning on the BERT attention weights')

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
    parser.add_argument('-precomputed', '--precomputed', default=False,
                        type=lambda x: bool(distutils.util.strtobool(x)),
                        help='avoid re-computing the attention similarity maps by loading previously saved results \
                        (with the --save option)')
    parser.add_argument('-save', '--save', default=True,
                        type=lambda x: bool(distutils.util.strtobool(x)),
                        help='boolean flag for saving attention similarity maps')

    # Parameters for data sampling
    parser.add_argument('-sample_size', '--sample_size', type=int,
                        help='size of the sample')
    parser.add_argument('-sample_target_class', '--sample_target_class', default='both', choices=['both', 0, 1],
                        help='classes to sample: match, non-match or both')
    parser.add_argument('-sample_seeds', '--sample_seeds', nargs='+', default=[42, 42],
                        help='seeds for each class sample. <seed non match> <seed match>')

    args = parser.parse_args()

    use_cases = args.use_cases
    if use_cases == ['all']:
        use_cases = DM_USE_CASES

    conf = {
        'use_case': use_cases,
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

    # Compute similarity maps
    sim_maps = get_pt_ft_attention_sim(conf, sampler_conf, precomputed=args.precomputed, save=args.save)

    # Plot the average similarity map over the entire benchmark
    all_sim_map = sim_maps['all']
    del sim_maps['all']
    plot_attention_sim(all_sim_map)

    # Plot the similarity maps for each use case
    plot_save_path = os.path.join(RESULTS_DIR, 'PLOT_attention_pt_ft_similarity.pdf')
    plot_attention_sim_maps(sim_maps, save_path=plot_save_path)
