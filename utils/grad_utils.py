import os
import pickle
import matplotlib.pyplot as plt
from core.explanation.gradient.extractors import EntityGradientExtractor
import numpy as np
import pandas as pd


def load_saved_grads_data(use_case, conf, sampler_conf, fine_tune, grad_conf, res_dir):
    tok = conf['tok']
    size = sampler_conf['size']
    text_unit = grad_conf['text_unit']
    special_tokens = grad_conf['special_tokens']
    out_fname = f"{use_case}_{tok}_{size}_{fine_tune}_{text_unit}_{special_tokens}"
    data_path = os.path.join(res_dir, use_case, out_fname)
    uc_grad = pickle.load(open(f"{data_path}.pkl", "rb"))
    return uc_grad


def plot_multi_use_case_grads(conf, sampler_conf, fine_tune, grads_conf, use_cases, out_dir, grad_agg_metrics=['avg'],
                              plot_type='box', ignore_special: bool = True, out_plot_name: str = None,
                              use_case_map=None):
    assert isinstance(use_cases, list), "Wrong data type for parameter 'use_cases'."
    assert len(use_cases) > 0, "Empty use case list."
    grad_agg_available_metrics = ['sum', 'avg', 'median', 'max']
    assert all([m in grad_agg_available_metrics for m in grad_agg_metrics]), "Wrong metric names."
    plot_types = ['box', 'error']
    assert plot_type in plot_types
    if plot_type == 'box':
        assert len(grad_agg_metrics) == 1, "Only one metric supported in the 'box' plot type."

    grad_special_tokens = grads_conf['special_tokens']

    ncols = 4
    nrows = 3
    if len(use_cases) == 1:
        ncols = 1
        nrows = 1
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 8), sharey=True)
    if len(use_cases) > 1:
        axes = axes.flat
    # loop over the use cases
    for idx, use_case in enumerate(use_cases):
        # load grads data
        uc_grad = load_saved_grads_data(use_case, conf, sampler_conf, fine_tune, grads_conf, out_dir)

        # check grads data format
        EntityGradientExtractor.check_extracted_grad(uc_grad)
        first_data = None
        for item in uc_grad:
            if item is not None:
                first_data = item
                break
        if not first_data:
            continue
        text_unit_names = first_data['grad']['all']
        sep_idxs = list(np.where(np.array(text_unit_names) == '[SEP]')[0])
        skip_idxs = [0] + sep_idxs
        assert all([item['grad']['all'] == text_unit_names for item in uc_grad if item is not None])

        if len(use_cases) > 1:
            ax = axes[idx]
        else:
            ax = axes

        uc_plot_data = {}
        for item in uc_grad:
            if item is not None:
                for m in item['grad']['all_grad']:
                    if m in grad_agg_metrics:

                        x = item['grad']['all_grad'][m]
                        if ignore_special:
                            if text_unit_names[0] == '[CLS]':
                                x = [x[i] for i in range(len(x)) if i not in skip_idxs]

                        if m not in uc_plot_data:
                            uc_plot_data[m] = [x]
                        else:
                            uc_plot_data[m].append(x)

        if ignore_special:
            text_unit_names = [text_unit_names[i] for i in range(len(text_unit_names)) if i not in skip_idxs]

        columns = []
        num_columns = len(text_unit_names)
        if grad_special_tokens and not ignore_special:
            num_columns -= 3
        if not ignore_special:
            columns.append('[CLS]')
        half_columns = num_columns // 2
        for num in range(1, half_columns + 1):
            columns.append(f'l{num}')
        if not ignore_special:
            columns.append('[SEP]')
        for num in range(1, half_columns + 1):
            columns.append(f'r{num}')
        if not ignore_special:
            columns.append('[SEP]')

        for metric in uc_plot_data:

            uc_plot_metric_table = pd.DataFrame(uc_plot_data[metric], columns=columns)

            if plot_type == 'error':
                uc_plot_metric_table_stats = uc_plot_metric_table.describe()
                medians = uc_plot_metric_table_stats.loc['50%', :].values
                percs_25 = uc_plot_metric_table_stats.loc['25%', :].values
                percs_75 = uc_plot_metric_table_stats.loc['75%', :].values
                uc_plot_metric_data = {
                    'x': range(len(uc_plot_metric_table_stats.columns)),
                    'y': medians,
                    'yerr': [medians - percs_25, percs_75 - medians],
                }

                ax.errorbar(**uc_plot_metric_data, alpha=.75, fmt=':', capsize=3, capthick=1, label=metric)
                uc_plot_metric_data_area = {
                    'x': uc_plot_metric_data['x'],
                    'y1': percs_25,
                    'y2': percs_75
                }
                ax.fill_between(**uc_plot_metric_data_area, alpha=.25)
                ax.set_xticks(range(len(uc_plot_metric_table.columns)))
                ax.set_xticklabels(uc_plot_metric_table.columns)
                ax.legend()

            elif plot_type == 'box':
                uc_plot_metric_table.boxplot(ax=ax)
            else:
                raise NotImplementedError("Wrong plot type.")

        if use_case_map is not None:
            ax.set_title(use_case_map[use_case])
        else:
            ax.set_title(use_case)
        if idx % ncols == 0:
            ax.set_ylabel('Gradient')
        if idx // ncols == nrows - 1:
            ax.set_xlabel('Attributes')

    plt.subplots_adjust(wspace=0.1, hspace=0.3)
    if out_plot_name:
       plt.savefig(out_plot_name, bbox_inches='tight')
    plt.show()


def plot_grads(grad_data: dict, target_entity: str, title: str = None, out_plot_name: str = None,
               ignore_special: bool = False, max_y = None):
    assert isinstance(grad_data, dict), "Wrong data type for parameter 'grad_data'."
    params = ['all', 'all_grad', 'left', 'left_grad', 'right', 'right_grad']
    assert all([p in grad_data for p in params]), "Wrong data format for parameter 'grad_data'."
    entities = ['all', 'left', 'right']
    assert isinstance(target_entity, str), "Wrong data type for parameter 'target_entity'."
    assert target_entity in entities, f"Wrong target entity: {target_entity} not in {entities}."
    if title is not None:
        assert isinstance(title, str), "Wrong data type for parameter 'title'."

    plt.subplots(figsize=(20, 10))

    x = grad_data[target_entity]
    sep_idxs = list(np.where(np.array(x) == '[SEP]')[0])
    skip_idxs = [0] + sep_idxs
    if ignore_special:
        if x[0] == '[CLS]':
            x = [x[i] for i in range(len(x)) if i not in skip_idxs]

    # check for duplicated labels
    label_counts = pd.Series(x).value_counts()
    not_unique_labels = label_counts[label_counts > 1]
    if len(not_unique_labels) > 0:
        new_x = x.copy()
        for nul in list(not_unique_labels.index):
            nul_idxs = np.where(np.array(x) == nul)[0]
            for i, nul_idx in enumerate(nul_idxs, 1):
                new_x[nul_idx] = f'{nul}_{i}'  # concatenating the duplicated label with an incremental id
        x = new_x.copy()

    if isinstance(grad_data[f'{target_entity}_grad'], dict):
        for m in grad_data[f'{target_entity}_grad']:

            if m not in ['avg']:
                continue

            y = grad_data[f'{target_entity}_grad'][m]
            if ignore_special:
                if grad_data[target_entity][0] == '[CLS]':
                    y = [y[i] for i in range(len(y)) if i not in skip_idxs]

            yerr = None
            if f'{target_entity}_error_grad' in grad_data:
                yerr = grad_data[f'{target_entity}_error_grad']
            barlist = plt.bar(x, y, yerr=yerr, label=m)
            max_grads_idxs = np.array(y).argsort()[-4:][::-1]
            for max_grads_idx in max_grads_idxs:
                barlist[max_grads_idx].set_color('r')
    else:
        y = grad_data[f'{target_entity}_grad']
        if ignore_special:
            if grad_data[target_entity][0] == '[CLS]':
                y = [y[i] for i in range(len(y)) if i not in skip_idxs]

        yerr = None
        if f'{target_entity}_error_grad' in grad_data:
            yerr = grad_data[f'{target_entity}_error_grad']
        barlist = plt.bar(x, y, yerr=yerr)
        max_grads_idxs = np.array(y).argsort()[-4:][::-1]
        for max_grads_idx in max_grads_idxs:
            barlist[max_grads_idx].set_color('r')

    plt.xticks(rotation=90)
    if max_y:
        plt.ylim(0, max_y)
    if title is not None:
        plt.title(title)
    plt.legend()
    plt.subplots_adjust(bottom=0.15)
    fig = plt.gcf()
    plt.show()
    decision = input("Press Enter to continue or press S to save the plot and continue...")
    if decision == "S":
        fig.savefig(out_plot_name, bbox_inches='tight')


def plot_batch_grads(grads_data: list, target_entity: str, title_prefix: str = None, out_plot_name: str = None,
                     ignore_special: bool = False):
    EntityGradientExtractor.check_extracted_grad(grads_data)

    # get max gradient to have an upper bound for the y-axis when plotting
    max_grad = 0
    for g in grads_data:
        text_units = g['grad'][f'{target_entity}']
        sep_idxs = list(np.where(np.array(text_units) == '[SEP]')[0])
        skip_idxs = [0] + sep_idxs
        if isinstance(g['grad'][f'{target_entity}_grad'], dict):
            x = g['grad'][f'{target_entity}_grad']['avg']
        else:
            x = g['grad'][f'{target_entity}_grad']
        if ignore_special:
            if text_units[0] == '[CLS]':
                x = [x[i] for i in range(len(x)) if i not in skip_idxs]
        max_g = np.max(x)
        if max_g > max_grad:
            max_grad = max_g

    for idx, grad_data in enumerate(grads_data):

        if grad_data is None:
            print(f"No gradients for item {idx}.")
            continue

        grad = grad_data['grad']
        label = grad_data['label']
        prob = grad_data['prob']
        pred = grad_data['pred']
        title = f"gradients for item#{idx} - label: {label} - pred: {pred} - prob: {prob}"

        if title_prefix is not None:
            title = f'{title_prefix} {title}'

        plot_grads(grad, target_entity, title=title, out_plot_name=f'{out_plot_name}_{idx}.pdf',
                   ignore_special=ignore_special, max_y=max_grad)
