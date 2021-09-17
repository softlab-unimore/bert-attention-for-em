import numpy as np
from matplotlib import pyplot as plt
import math
import seaborn as sns
from utils.result_collector import TestResultCollector


def plot_layers_heads_attention(attns, mask=None, out_file_name: str = None):
    x = attns.shape[0]
    y = attns.shape[1]

    if mask is not None:

        assert attns.shape[:2] == mask.shape

        nplots = mask.sum()
        plot_grid_size = math.floor(math.sqrt(nplots))
        if plot_grid_size * plot_grid_size == nplots:
            nrows, ncols = plot_grid_size, plot_grid_size
        else:
            nrows, ncols = plot_grid_size + 1, plot_grid_size + 1
    else:
        nrows = attns.shape[0]
        ncols = attns.shape[1]

    figsize = (10, 10)
    if nrows * ncols > 25:
        figsize = (20, 20)
    fig, axes = plt.subplots(nrows, ncols, sharey=True, figsize=figsize)

    count = 0
    for i in range(x):
        for j in range(y):

            plt_x = count // nrows
            plt_y = count % ncols

            if mask is not None:
                if mask[i][j] > 0:
                    ax = axes[plt_x][plt_y]
                    ax.set_title(f"L: {i}, H: {j}")
                    sns.heatmap(attns[i][j], ax=ax, cbar=False)
                    count += 1
            else:
                ax = axes[plt_x][plt_y]
                ax.set_title(f"L: {i}, H: {j}")
                sns.heatmap(attns[i][j], ax=ax, cbar=False)
                count += 1
    plt.subplots_adjust(hspace=0.5)

    if out_file_name:
        plt.savefig(out_file_name, bbox_inches='tight')

    plt.show()


def plot_results(results, tester, target_cats=None, plot_params=None, vmin=0, vmax=1, plot_type='simple',
                 save_path=None):
    assert isinstance(results, dict)
    if target_cats is not None:
        assert isinstance(target_cats, list)
        assert len(target_cats) > 0

    if plot_params is None:
        plot_params = ['match_attr_attn_loc', 'match_attr_attn_over_mean',
                       'avg_attr_attn', 'attr_attn_3_last', 'attr_attn_last_1',
                       'attr_attn_last_2', 'attr_attn_last_3',
                       'avg_attr_attn_3_last', 'avg_attr_attn_last_1',
                       'avg_attr_attn_last_2', 'avg_attr_attn_last_3']

    for use_case in results:
        print(use_case)
        uc_res = results[use_case]

        for cat, cat_res in uc_res.items():

            if cat_res is None:
                continue

            if target_cats:
                if cat not in target_cats:
                    continue

            print(cat)
            assert isinstance(cat_res, TestResultCollector)

            tester.plot(cat_res, plot_params=plot_params, labels=True, vmin=vmin, vmax=vmax, title_prefix=use_case,
                        plot_type=plot_type, out_file_name_prefix=save_path)


def plot_comparison(res1, res2, cmp_res, tester, cmp_vals, target_cats=None, plot_params=None):
    assert isinstance(res1, dict)
    assert isinstance(res2, dict)
    assert isinstance(cmp_res, dict)
    assert isinstance(cmp_vals, list)
    assert len(cmp_vals) == 2
    if target_cats is not None:
        assert isinstance(target_cats, list)
        assert len(target_cats) > 0

    if plot_params is None:
        plot_params = ['match_attr_attn_loc', 'match_attr_attn_over_mean',
                       'avg_attr_attn', 'attr_attn_3_last', 'attr_attn_last_1',
                       'attr_attn_last_2', 'attr_attn_last_3',
                       'avg_attr_attn_3_last', 'avg_attr_attn_last_1',
                       'avg_attr_attn_last_2', 'avg_attr_attn_last_3']

    for cat in set(res1).intersection(set(res2)):
        cat_res1 = res1[cat]
        cat_res2 = res2[cat]
        cmp_cat_res = cmp_res[cat]

        if cat_res1 is None or cat_res2 is None or cmp_cat_res is None:
            continue

        if target_cats:
            if cat not in target_cats:
                continue

        print(cat)
        assert isinstance(cat_res1, TestResultCollector)
        assert isinstance(cat_res2, TestResultCollector)
        assert isinstance(cmp_cat_res, TestResultCollector)

        tester.plot_comparison(cat_res1, cat_res2, cmp_cat_res, plot_params=plot_params, labels=True,
                               title_prefix=f"{cmp_vals[0]}_vs_{cmp_vals[1]}")


def plot_benchmark_results(results, tester, use_cases, target_cats=None, plot_params=None, title_prefix=None, vmin=0,
                           vmax=1, save_path: str = None):
    assert isinstance(results, dict)
    assert isinstance(use_cases, list)
    assert len(use_cases) > 0
    for use_case in use_cases:
        assert use_case in results
    if target_cats is not None:
        assert isinstance(target_cats, list)
        assert len(target_cats) > 0
    if title_prefix is not None:
        assert isinstance(title_prefix, str)

    first_use_case = use_cases[0]
    first_results = results[first_use_case]
    cats = list(first_results)

    if plot_params is None:
        plot_params = ['match_attr_attn_loc', 'match_attr_attn_over_mean',
                       'avg_attr_attn', 'attr_attn_3_last', 'attr_attn_last_1',
                       'attr_attn_last_2', 'attr_attn_last_3',
                       'avg_attr_attn_3_last', 'avg_attr_attn_last_1',
                       'avg_attr_attn_last_2', 'avg_attr_attn_last_3']

    for cat in cats:

        if target_cats:
            if cat not in target_cats:
                continue

        print(cat)

        for plot_param in plot_params:
            nrows = 3
            ncols = 4
            fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(24, 18))

            title = f'{plot_param}'
            if title_prefix is not None:
                title = f'{title_prefix} {title}'
            fig.suptitle(title)

            for idx, use_case in enumerate(use_cases):
                ax = axes[idx // ncols][idx % ncols]
                ax.set_title(use_case)
                cat_res = results[use_case][cat]

                if cat_res is None:
                    continue

                assert isinstance(cat_res, TestResultCollector)

                labels = False
                if idx % ncols == 0:
                    labels = True

                tester.plot(cat_res, plot_params=[plot_param], ax=ax, labels=labels,
                            vmin=vmin, vmax=vmax)
            plt.subplots_adjust(wspace=0.005, hspace=0.2)
            plt.show()
            # plt.savefig("avg_attn.pdf", bbox_inches='tight')


def plot_compared_results(res1, res2, diff_res, res1_name, res2_name, out_file_name=None):
    max_score = np.max([np.max(res1), np.max(res2)])
    fig, axes = plt.subplots(1, 3, figsize=(10, 5))
    cbar_ax = fig.add_axes([.91, 0.11, .03, .77])
    cbar_ax.tick_params(labelsize=16)
    axes[0].set_title(res1_name, fontdict={'fontsize': 16})
    if res1.ndim == 1:
        res1 = res1.reshape(-1, 1)
    h = sns.heatmap(res1, annot=True, ax=axes[0], vmax=max_score, vmin=0, yticklabels=range(1, len(res1) + 1),
                    xticklabels=False, cbar=True, cbar_ax=cbar_ax, annot_kws={"size": 14})
    h.set_yticklabels(h.get_yticklabels(), fontsize=16, rotation=0)
    axes[0].set_ylabel('layers', fontsize=14)
    axes[1].set_title(res2_name, fontdict={'fontsize': 16})
    if res2.ndim == 1:
        res2 = res2.reshape(-1, 1)
    sns.heatmap(res2, annot=True, ax=axes[1], vmax=max_score, vmin=0, xticklabels=False, yticklabels=False, cbar=False,
                cbar_ax=None, annot_kws={"size": 14})
    axes[2].set_title("Diff", fontdict={'fontsize': 16})
    if diff_res.ndim == 1:
        diff_res = diff_res.reshape(-1, 1)
    sns.heatmap(diff_res, annot=True, ax=axes[2], vmax=max_score, vmin=0, xticklabels=False, yticklabels=False,
                cbar=False, cbar_ax=None, annot_kws={"size": 14})
    plt.subplots_adjust(wspace=0.2)

    if out_file_name is not None:
        plt.savefig(out_file_name, bbox_inches='tight')

    plt.show()


def plot_agg_results(results, target_cats=None, title_prefix=None, xlabel=None, ylabel=None,
                     xticks=None, yticks=None, agg=False, vmin=0, vmax=0.5, plot_type='simple', save_path=None,
                     res1=None, res2=None, res1_name=None, res2_name=None):
    assert isinstance(results, dict)
    if target_cats is not None:
        assert isinstance(target_cats, list)
        assert len(target_cats) > 0
    if title_prefix is not None:
        assert isinstance(title_prefix, str)
    if res1 is not None:
        assert res2 is not None
        assert res1_name is not None
    if res2 is not None:
        assert res1 is not None
        assert res2_name is not None

    out_path = save_path.replace('.pdf', '')

    for cat in results:

        if target_cats is not None:
            if cat not in target_cats:
                continue

        cat_res = results[cat]
        print(cat)

        for metric in cat_res:

            print(metric)

            assert isinstance(cat_res[metric], dict)
            assert len(cat_res[metric]) == 1
            res_id = list(cat_res[metric].keys())[0]

            if res1 is None:
                figsize = (20, 10)
                if cat_res[metric][res_id].shape[1] < 3:
                    figsize = (6, 10)
                fig, ax = plt.subplots(figsize=figsize)

                title = f'{res_id} {metric}'
                if title_prefix is not None:
                    title = f'{title_prefix} {title}'
                fig.suptitle(title)

                sns.heatmap(cat_res[metric][res_id], annot=True, fmt='.2f', vmin=vmin, vmax=vmax,
                            xticklabels=xticks, ax=ax)
                plt.show()

            else:
                if not agg:
                    res1_cat = res1[cat][metric][res_id]
                    res2_cat = res2[cat][metric][res_id]
                    diff_res = cat_res[metric][res_id]
                    plot_compared_results(res1_cat, res2_cat, diff_res, res1_name, res2_name,
                                          out_file_name=f'{out_path}_{metric.split("_")[-1]}.pdf')

            if agg:
                if res1 is None:
                    fig, ax = plt.subplots(figsize=(5, 4))
                    fig.suptitle(f'{res_id} {metric} agg')
                    sns.heatmap(cat_res[metric][res_id].mean(1).reshape((-1, 1)), annot=True,
                                fmt='.2f', vmin=vmin, vmax=vmax, ax=ax)
                    plt.show()

                else:
                    res1_cat_agg = res1[cat][metric][res_id].mean(1).reshape((-1, 1))
                    res2_cat_agg = res2[cat][metric][res_id].mean(1).reshape((-1, 1))
                    diff_res_agg = cat_res[metric][res_id].mean(1).reshape((-1, 1))
                    plot_compared_results(res1_cat_agg, res2_cat_agg, diff_res_agg, res1_name, res2_name,
                                          out_file_name=f'{out_path}_{metric.split("_")[-1]}.pdf')


def plot_left_to_right_heatmap(data: np.ndarray, vmin: (int, float), vmax: (int, float), title: str = None,
                               is_annot: bool = True, out_file_name: str = None):
    fig, new_ax = plt.subplots(2, 2)
    cbar_ax = fig.add_axes([.89, .11, .03, .81])
    # fontsize = 14
    fontsize = 13
    cbar_ax.tick_params(labelsize=fontsize)
    n = len(data)
    h = n // 2
    score1 = data[:h, :h]
    score2 = data[:h, h:]
    score3 = data[h:, :h]
    score4 = data[h:, h:]
    attr_labels = [f'{i}' for i in range(1, h + 1)]

    annot = True
    if not is_annot:
        annot = False

    h1 = sns.heatmap(score1, annot=annot, fmt='.1f', ax=new_ax[0, 0], vmin=vmin, vmax=vmax, xticklabels=False,
                     yticklabels=attr_labels, cbar=True, cbar_ax=cbar_ax, annot_kws={"size": fontsize})
    h1.set_yticklabels(h1.get_yticklabels(), fontsize=fontsize, rotation=0)
    new_ax[0, 0].set_ylabel('left entity attrs', fontdict={'fontsize': 14})
    h2 = sns.heatmap(score2, annot=annot, fmt='.1f', ax=new_ax[0, 1], vmin=vmin, vmax=vmax, xticklabels=False,
                     yticklabels=False, cbar=False, cbar_ax=None, annot_kws={"size": fontsize})
    h3 = sns.heatmap(score3, annot=annot, fmt='.1f', ax=new_ax[1, 0], vmin=vmin, vmax=vmax, xticklabels=attr_labels,
                     yticklabels=attr_labels, cbar=False, cbar_ax=None, annot_kws={"size": fontsize})
    new_ax[1, 0].set_ylabel('right entity attrs', fontdict={'fontsize': 14})
    new_ax[1, 0].set_xlabel('left entity attrs', fontdict={'fontsize': 14})
    h3.set_xticklabels(h3.get_xticklabels(), rotation=0, fontsize=fontsize)
    h3.set_yticklabels(h3.get_yticklabels(), rotation=0, fontsize=fontsize)
    h4 = sns.heatmap(score4, annot=annot, fmt='.1f', ax=new_ax[1, 1], vmin=vmin, vmax=vmax, yticklabels=False,
                     xticklabels=attr_labels, cbar=False, cbar_ax=None, annot_kws={"size": fontsize})
    new_ax[1, 1].set_xlabel('right entity attrs', fontdict={'fontsize': 14})
    h4.set_xticklabels(h4.get_xticklabels(), rotation=0, fontsize=fontsize)
    # plt.subplots_adjust(top=0.99, wspace=0.01, hspace=0.01, right=0.88)
    plt.subplots_adjust(top=0.92, wspace=0.01, hspace=0.01, right=0.88)

    if title is not None:
        fig.suptitle(title, fontsize=16)

    if out_file_name is not None:
        plt.savefig(out_file_name, bbox_inches='tight')

    plt.show()


def plot_images_grid(imgs, nrows=3, ncols=4, save_path=None):
    fig, axes = plt.subplots(nrows, ncols, figsize=(24, 12))
    for i in range(len(imgs)):
        ax = axes[i // ncols][i % ncols]
        ax.imshow(imgs[i])
        ax.set_axis_off()
        ax.autoscale(False)
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                        hspace=0, wspace=0)

    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')

    plt.show()