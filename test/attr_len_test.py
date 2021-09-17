from utils.general import get_benchmark_avg_attr_len
import matplotlib.pyplot as plt


def plot_avg_attr_len(avg_attr_len):
    ncols = 4
    nrows = 3
    if len(avg_attr_len) == 1:
        ncols = 1
        nrows = 1
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 14), sharey=True)
    if len(avg_attr_len) > 1:
        axes = axes.flat
    for idx, use_case in enumerate(avg_attr_len):

        if len(avg_attr_len) > 1:
            ax = axes[idx]
        else:
            ax = axes

        use_case_stats = avg_attr_len[use_case]
        use_case_stats.plot(kind='bar', ax=ax, legend=False, rot=45, width=0.5)
        ax.set_title(use_case, fontsize=18)
        if idx % ncols == 0:
            ax.set_ylabel('Avg length', fontsize=20)
        ax.xaxis.set_tick_params(labelsize=15)
        ax.yaxis.set_tick_params(labelsize=20)
        # for p in ax.patches:
        #     ax.annotate(str(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.005))
        labels = use_case_stats.values
        for i, v in enumerate(labels):
            ax.text(i - .25,
                      v / labels[i],
                      labels[i],
                      fontsize=18)

    plt.subplots_adjust(wspace=0.05, hspace=0.85)
    # if out_plot_name:
    #     plt.savefig(out_plot_name, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':

    use_cases = ["Structured_Fodors-Zagats", "Structured_DBLP-GoogleScholar", "Structured_DBLP-ACM",
                 "Structured_Amazon-Google", "Structured_Walmart-Amazon", "Structured_Beer",
                 "Structured_iTunes-Amazon", "Textual_Abt-Buy", "Dirty_iTunes-Amazon", "Dirty_DBLP-ACM",
                 "Dirty_DBLP-GoogleScholar", "Dirty_Walmart-Amazon"]

    conf = {
        'data_type': 'train',  # 'train', 'test', 'valid'
        'model_name': 'bert-base-uncased',
        'tok': 'sent_pair',  # 'sent_pair', 'attr', 'attr_pair'
        'label_col': 'label',
        'left_prefix': 'left_',
        'right_prefix': 'right_',
        'max_len': 128,
        'permute': False,
        'verbose': False,
    }

    sampler_conf = {
        'size': 50,
        'target_class': 'both',  # 'both', 0, 1
        'seeds': [42, 42],  # [42 -> class 0, 42 -> class 1]
    }

    avg_attr_len = get_benchmark_avg_attr_len(use_cases, conf, sampler_conf, pair_mode=True, text_unit='word')
    plot_avg_attr_len(avg_attr_len)
