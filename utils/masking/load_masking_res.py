import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def save_sent_sim_pair_plot(data):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8.75, 3.5), sharey=True)
    axes = axes.flat
    df_match = data[data['data type'] == 'match']

    sns.boxplot(x="model type", hue="model", y="cosine sim", data=df_match, ax=axes[0], linewidth=0.7)
    axes[0].set_title('Match', fontsize=14)
    axes[0].get_legend().remove()
    axes[0].tick_params(axis='both', which='major', labelsize=14)
    axes[0].tick_params(axis='both', which='minor', labelsize=14)
    axes[0].set_xlabel('model type', fontsize=14)
    axes[0].set_ylabel('cosine sim', fontsize=14)

    df_non_match = data[data['data type'] == 'non-match']
    sns.boxplot(x="model type", hue="model", y="cosine sim", data=df_non_match, ax=axes[1], linewidth=0.7)
    axes[1].set_title('Non-match', fontsize=14)
    axes[1].get_legend().remove()
    axes[1].get_yaxis().set_visible(False)
    axes[1].tick_params(axis='both', which='major', labelsize=14, left=False)
    axes[1].tick_params(axis='both', which='minor', labelsize=14)
    axes[1].set_xlabel('model type', fontsize=14)

    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=2)

    plt.tight_layout()
    plt.savefig("sbert_sent_sim_all.pdf")


def save_masking_pair_plot(data):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 4), sharey=True)
    axes = axes.flat
    df_sent = data[data['encoding'] == 'sent-pair']
    sns.boxplot(x="masking", hue="model", y="F1", data=df_sent, ax=axes[0])
    axes[0].set_title('Sent-pair', fontsize=14)
    axes[0].get_legend().remove()
    axes[0].tick_params(axis='both', which='major', labelsize=14)
    axes[0].tick_params(axis='both', which='minor', labelsize=14)
    axes[0].set_xlabel('masking', fontsize=14)
    axes[0].set_ylabel('F1', fontsize=14)

    df_attr = data[data['encoding'] == 'attr-pair']
    sns.boxplot(x="masking", hue="model", y="F1", data=df_attr, ax=axes[1])
    axes[1].set_title('Attr-pair', fontsize=14)
    axes[1].get_legend().remove()
    axes[1].get_yaxis().set_visible(False)
    axes[1].tick_params(axis='both', which='major', labelsize=14, left=False)
    axes[1].tick_params(axis='both', which='minor', labelsize=14)
    axes[1].set_xlabel('masking', fontsize=14)

    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=2)

    plt.tight_layout()
    plt.savefig("sbert_masking_all.pdf")


def save_masking_plot(data, key):
    sel_data = data[data['encoding'] == key]
    ax = sns.boxplot(x="masking", hue="model", y="F1", data=sel_data)
    ax.tick_params(axis='both', which='major', labelsize=14, left=False)
    ax.tick_params(axis='both', which='minor', labelsize=14)
    ax.set_xlabel('masking', fontsize=14)
    ax.set_ylabel('F1', fontsize=14)
    fig = plt.gcf()
    fig.set_size_inches(6, 4)
    plt.tight_layout()
    plt.savefig(f"sbert_masking_{key}.pdf")


def plot_masking_res():
    res_path = 'C:\\Users\\matte\\Downloads\\MeltRes3.csv'
    df = pd.read_csv(res_path)

    # save_masking_pair_plot(df)
    # save_masking_plot(df, 'sent-pair')
    save_masking_plot(df, 'attr-pair')


def plot_sentence_sim_res():
    res_path = 'C:\\Users\\matte\\Downloads\\SentSimilarity.csv'
    df = pd.read_csv(res_path, decimal=',')

    save_sent_sim_pair_plot(df)


if __name__ == '__main__':
    plot_masking_res()
    # plot_sentence_sim_res()
