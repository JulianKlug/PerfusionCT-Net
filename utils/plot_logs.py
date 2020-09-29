import os, argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_logs(log_file, split_names=None, save_dir=None, single_metric=None):
    if split_names is None:
        split_names = ['train', 'validation', 'test']
    if save_dir is None:
        save_dir = os.path.dirname(log_file)

    split_dfs = {}
    for split in split_names:
        split_dfs[split] = pd.read_excel(log_file, sheet_name=split)
    flatten = lambda l: [item for sublist in l for item in sublist]
    metrics = np.unique(flatten([df.keys()[1:].values.tolist() for _, df in split_dfs.items()]))

    n_cols = 3
    n_rows = int(len(metrics) / n_cols) + 1
    if single_metric is not None:
        n_cols = 1
        n_rows = 1
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 10))
    for index, metric in enumerate(metrics):
        if metric == 'epoch': continue
        if single_metric is not None:
            if metric != single_metric: continue
            else:
                axes = np.array([[axes]])
                index = 0
        col = index % n_cols
        row = int(index / n_cols)

        axes[row, col].set_ylabel('Epochs')
        axes[row, col].set_title(metric)
        axes[row, col].set_xlabel(metric)
        for split_name, df in split_dfs.items():
            if metric not in df.columns: continue
            df.plot(ax=axes[row, col], x='epoch', y=metric, label=split_name)
    experiment_name = os.path.basename(os.path.dirname(log_file))
    plt.suptitle(f'Scores for logs of {experiment_name}')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # remove empty last plots
    n_unused_plots = n_rows * n_cols - (metrics.size - 1)
    if n_unused_plots > 0:
        for ax in axes.flatten()[-1*(n_unused_plots):]:
            ax.set_visible(False)

    figure_name = experiment_name + '_figures'
    if single_metric is not None: figure_name = experiment_name + '_' + single_metric
    plt.savefig(os.path.join(save_dir, figure_name + '.png'), format="png", dpi=300)
    plt.close(fig)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Plot logs')
    parser.add_argument('log_file')

    parser.add_argument('-m', '--metric', help='Plot only this metric')
    parser.add_argument('-s', '--save-dir',  help='directory to save plots to')
    parser.add_argument('-n', '--split-names',   help='names of data splits in log ie. ["train", "validation", "test"]')
    args = parser.parse_args()

    plot_logs(args.log_file, args.split_names, args.save_dir, single_metric=args.metric)
