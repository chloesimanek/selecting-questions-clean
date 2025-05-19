import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_metric(df, metric_col, ylabel, y_range, output_file, theoretical_best=None):
    plt.figure(figsize=(8, 7))
    custom_colors = ['#1f77b4', '#ff7f0e', '#d62728', '#2ca02c', 'purple']

    plt.yticks(np.arange(y_range[0], y_range[1] + 0.001, y_range[2]))

    for i, model in enumerate(df['model'].unique()):
        subset = df[df['model'] == model]
        plt.plot(
            subset['n_query'],
            subset[metric_col],
            marker='.',
            label=model,
            linewidth=1,
            color=custom_colors[i % len(custom_colors)]
        )

    if theoretical_best is not None:
        plt.axhline(y=theoretical_best, color='black', linestyle='--')

    plt.xlabel('Number of Questions')
    plt.ylabel(ylabel)
    plt.title('')
    plt.legend(title='Model')
    plt.grid(False)
    plt.xticks([1, 5, 10])
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

def plot_test_accuracy(df):
    plot_metric(
        df,
        metric_col='test_accuracy',
        ylabel='Test Accuracy',
        y_range=(0.700, 0.800, 0.005),
        output_file='model-comparison-accuracy.png',
        theoretical_best=0.757588
    )

def plot_test_auc(df):
    plot_metric(
        df,
        metric_col='test_auc',
        ylabel='Test AUC',
        y_range=(0.600, 0.800, 0.01),
        output_file='model-comparison-auc.png',
    )

# Usage:
df = pd.read_csv('results-0.75.csv')
df.columns = df.columns.str.strip()

plot_test_accuracy(df)
plot_test_auc(df)
