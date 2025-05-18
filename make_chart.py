import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


df = pd.read_csv('results-0.75.csv')
df.columns = df.columns.str.strip()

plt.figure(figsize=(8, 7))
custom_colors = ['#1f77b4', '#ff7f0e', '#d62728','#2ca02c']
plt.yticks(np.arange(0.700, 0.800 + 0.001, 0.005))

# Plot each model as a separate line
for i, model in enumerate(df['model'].unique()):
    subset = df[df['model'] == model]
    plt.plot(
        subset['n_query'],
        subset['test_accuracy'],
        marker='.',
        label=model,
        linewidth=1,
        color=custom_colors[i % len(custom_colors)])

theoretical_best = 0.757588
plt.axhline(y=theoretical_best, color='black', linestyle='--')

plt.xlabel('Number of Questions')
plt.ylabel('Test Accuracy')
plt.title('')
plt.legend(title='Model')
plt.grid(False)
plt.xticks([1, 5, 10])  # Only show specific x-axis values
plt.tight_layout()


plt.savefig('model-comparison.png')

