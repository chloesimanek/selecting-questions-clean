import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv('results-0.75.csv')
df.columns = df.columns.str.strip()

plt.figure(figsize=(8, 5))

# Plot each model as a separate line
for model in df['model'].unique():
    subset = df[df['model'] == model]
    plt.plot(subset['n_query'], subset['test_accuracy'], marker='o', label=model)

theoretical_best = 0.757588
plt.axhline(y=theoretical_best, color='black', linestyle='--')

plt.xlabel('Number of Questions')
plt.ylabel('Test Accuracy')
plt.title('')
plt.legend(title='Model')
plt.grid(False)
plt.xticks([1, 3, 5, 10])  # Only show specific x-axis values
plt.tight_layout()


plt.savefig('model-comparison.png')

