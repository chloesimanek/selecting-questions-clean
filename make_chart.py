import pandas as pd
import matplotlib.pyplot as plt

# Load data and strip column names to remove hidden spaces
df = pd.read_csv('results.csv')
df.columns = df.columns.str.strip()

# Create the line chart
plt.figure(figsize=(8, 5))

# Plot each model as a separate line
for model in df['model'].unique():
    subset = df[df['model'] == model]
    plt.plot(subset['n_query'], subset['test_accuracy'], marker='o', label=model)

# Configure plot
plt.xlabel('Number of Questions')
plt.ylabel('Test Accuracy')
plt.title('Test Accuracy vs Number of Questions per Model')
plt.legend(title='Model')
plt.grid(False)
plt.xticks([1, 3, 5, 10])  # Only show specific x-axis values
plt.tight_layout()

# Save figure
plt.savefig('model-comparison.png')

# import pandas as pd
# import matplotlib.pyplot as plt

# # Sample data
# data = {
#     "Model": [
#         "biirt-bad", # 1, 0.7008
#         "biirt-random", # 2
#         "biirt-active", # 2
#         "biirt-oracle", # 2
#         "biirt-random", # 5
#         "biirt-active", # 5
#         "biirt-oracle", # 5
#         "biirt-random", # 10
#         "biirt-active", # 10
#         "biirt-oracle" # 10
#     ],
#     "Questions": [1, 2, 2, 2, 5, 5, 5, 10, 10, 10],
#     "Test Accuracy": [0.7008, 0.72, 0.74, 0.75, 0.73, 0.74, 0.75, 0.73, 0.74, 0.75]
# }

# df = pd.DataFrame(data)

# # Create the line chart
# plt.figure(figsize=(8, 5))
# for model in df['Model'].unique():
#     subset = df[df['Model'] == model]
#     plt.plot(subset['Questions'], subset['Test Accuracy'], marker='o', label=model)

# plt.xlabel('Number of Questions')
# plt.ylabel('Test Accuracy')
# plt.title('Test Accuracy vs Number of Questions per Model')
# plt.legend(title='Model')
# plt.grid(False)
# plt.xticks([1, 3, 5, 10]) 
# plt.tight_layout()
# plt.savefig('data/model-comparision.png')
