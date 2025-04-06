import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("results/results.csv", index_col=0)
fig, ax = plt.subplots(figsize=(10, 2))  # Adjust size as needed
ax.axis('off')  # No axis

table = ax.table(cellText=df.values,
                 colLabels=df.columns,
                 loc='center',
                 cellLoc='center')

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.2)

plt.savefig("results/results_table.png", bbox_inches='tight', dpi=300)