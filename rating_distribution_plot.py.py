import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('ratings_clean.csv')


sns.set_style("whitegrid")
plt.figure(figsize=(10, 6))


ax = sns.countplot(x='rating', data=df, palette="viridis")


plt.title('Distribution of User Ratings in MovieLens Dataset', fontsize=16, fontweight='bold')
plt.xlabel('Rating (1-5 Stars)', fontsize=12)
plt.ylabel('Number of Ratings', fontsize=12)


for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', fontsize=11, color='black', xytext=(0, 5),
                textcoords='offset points')


text_str = "Insight: \nMost ratings are positive (>=4).\nThis justifies using Precision/Recall\nto distinguish 'Good' from 'Great'."
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
plt.text(0.5, 25000, text_str, fontsize=12, bbox=props)

plt.tight_layout()
plt.savefig('rating_distribution_metrics.png')
plt.show()