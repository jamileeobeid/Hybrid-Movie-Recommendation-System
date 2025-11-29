import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


movies = pd.read_csv('movies_clean.csv')
ratings = pd.read_csv('ratings_clean.csv')


genre_cols = ['Action', 'Adventure', 'Animation', "Children's", 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']


item_counts = ratings['item_id'].value_counts().sort_values(ascending=False).reset_index()
item_counts.columns = ['item_id', 'rating_count']


plt.figure(figsize=(10, 6))
plt.loglog(range(1, len(item_counts) + 1), item_counts['rating_count'], color='#2c3e50', linewidth=2)
plt.fill_between(range(1, len(item_counts) + 1), item_counts['rating_count'], color='#3498db', alpha=0.3)
plt.title('The Long Tail - Item Popularity Distribution (Log-Log)', fontsize=14)
plt.xlabel('Movie Rank (by Popularity)', fontsize=12)
plt.ylabel('Number of Ratings (Log Scale)', fontsize=12)
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.text(500, 100, 'The "Tail":\nMost movies have\nvery few ratings', fontsize=12, color='red')
plt.tight_layout()
plt.savefig('long_tail.png')
plt.show()
