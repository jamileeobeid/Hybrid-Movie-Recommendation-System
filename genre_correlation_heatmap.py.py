import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


movies = pd.read_csv('movies_clean.csv')
ratings = pd.read_csv('ratings_clean.csv')


genre_cols = ['Action', 'Adventure', 'Animation', "Children's", 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']


item_counts = ratings['item_id'].value_counts().sort_values(ascending=False).reset_index()
item_counts.columns = ['item_id', 'rating_count']


genre_matrix = movies[genre_cols]
corr_matrix = genre_matrix.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, cmap='coolwarm', vmin=-0.5, vmax=0.5, linewidths=.5, annot=False)
plt.title('Correlation Heatmap of Movie Genres', fontsize=14)
plt.tight_layout()
plt.savefig('genre_corr.png')
plt.show()
