import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


movies = pd.read_csv('movies_clean.csv')
ratings = pd.read_csv('ratings_clean.csv')


genre_cols = ['Action', 'Adventure', 'Animation', "Children's", 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']


item_counts = ratings['item_id'].value_counts().sort_values(ascending=False).reset_index()
item_counts.columns = ['item_id', 'rating_count']


user_item_matrix = ratings.pivot(index='user_id', columns='item_id', values='rating')


subset_matrix = user_item_matrix.iloc[:100, :100].notnull().astype(int)
plt.figure(figsize=(10, 10))
plt.imshow(subset_matrix, cmap='Greys', interpolation='nearest', aspect='auto')
plt.title('User-Item Interaction Matrix (Spy Plot) - 100x100 Slice', fontsize=14)
plt.xlabel('Item ID (First 100)')
plt.ylabel('User ID (First 100)')
plt.grid(False)
plt.tight_layout()
plt.savefig('sparsity.png')
plt.show()
