import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


movies = pd.read_csv('movies_clean.csv')
ratings = pd.read_csv('ratings_clean.csv')


genre_cols = ['Action', 'Adventure', 'Animation', "Children's", 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']


item_counts = ratings['item_id'].value_counts().sort_values(ascending=False).reset_index()
item_counts.columns = ['item_id', 'rating_count']

df_merged = pd.merge(ratings, movies, on='item_id')


genre_ratings = []
for genre in genre_cols:
    ratings_in_genre = df_merged[df_merged[genre] == 1]['rating']
    for r in ratings_in_genre:
        genre_ratings.append({'Genre': genre, 'Rating': r})


df_genre_vis = pd.DataFrame(genre_ratings)


plt.figure(figsize=(14, 6))
sns.boxplot(x='Genre', y='Rating', data=df_genre_vis, palette='Set2')
plt.title('Distribution of Ratings by Genre', fontsize=14)
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('genre_boxplots.png')
plt.show()
