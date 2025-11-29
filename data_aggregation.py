import pandas as pd

ratings_clean = pd.read_csv("ratings_clean.csv")
users_clean   = pd.read_csv("users_clean.csv")
items_clean   = pd.read_csv("movies_clean.csv")   # load cleaned movie data

# Define genre columns (same as in cleaning file)
genre_cols = [
    'unknown','Action','Adventure','Animation',"Children's",'Comedy','Crime',
    'Documentary','Drama','Fantasy','Film-Noir','Horror','Musical','Mystery',
    'Romance','Sci-Fi','Thriller','War','Western'
]

# Merging ratings with user data
ratings_users = ratings_clean.merge(users_clean[['user_id', 'occupation']], on='user_id', how='left')

# Computing average rating per occupation
avg_rating_by_occ = ratings_users.groupby('occupation')['rating'].mean().sort_values(ascending=False)

# Displaying result
print("Average Rating by Occupation:\n")
print(avg_rating_by_occ.round(2))

# Merging ratings with movie data
ratings_movies = ratings_clean.merge(items_clean[['item_id'] + genre_cols], on='item_id', how='left')

# Multiplying ratings by each genre indicator to weight averages
for g in genre_cols:
    ratings_movies[g] = ratings_movies[g] * ratings_movies['rating']

# Computing average rating per genre
genre_avg = pd.DataFrame({
    'genre': genre_cols,
    'avg_rating': [
        ratings_movies.loc[ratings_movies[g] > 0, 'rating'].mean()
        if (ratings_movies[g] > 0).any() else float('nan')
        for g in genre_cols
    ]
})

# Sorting and displaying
genre_avg = genre_avg.sort_values(by='avg_rating', ascending=False)
print("Average Rating by Genre:\n")
print(genre_avg.round(2))

import matplotlib.pyplot as plt

plt.figure(figsize=(8,4))
avg_rating_by_occ.sort_values().plot(kind='barh', color='indianred')
plt.title("Average Movie Rating by Occupation")
plt.xlabel("Average Rating")
plt.ylabel("Occupation")
plt.show()

plt.figure(figsize=(8,4))
plt.barh(genre_avg['genre'], genre_avg['avg_rating'], color='teal')
plt.title("Average Movie Rating by Genre")
plt.xlabel("Average Rating")
plt.ylabel("Genre")
plt.show()
