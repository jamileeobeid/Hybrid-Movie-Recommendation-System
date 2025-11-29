import pandas as pd

ratings_clean = pd.read_csv("ratings_clean.csv")

# Pivot user-item rating matrix
user_movie_matrix = ratings_clean.pivot(
    index='user_id',
    columns='item_id',
    values='rating'
).fillna(0)

print("User–Movie Matrix shape:", user_movie_matrix.shape)
user_movie_matrix.head()

from sklearn.decomposition import TruncatedSVD

# Initialize SVD with 50 latent features
svd = TruncatedSVD(n_components=50, random_state=42)

# Fit and transform the matrix
user_features = svd.fit_transform(user_movie_matrix)

print("Original dimensions:", user_movie_matrix.shape)
print("Reduced dimensions:", user_features.shape)

explained_variance = svd.explained_variance_ratio_.sum()
print(f"Total variance retained: {explained_variance:.2%}")

# Converting reduced features to a dataframe
user_features_df = pd.DataFrame(
    user_features,
    index=user_movie_matrix.index,
    columns=[f"latent_feature_{i+1}" for i in range(50)]
)

user_features_df.to_csv("user_features_svd.csv")
print("Reduced feature matrix saved as user_features_svd.csv")
