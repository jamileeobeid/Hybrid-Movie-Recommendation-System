from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import re
from datetime import datetime

# Define base path (where ml-100k folder is located)
base_path = "ml-100k/"

# Ratings: user_id, item_id, rating, timestamp
ratings = pd.read_csv(
    base_path + "u.data",
    sep="\t",
    header=None,
    names=["user_id", "item_id", "rating", "timestamp"],
    engine="python"
)

# Movies: item_id | title | release_date | video_release_date | imdb_url | 19 genres
genre_cols = [
    'unknown','Action','Adventure','Animation',"Children's",'Comedy','Crime',
    'Documentary','Drama','Fantasy','Film-Noir','Horror','Musical','Mystery',
    'Romance','Sci-Fi','Thriller','War','Western'
]

item_cols = ['item_id','title','release_date','video_release_date','imdb_url'] + genre_cols

items = pd.read_csv(
    base_path + "u.item",
    sep="|",
    header=None,
    names=item_cols,
    encoding="latin-1",
    engine="python"
)

# Users: user_id | age | gender | occupation | zip_code
users = pd.read_csv(
    base_path + "u.user",
    sep="|",
    header=None,
    names=["user_id","age","gender","occupation","zip_code"],
    engine="python"
)

print("Files loaded successfully!")
print("Ratings:", ratings.shape)
print("Movies:", items.shape)
print("Users:", users.shape)

print("\nRatings sample:")
print(ratings.head())

print("\nMovies sample:")
print(items.head())

print("\nUsers sample:")
print(users.head())

plt.figure(figsize=(6,4))

plt.hist(ratings['rating'], bins=5, edgecolor='black')

plt.title("Distribution of Ratings (1–5)")
plt.xlabel("Rating")
plt.ylabel("Count")

plt.grid(True)                  
plt.gca().set_axisbelow(True)    

plt.show()

ratings['item_id'].value_counts().hist(bins=40, figsize=(6,4), edgecolor='black')
plt.title("Distribution of Number of Ratings per Movie")
plt.xlabel("Number of Ratings")
plt.ylabel("Number of Movies")
plt.grid(True)                  
plt.gca().set_axisbelow(True)  
plt.show()

ratings['user_id'].value_counts().hist(bins=40, figsize=(6,4), edgecolor='black')
plt.title("Distribution of Number of Ratings per User")
plt.xlabel("Number of Ratings")
plt.ylabel("Number of Users")
plt.grid(True)                  
plt.gca().set_axisbelow(True)  
plt.show()

# Convert UNIX timestamp to datetime
ratings["timestamp_dt"] = pd.to_datetime(ratings["timestamp"], unit="s", errors="coerce")

# If duplicates exist: keep the latest rating per (user_id, item_id)
ratings_sorted = ratings.sort_values(["user_id","item_id","timestamp_dt"])
ratings_clean = ratings_sorted.drop_duplicates(subset=["user_id","item_id"], keep="last")

print("Before:", ratings.shape, "After de-duplicating:", ratings_clean.shape)
ratings_clean.head()

items_clean = items.copy()

# Extract year from title (e.g., "Toy Story (1995)")
def extract_year(title):
    if isinstance(title, str):
        m = re.search(r"\((\d{4})\)", title)
        return int(m.group(1)) if m else np.nan
    return np.nan

items_clean["year"] = items_clean["title"].apply(extract_year)

# Parse release_date to datetime (many blanks will become NaT)
items_clean["release_date_dt"] = pd.to_datetime(
    items_clean["release_date"], format="%d-%b-%Y", errors="coerce"
)

print(items_clean["title"].head(25))

# Ensure genre columns are {0,1} integers
for g in genre_cols:
    items_clean[g] = items_clean[g].fillna(0).astype(int)

items_clean[["item_id","title","year","release_date","release_date_dt"] + genre_cols].head()

users_clean = users.copy()

# Normalize occupation labels (lowercase) and trim
users_clean["occupation"] = users_clean["occupation"].str.strip().str.lower()

# Standardize gender to uppercase single-letter
users_clean["gender"] = users_clean["gender"].str.strip().str.upper()

# Validate ages: keep within a reasonable range (5..100); set others to NaN (none expected)
invalid_age = ~users_clean["age"].between(5, 100)
users_clean.loc[invalid_age, "age"] = np.nan

users_clean.head()

valid_users = set(users_clean["user_id"].unique())
valid_items = set(items_clean["item_id"].unique())

fk_ok = ratings_clean["user_id"].isin(valid_users) & ratings_clean["item_id"].isin(valid_items)
fk_violations = ratings_clean[~fk_ok]
print("FK violations:", len(fk_violations))

# Drop FK-violating rows if any (ML-100K typically has none)
ratings_clean = ratings_clean[fk_ok]

ratings_clean[["user_id","item_id","rating","timestamp","timestamp_dt"]].to_csv("ratings_clean.csv", index=False)
items_clean[["item_id","title","year","release_date","release_date_dt","imdb_url"] + genre_cols].to_csv("movies_clean.csv", index=False)
users_clean.to_csv("users_clean.csv", index=False)

print("Saved: ratings_clean.csv, movies_clean.csv, users_clean.csv")
