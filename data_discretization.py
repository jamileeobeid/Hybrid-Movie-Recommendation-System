import pandas as pd

# Loading the cleaned users metadata
users_clean = pd.read_csv("users_clean.csv")

# Copy for discretization
users_disc = users_clean.copy()
users_disc["age"] = users_disc["age"].astype("Int64")

# Define bins and labels
bins = [0, 18, 30, 50, 100]
labels = ['Teen', 'Young Adult', 'Adult', 'Senior']

# Apply discretization
users_disc['age_group'] = pd.cut(
    users_disc['age'],
    bins=bins,
    labels=labels,
    include_lowest=True
)

print(users_disc[['user_id', 'age', 'age_group']].head(10))

age_group_counts = users_disc['age_group'].value_counts().sort_index()
print(age_group_counts)

users_disc.to_csv("users_discretized.csv", index=False)
print("Discretized users data saved as users_discretized.csv")