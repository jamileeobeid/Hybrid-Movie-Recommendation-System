import pandas as pd
import matplotlib.pyplot as plt

# Loading the cleaned ratings dataset
ratings_clean = pd.read_csv("ratings_clean.csv")

# Defining sampling fraction
sample_fraction = 0.10  # Take 10% sample

# Performing simple random sampling
ratings_sample = ratings_clean.sample(frac=sample_fraction, random_state=42)

# Showing results
print("Original dataset:", ratings_clean.shape)
print("Sampled dataset:", ratings_sample.shape)

# Comparing rating distributions
plt.figure(figsize=(6,4))
ratings_clean['rating'].value_counts(normalize=True).sort_index().plot(kind='bar', color='steelblue', alpha=0.6, label='Full Data')
ratings_sample['rating'].value_counts(normalize=True).sort_index().plot(kind='bar', color='indianred', alpha=0.6, label='Sample')
plt.title("Comparison of Rating Distribution (Full vs Sample)")
plt.xlabel("Rating")
plt.ylabel("Proportion")
plt.legend()
plt.show()

from sklearn.model_selection import train_test_split

# Stratified sample based on 'rating'
_, ratings_stratified = train_test_split(
    ratings_clean,
    test_size=0.10,
    stratify=ratings_clean['rating'],
    random_state=42
)

print("Stratified sample shape:", ratings_stratified.shape)

ratings_sample.to_csv("ratings_sample.csv", index=False)
print("10% sampled data saved as ratings_sample.csv")
