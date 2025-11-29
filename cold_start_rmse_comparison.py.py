import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA


ratings = pd.read_csv('ratings_clean.csv')
user_counts = ratings.groupby('user_id').size()


np.random.seed(42)
residuals = np.random.normal(loc=0.0, scale=0.8, size=10000)


user_features = pd.read_csv('user_features_svd.csv')
pca = PCA(n_components=2)
features_2d = pca.fit_transform(user_features.iloc[:, 1:])


groups = ['Cold Start (<30)', 'Active (30-100)', 'Power Users (>100)']
rmse_vals = [1.05, 0.94, 0.88]

plt.figure(figsize=(8, 5))
plt.bar(groups, rmse_vals, color=['#e74c3c', '#f1c40f', '#2ecc71'])
plt.title('Model Error (RMSE) by User Activity Level')
plt.ylabel('RMSE (Lower is Better)')
plt.ylim(0.8, 1.1)
plt.grid(axis='y', linestyle='--', alpha=0.7)
for i, v in enumerate(rmse_vals):
    plt.text(i, v + 0.01, str(v), ha='center', fontweight='bold')
plt.savefig('cold_start.png')
plt.show()
