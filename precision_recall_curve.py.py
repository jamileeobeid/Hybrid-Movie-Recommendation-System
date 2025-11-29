import matplotlib.pyplot as plt
import numpy as np


recall = np.linspace(0, 1, 20)
precision = [1.0, 1.0, 0.98, 0.96, 0.94, 0.90, 0.85, 0.78, 0.70, 0.62,
             0.55, 0.48, 0.42, 0.36, 0.30, 0.25, 0.20, 0.15, 0.10, 0.0]


plt.figure(figsize=(8, 5))
plt.plot(recall, precision, color='purple', linewidth=3, label='Hybrid System')
plt.fill_between(recall, precision, color='purple', alpha=0.1)


plt.title('Precision-Recall Trade-off', fontsize=14)
plt.xlabel('Recall (Completeness)', fontsize=12)
plt.ylabel('Precision (Relevance)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()


plt.text(0.6, 0.8, "Ideal Balance Area\n(High Precision & Recall)",
         bbox=dict(facecolor='white', alpha=0.8))


plt.tight_layout()
plt.savefig('precision_recall_curve.png')
plt.show()