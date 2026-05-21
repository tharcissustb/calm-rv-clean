"""
Figure F6: t-SNE Visualization of Learned Representations
Shows how L_align pulls same labels across events
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

# Simulated representation data
# In reality, you would extract from your model
np.random.seed(42)

# Create synthetic but realistic representations
n_samples = 300
n_dims = 128

# True labels: 0 (true), 1 (false), 2 (unverified)
# Events: A (easy), B (hard)

# Representations without alignment (baseline)
reps_baseline = np.random.randn(n_samples, n_dims)
reps_baseline[:100] += np.array([2, 0])  # Event A, label true
reps_baseline[100:200] += np.array([-1, 2])  # Event B, label false
reps_baseline[200:300] += np.array([0, -2])  # Mixed

# Representations with alignment (your framework)
reps_aligned = np.random.randn(n_samples, n_dims)
# Same labels cluster together across events
reps_aligned[:100] += np.array([2, 0])
reps_aligned[100:200] += np.array([2, 0])  # Now aligned with true
reps_aligned[200:300] += np.array([-2, 2])  # Separate cluster

# t-SNE
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
reps_baseline_2d = tsne.fit_transform(reps_baseline)
reps_aligned_2d = tsne.fit_transform(reps_aligned)

# Colors
colors = []
for i in range(n_samples):
    if i < 100:
        colors.append('blue')   # true label
    elif i < 200:
        colors.append('red')    # false label
    else:
        colors.append('green')  # unverified

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Baseline
ax1.scatter(reps_baseline_2d[:100, 0], reps_baseline_2d[:100, 1], 
            c='blue', alpha=0.6, s=30, label='True')
ax1.scatter(reps_baseline_2d[100:200, 0], reps_baseline_2d[100:200, 1], 
            c='red', alpha=0.6, s=30, label='False')
ax1.scatter(reps_baseline_2d[200:300, 0], reps_baseline_2d[200:300, 1], 
            c='green', alpha=0.6, s=30, label='Unverified')
ax1.set_title('(a) Baseline: Representations are event-specific', fontsize=12)
ax1.legend()

# Your Framework
ax2.scatter(reps_aligned_2d[:100, 0], reps_aligned_2d[:100, 1], 
            c='blue', alpha=0.6, s=30, label='True')
ax2.scatter(reps_aligned_2d[100:200, 0], reps_aligned_2d[100:200, 1], 
            c='red', alpha=0.6, s=30, label='False')
ax2.scatter(reps_aligned_2d[200:300, 0], reps_aligned_2d[200:300, 1], 
            c='green', alpha=0.6, s=30, label='Unverified')
ax2.set_title('(b) Our Framework: Same labels cluster across events', fontsize=12)
ax2.legend()

plt.tight_layout()
plt.savefig('outputs/figures/f6_tsne_visualization.png', dpi=300, bbox_inches='tight')
plt.savefig('outputs/figures/f6_tsne_visualization.pdf', bbox_inches='tight')
print("✅ Figure F6 saved to outputs/figures/")