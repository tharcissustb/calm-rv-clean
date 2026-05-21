"""
Figure F3: In-domain Ablation Results
Baseline vs Align+Aug vs Full Framework
"""

import matplotlib.pyplot as plt
import numpy as np

experiments = ['Baseline', 'Align+Aug', 'Full Framework']
f1_scores = [0.6236, 0.6635, 0.6919]
accuracy = [0.6466, 0.6778, 0.7027]
ece = [0.1094, 0.2459, 0.2028]

x = np.arange(len(experiments))
width = 0.25

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Subplot 1: F1 and Accuracy
bars1 = ax1.bar(x - width/2, f1_scores, width, label='Macro-F1', color='steelblue', edgecolor='black')
bars2 = ax1.bar(x + width/2, accuracy, width, label='Accuracy', color='lightblue', edgecolor='black')

ax1.set_xlabel('Experiment', fontsize=12)
ax1.set_ylabel('Score', fontsize=12)
ax1.set_title('(a) Performance Improvement', fontsize=12)
ax1.set_xticks(x)
ax1.set_xticklabels(experiments)
ax1.set_ylim(0.6, 0.75)
ax1.legend()

# Add value labels
for bar in bars1:
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003, 
             f'{bar.get_height():.4f}', ha='center', va='bottom', fontsize=9)
for bar in bars2:
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003, 
             f'{bar.get_height():.4f}', ha='center', va='bottom', fontsize=9)

# Subplot 2: ECE (lower is better)
bars3 = ax2.bar(experiments, ece, color='coral', edgecolor='black')
ax2.set_xlabel('Experiment', fontsize=12)
ax2.set_ylabel('Expected Calibration Error (ECE)', fontsize=12)
ax2.set_title('(b) Calibration Error (lower is better)', fontsize=12)
ax2.set_ylim(0, 0.30)

for bar, e in zip(bars3, ece):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
             f'{e:.4f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('outputs/figures/f3_in_domain_ablation.png', dpi=300, bbox_inches='tight')
plt.savefig('outputs/figures/f3_in_domain_ablation.pdf', bbox_inches='tight')
print("✅ Figure F3 saved to outputs/figures/")