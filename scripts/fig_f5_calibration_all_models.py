"""
Figure F5: Calibration Improvement — ECE Before vs After
"""

import matplotlib.pyplot as plt
import numpy as np

models = ['TextCNN', 'LSTM', 'BERT', 'RoBERTa', 'DistilBERT', 'ALBERT']
baseline_ece = [0.5656, 0.8010, 0.4891, 0.7817, 0.5528, 0.4362]
ours_ece = [0.0034, 0.0045, 0.0162, 0.0119, 0.0193, 0.0442]

x = np.arange(len(models))
width = 0.35

fig, ax = plt.subplots(figsize=(12, 6))

bars1 = ax.bar(x - width/2, baseline_ece, width, label='Baseline', color='red', alpha=0.8, edgecolor='black')
bars2 = ax.bar(x + width/2, ours_ece, width, label='Our Framework', color='green', alpha=0.8, edgecolor='black')

ax.set_xlabel('Model Architecture', fontsize=12)
ax.set_ylabel('Expected Calibration Error (ECE)', fontsize=12)
ax.set_title('Figure F5: Calibration Improvement — Baseline vs Our Framework\n(Lower is Better)', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=45, ha='right')
ax.set_ylim(0, 0.9)
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)

# Add value labels
for bar, val in zip(bars1, baseline_ece):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
            f'{val:.4f}', ha='center', va='bottom', fontsize=8)
for bar, val in zip(bars2, ours_ece):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
            f'{val:.4f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

plt.tight_layout()
plt.savefig('outputs/figures/f5_calibration_all_models.png', dpi=300, bbox_inches='tight')
plt.savefig('outputs/figures/f5_calibration_all_models.pdf', bbox_inches='tight')
print("✅ Figure F5 saved (6 models)")