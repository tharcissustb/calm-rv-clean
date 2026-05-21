"""
Figure F4: Main Result — Baseline vs Our Framework Across 6 Models
"""

import matplotlib.pyplot as plt
import numpy as np

models = ['TextCNN', 'LSTM', 'BERT', 'RoBERTa', 'DistilBERT', 'ALBERT']
baseline_f1 = [0.0658, 0.0974, 0.0194, 0.0314, 0.0260, 0.0381]
ours_f1 = [1.0000, 0.9772, 0.9389, 0.9359, 0.8531, 0.7838]

x = np.arange(len(models))
width = 0.35

fig, ax = plt.subplots(figsize=(12, 7))

bars1 = ax.bar(x - width/2, baseline_f1, width, label='Baseline', color='red', alpha=0.8, edgecolor='black')
bars2 = ax.bar(x + width/2, ours_f1, width, label='Our Framework', color='green', alpha=0.8, edgecolor='black')

# Add improvement annotations
for i, (base, ours) in enumerate(zip(baseline_f1, ours_f1)):
    improvement = ours / base if base > 0 else 0
    ax.annotate(f'{improvement:.0f}×', 
                xy=(i, ours), xytext=(i, ours + 0.05),
                ha='center', fontsize=9, fontweight='bold', color='darkgreen')

ax.set_xlabel('Model Architecture', fontsize=12)
ax.set_ylabel('Macro-F1 Score', fontsize=12)
ax.set_title('Figure F4: Cross-Event Generalization — Baseline vs Our Framework\n(Ferguson LOEO)', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.set_ylim(0, 1.1)
ax.legend(loc='upper left')
ax.axhline(y=0.80, color='gray', linestyle=':', alpha=0.7, label='Good Performance Threshold')
ax.grid(True, alpha=0.3)

# Add value labels
for bar, val in zip(bars1, baseline_f1):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
            f'{val:.4f}', ha='center', va='bottom', fontsize=8)
for bar, val in zip(bars2, ours_f1):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
            f'{val:.4f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

plt.tight_layout()
plt.savefig('outputs/figures/f4_main_all_models.png', dpi=300, bbox_inches='tight')
plt.savefig('outputs/figures/f4_main_all_models.pdf', bbox_inches='tight')
print("✅ Figure F4 saved (6 models)")