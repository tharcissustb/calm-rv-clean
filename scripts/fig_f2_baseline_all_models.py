"""
Figure F2: Baseline LOEO Performance Across All 6 Models
Shows catastrophic failure before our framework
"""

import matplotlib.pyplot as plt
import numpy as np

models = ['TextCNN', 'LSTM', 'BERT', 'RoBERTa', 'DistilBERT', 'ALBERT']
baseline_f1 = [0.0658, 0.0974, 0.0194, 0.0314, 0.0260, 0.0381]
baseline_ece = [0.5656, 0.8010, 0.4891, 0.7817, 0.5528, 0.4362]

x = np.arange(len(models))
width = 0.35

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Subplot 1: Baseline F1 (all terrible)
colors = ['red' if f < 0.05 else 'orange' for f in baseline_f1]
bars1 = ax1.bar(x, baseline_f1, color=colors, edgecolor='black', linewidth=1.5)
ax1.set_xlabel('Model Architecture', fontsize=12)
ax1.set_ylabel('Macro-F1 Score', fontsize=12)
ax1.set_title('(a) Baseline Performance — Catastrophic Failure on All Models', fontsize=12)
ax1.set_xticks(x)
ax1.set_xticklabels(models, rotation=45, ha='right')
ax1.set_ylim(0, 0.15)
ax1.axhline(y=0.033, color='gray', linestyle='--', label='Random Guessing (0.33)')
ax1.legend()

for bar, f1 in zip(bars1, baseline_f1):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003, 
             f'{f1:.4f}', ha='center', va='bottom', fontsize=9)

# Subplot 2: Baseline ECE (all severely overconfident)
bars2 = ax2.bar(x, baseline_ece, color='coral', edgecolor='black', linewidth=1.5)
ax2.set_xlabel('Model Architecture', fontsize=12)
ax2.set_ylabel('Expected Calibration Error (ECE)', fontsize=12)
ax2.set_title('(b) Baseline Calibration — Severely Overconfident', fontsize=12)
ax2.set_xticks(x)
ax2.set_xticklabels(models, rotation=45, ha='right')
ax2.set_ylim(0, 1.0)

for bar, ece in zip(bars2, baseline_ece):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
             f'{ece:.4f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('outputs/figures/f2_baseline_all_models.png', dpi=300, bbox_inches='tight')
plt.savefig('outputs/figures/f2_baseline_all_models.pdf', bbox_inches='tight')
print("✅ Figure F2 saved (6 models)")