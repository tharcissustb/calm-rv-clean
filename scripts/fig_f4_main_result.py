"""
Figure F4: Main Result — Hard Events Performance
Baseline vs Full Framework on 4 Hard Events
"""

import matplotlib.pyplot as plt
import numpy as np

events = ['ferguson', 'gurlitt', 'prince-toronto', 'putinmissing']
baseline_f1 = [0.0314, 0.0680, 0.0400, 0.0350]
framework_f1 = [0.9359, 1.0000, 0.6395, 1.0000]

x = np.arange(len(events))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))

bars1 = ax.bar(x - width/2, baseline_f1, width, label='Baseline LOEO', 
               color='red', edgecolor='black', alpha=0.8)
bars2 = ax.bar(x + width/2, framework_f1, width, label='Our Full Framework', 
               color='green', edgecolor='black', alpha=0.8)

# Add improvement arrows and labels
for i, (base, fw) in enumerate(zip(baseline_f1, framework_f1)):
    improvement = fw - base
    ax.annotate(f'+{improvement:.1f}×', 
                xy=(i, fw), xytext=(i, fw + 0.1),
                ha='center', fontsize=10, fontweight='bold',
                color='darkgreen')

# Add threshold line for "good performance"
ax.axhline(y=0.80, color='gray', linestyle=':', linewidth=1.5, alpha=0.7, label='Good Performance Threshold')

ax.set_xlabel('Hard Event', fontsize=12)
ax.set_ylabel('Macro-F1 Score', fontsize=12)
ax.set_title('Figure F4: Cross-Event Generalization — Before vs After Our Framework', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(events)
ax.set_ylim(0, 1.1)
ax.legend(loc='upper left')
ax.grid(True, alpha=0.3)

# Add value labels
for bar, val in zip(bars1, baseline_f1):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
            f'{val:.4f}', ha='center', va='bottom', fontsize=9)
for bar, val in zip(bars2, framework_f1):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
            f'{val:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('outputs/figures/f4_main_result.png', dpi=300, bbox_inches='tight')
plt.savefig('outputs/figures/f4_main_result.pdf', bbox_inches='tight')
print("✅ Figure F4 saved to outputs/figures/")