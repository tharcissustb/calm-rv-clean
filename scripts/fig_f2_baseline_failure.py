"""
Figure F2: Baseline LOEO Performance Across All 9 Events
Shows catastrophic failure on hard events
"""

import matplotlib.pyplot as plt
import numpy as np

# Data from your LOEO baseline (from your Excel)
events = ['charliehebdo', 'ebola-essien', 'ferguson', 'germanwings-crash', 
          'gurlitt', 'ottawashooting', 'prince-toronto', 'putinmissing', 'sydneysiege']

f1_scores = [0.313, 0.148, 0.046, 0.229, 0.068, 0.300, 0.072, 0.045, 0.299]

# Color coding: Green = Easy (F1 > 0.25), Red = Hard (F1 < 0.10)
colors = ['green' if f > 0.25 else 'red' for f in f1_scores]

plt.figure(figsize=(12, 6))
bars = plt.bar(events, f1_scores, color=colors, edgecolor='black', linewidth=1.5)

# Add threshold line
plt.axhline(y=0.25, color='blue', linestyle='--', linewidth=2, label='Easy/Hard Threshold')
plt.axhline(y=0.10, color='orange', linestyle=':', linewidth=2, label='Random Guessing ≈ 0.33')

# Labels and title
plt.xlabel('Event', fontsize=12)
plt.ylabel('Macro-F1 Score', fontsize=12)
plt.title('Figure F2: Baseline LOEO Performance Reveals Catastrophic Failure\non Hard Events', fontsize=14)

# Add value labels on bars
for bar, f1 in zip(bars, f1_scores):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
             f'{f1:.3f}', ha='center', va='bottom', fontsize=9)

plt.xticks(rotation=45, ha='right')
plt.ylim(0, 0.45)
plt.legend(loc='upper right')
plt.tight_layout()

plt.savefig('outputs/figures/f2_baseline_failure.png', dpi=300, bbox_inches='tight')
plt.savefig('outputs/figures/f2_baseline_failure.pdf', bbox_inches='tight')
plt.savefig('outputs/figures/f2_baseline_failure.svg', bbox_inches='tight')

print("✅ Figure F2 saved to outputs/figures/")