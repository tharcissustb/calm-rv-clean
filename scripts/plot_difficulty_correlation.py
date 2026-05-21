"""
Plot Difficulty-F1 Correlation for Paper 2
Exports to outputs/figures/ and outputs/tables/
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

# Create directories if they don't exist
Path("outputs/figures").mkdir(parents=True, exist_ok=True)
Path("outputs/tables").mkdir(parents=True, exist_ok=True)

# Load your results
with open('outputs/event_difficulty_v4.json', 'r') as f:
    difficulty_data = json.load(f)

# Actual F1 from your Excel (LOEO results)
actual_f1 = {
    "charliehebdo": 0.313,
    "ebola-essien": 0.148,
    "ferguson": 0.046,
    "germanwings-crash": 0.229,
    "gurlitt": 0.068,
    "ottawashooting": 0.300,
    "prince-toronto": 0.072,
    "putinmissing": 0.045,
    "sydneysiege": 0.299
}

# Prepare data for table
table_data = []
events = []
difficulties = []
f1_scores = []

for event, f1 in actual_f1.items():
    if event in difficulty_data:
        events.append(event)
        diff = difficulty_data[event]['difficulty']
        f1_val = f1
        difficulties.append(diff)
        f1_scores.append(f1_val)
        
        # For table
        table_data.append({
            'Event': event.capitalize(),
            'Difficulty Score': f"{diff:.4f}",
            'LOEO Macro-F1': f"{f1_val:.3f}",
            'Performance': 'FAIL' if f1_val < 0.10 else 'PASS',
            'N_Test': difficulty_data[event]['n_test']
        })

# Create table
df = pd.DataFrame(table_data)
df_sorted = df.sort_values('Difficulty Score', ascending=False)

# Save table to CSV
csv_path = 'outputs/tables/event_difficulty_f1_table.csv'
df_sorted.to_csv(csv_path, index=False)
print(f"Table saved to: {csv_path}")

# Save as formatted text table
txt_path = 'outputs/tables/event_difficulty_f1_table.txt'
with open(txt_path, 'w') as f:
    f.write("="*70 + "\n")
    f.write("TABLE 1: Event Difficulty vs. Cross-Event Generalization Performance\n")
    f.write("="*70 + "\n\n")
    f.write(df_sorted.to_string(index=False))
    f.write("\n\n" + "="*70 + "\n")
    f.write(f"Correlation coefficient: {np.corrcoef(difficulties, f1_scores)[0,1]:.3f}\n")
    f.write("Note: Difficulty > 0.002 correlates with F1 < 0.08 (model failure)\n")
print(f"Text table saved to: {txt_path}")

# Create figure
fig, ax = plt.subplots(figsize=(10, 6))

# Scatter plot with color by F1
scatter = ax.scatter(difficulties, f1_scores, s=200, c=f1_scores, 
                     cmap='RdYlGn_r', vmin=0, vmax=0.35, edgecolors='black', linewidth=1.5)

# Add colorbar
cbar = plt.colorbar(scatter)
cbar.set_label('LOEO Macro-F1', fontsize=11)

# Add labels
for i, event in enumerate(events):
    ax.annotate(event.capitalize(), (difficulties[i], f1_scores[i]), 
                xytext=(8, 8), textcoords='offset points',
                fontsize=10, fontweight='bold')

# Add trend line
z = np.polyfit(difficulties, f1_scores, 1)
p = np.poly1d(z)
ax.plot(difficulties, p(difficulties), "r--", alpha=0.8, linewidth=2, 
        label=f'Trend: F1 = {z[0]:.0f}×Diff + {z[1]:.2f}')

# Add threshold lines
ax.axvline(x=0.002, color='gray', linestyle=':', alpha=0.7, linewidth=2, 
           label='Difficulty threshold (0.002)')
ax.axhline(y=0.15, color='gray', linestyle=':', alpha=0.7, linewidth=2)

# Labels and title
ax.set_xlabel('Event Difficulty Score\n(Higher = More Distributionally Different)', fontsize=12)
ax.set_ylabel('LOEO Macro-F1 (Cross-Event Performance)', fontsize=12)
ax.set_title('Cross-Event Generalization Failure in Rumour Verification\nHigher Event Difficulty → Lower Model Performance', fontsize=14)

# Highlight hard events region
hard_events = ['ferguson', 'gurlitt', 'prince-toronto', 'putinmissing']
for event in hard_events:
    idx = events.index(event) if event in events else None
    if idx:
        ax.annotate('⚠️ MODEL FAILS\n(F1 < 0.08)', (difficulties[idx], f1_scores[idx]),
                    xytext=(25, -30), textcoords='offset points',
                    color='red', fontsize=9, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

# Easy events annotation
easy_events = ['charliehebdo', 'sydneysiege']
for event in easy_events:
    idx = events.index(event) if event in events else None
    if idx:
        ax.annotate('✓ WORKS WELL', (difficulties[idx], f1_scores[idx]),
                    xytext=(-50, 10), textcoords='offset points',
                    color='green', fontsize=9, fontweight='bold')

ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)

# Set axis limits
ax.set_xlim(0.0016, 0.0028)
ax.set_ylim(-0.02, 0.38)

plt.tight_layout()

# Save figure in multiple formats
fig.savefig('outputs/figures/difficulty_f1_correlation.png', dpi=300, bbox_inches='tight')
fig.savefig('outputs/figures/difficulty_f1_correlation.pdf', bbox_inches='tight')
fig.savefig('outputs/figures/difficulty_f1_correlation.svg', bbox_inches='tight')
print(f"Figures saved to: outputs/figures/")

plt.show()

# Print statistics
print("\n" + "="*50)
print("STATISTICAL SUMMARY")
print("="*50)
print(f"Correlation coefficient: {np.corrcoef(difficulties, f1_scores)[0,1]:.3f}")
print(f"Hard events (F1 < 0.10): {sum(1 for f in f1_scores if f < 0.10)}")
print(f"Hard events avg difficulty: {np.mean([d for d,f in zip(difficulties, f1_scores) if f < 0.10]):.4f}")
print(f"Easy events (F1 > 0.25): {sum(1 for f in f1_scores if f > 0.25)}")
print(f"Easy events avg difficulty: {np.mean([d for d,f in zip(difficulties, f1_scores) if f > 0.25]):.4f}")
print(f"\nConclusion: Difficulty threshold ≈ 0.002 separates success from failure")