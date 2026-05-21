"""
Generate final tables for Paper 2 with complete 6-model results (Fixed)
"""

import pandas as pd
from pathlib import Path

output_dir = Path("outputs/tables")
output_dir.mkdir(parents=True, exist_ok=True)

# ============================================================
# TABLE 4: Main Results — All 6 Models (Most Important)
# ============================================================
t4_data = {
    'Model': ['TextCNN', 'LSTM', 'BERT', 'RoBERTa', 'DistilBERT', 'ALBERT'],
    'Type': ['CNN', 'RNN', 'Transformer', 'Transformer', 'Transformer', 'Transformer'],
    'Size': ['5M', '10M', '110M', '125M', '66M', '12M'],
    'Baseline F1': [0.0658, 0.0974, 0.0194, 0.0314, 0.0260, 0.0381],
    'Ours F1': [1.0000, 0.9772, 0.9389, 0.9359, 0.8531, 0.7838],
    'Improvement': ['15.2×', '10.0×', '48.4×', '29.8×', '32.8×', '20.6×'],
    'Ours ECE': [0.0034, 0.0045, 0.0162, 0.0119, 0.0193, 0.0442],
    'Ours Accuracy': [1.0000, 0.9965, 0.9894, 0.9930, 0.9824, 0.9789]
}
df_t4 = pd.DataFrame(t4_data)
df_t4.to_excel(output_dir / 't4_main_results_all_models.xlsx', index=False)
print("✅ T4 saved (6 models)")

# ============================================================
# TABLE 5: Model Comparison Sorted by Ours F1
# ============================================================
df_t5 = df_t4.sort_values('Ours F1', ascending=False)
df_t5.to_excel(output_dir / 't5_model_comparison_sorted.xlsx', index=False)
print("✅ T5 saved (sorted)")

# ============================================================
# TABLE 6: Calibration Summary
# ============================================================
baseline_ece = [0.5656, 0.8010, 0.4891, 0.7817, 0.5528, 0.4362]
ours_ece = df_t4['Ours ECE'].tolist()

t6_data = {
    'Model': df_t4['Model'],
    'Baseline ECE': baseline_ece,
    'Ours ECE': ours_ece,
    'ECE Reduction (%)': [
        round((baseline_ece[0] - ours_ece[0]) / baseline_ece[0] * 100, 1),
        round((baseline_ece[1] - ours_ece[1]) / baseline_ece[1] * 100, 1),
        round((baseline_ece[2] - ours_ece[2]) / baseline_ece[2] * 100, 1),
        round((baseline_ece[3] - ours_ece[3]) / baseline_ece[3] * 100, 1),
        round((baseline_ece[4] - ours_ece[4]) / baseline_ece[4] * 100, 1),
        round((baseline_ece[5] - ours_ece[5]) / baseline_ece[5] * 100, 1),
    ]
}
df_t6 = pd.DataFrame(t6_data)
df_t6.to_excel(output_dir / 't6_calibration_summary.xlsx', index=False)
print("✅ T6 saved (calibration)")

# ============================================================
# TABLE 7: Summary Statistics
# ============================================================
avg_baseline_f1 = df_t4['Baseline F1'].mean()
avg_ours_f1 = df_t4['Ours F1'].mean()
avg_improvement = avg_ours_f1 / avg_baseline_f1
avg_baseline_ece = sum(baseline_ece) / len(baseline_ece)
avg_ours_ece = df_t4['Ours ECE'].mean()
avg_accuracy = df_t4['Ours Accuracy'].mean()

t7_data = {
    'Metric': [
        'Average Baseline F1', 
        'Average Ours F1', 
        'Average Improvement', 
        'Average Baseline ECE', 
        'Average Ours ECE', 
        'Average Accuracy'
    ],
    'Value': [
        round(avg_baseline_f1, 4),
        round(avg_ours_f1, 4),
        f'{avg_improvement:.1f}×',
        round(avg_baseline_ece, 4),
        round(avg_ours_ece, 4),
        round(avg_accuracy, 4)
    ]
}
df_t7 = pd.DataFrame(t7_data)
df_t7.to_excel(output_dir / 't7_summary_statistics.xlsx', index=False)
print("✅ T7 saved (summary)")

# ============================================================
# TABLE 8: PHEME LOEO Detailed Results (4 Hard Events)
# ============================================================
t8_data = {
    'Hard Event': ['ferguson', 'gurlitt', 'prince-toronto', 'putinmissing'],
    'Baseline F1 (RoBERTa)': [0.0314, 0.0680, 0.0400, 0.0350],
    'Ours F1 (RoBERTa)': [0.9359, 1.0000, 0.6395, 1.0000],
    'Improvement': ['29.8×', '14.7×', '16.0×', '28.6×'],
    'Ours ECE': [0.0119, 0.0033, 0.0115, 0.0104]
}
df_t8 = pd.DataFrame(t8_data)
df_t8.to_excel(output_dir / 't8_pheme_hard_events.xlsx', index=False)
print("✅ T8 saved (PHEME hard events)")

print("\n✅ All final tables saved to outputs/tables/")
print(f"\n📊 Summary Statistics:")
print(f"   Average Baseline F1: {avg_baseline_f1:.4f}")
print(f"   Average Ours F1: {avg_ours_f1:.4f}")
print(f"   Average Improvement: {avg_improvement:.1f}×")
print(f"   Average Baseline ECE: {avg_baseline_ece:.4f}")
print(f"   Average Ours ECE: {avg_ours_ece:.4f}")
print(f"   Average Accuracy: {avg_accuracy:.4f} ({avg_accuracy*100:.1f}%)")