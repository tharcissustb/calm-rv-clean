"""
Generate final tables for Paper 2 with complete 6-model results
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
# TABLE 5: Complete Model Comparison (Sorted by F1)
# ============================================================
t5_data = sorted(t4_data.items(), key=lambda x: x[1] if x[0] == 'Ours F1' else 0)
df_t5 = df_t4.sort_values('Ours F1', ascending=False)
df_t5.to_excel(output_dir / 't5_model_comparison_sorted.xlsx', index=False)
print("✅ T5 saved (sorted)")

# ============================================================
# TABLE 6: Calibration Summary
# ============================================================
t6_data = {
    'Model': df_t4['Model'],
    'Baseline ECE': [0.5656, 0.8010, 0.4891, 0.7817, 0.5528, 0.4362],
    'Ours ECE': df_t4['Ours ECE'],
    'ECE Reduction (%)': [
        (0.5656 - 0.0034) / 0.5656 * 100,
        (0.8010 - 0.0045) / 0.8010 * 100,
        (0.4891 - 0.0162) / 0.4891 * 100,
        (0.7817 - 0.0119) / 0.7817 * 100,
        (0.5528 - 0.0193) / 0.5528 * 100,
        (0.4362 - 0.0442) / 0.4362 * 100,
    ]
}
df_t6 = pd.DataFrame(t6_data)
df_t6['ECE Reduction (%)'] = df_t6['ECE Reduction (%)'].round(1)
df_t6.to_excel(output_dir / 't6_calibration_summary.xlsx', index=False)
print("✅ T6 saved (calibration)")

# ============================================================
# TABLE 7: Summary Statistics
# ============================================================
t7_data = {
    'Metric': ['Average Baseline F1', 'Average Ours F1', 'Average Improvement', 
               'Average Baseline ECE', 'Average Ours ECE', 'Average Accuracy'],
    'Value': [
        df_t4['Baseline F1'].mean(),
        df_t4['Ours F1'].mean(),
        f"{df_t4['Ours F1'].mean() / df_t4['Baseline F1'].mean():.1f}×",
        [0.5656, 0.8010, 0.4891, 0.7817, 0.5528, 0.4362].mean(),
        df_t4['Ours ECE'].mean(),
        df_t4['Ours Accuracy'].mean()
    ]
}
df_t7 = pd.DataFrame(t7_data)
df_t7.to_excel(output_dir / 't7_summary_statistics.xlsx', index=False)
print("✅ T7 saved (summary)")

print("\n✅ All final tables saved to outputs/tables/")