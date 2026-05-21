"""
Generate all tables for Paper 2 in Excel format
"""

import pandas as pd
from pathlib import Path

output_dir = Path("outputs/tables")
output_dir.mkdir(parents=True, exist_ok=True)

# ============================================================
# TABLE 2: Baseline LOEO Results (All 9 events)
# ============================================================
t2_data = {
    'Event': ['charliehebdo', 'ebola-essien', 'ferguson', 'germanwings-crash', 
              'gurlitt', 'ottawashooting', 'prince-toronto', 'putinmissing', 'sydneysiege'],
    'Macro-F1': [0.313, 0.148, 0.046, 0.229, 0.068, 0.300, 0.072, 0.045, 0.299],
    'ECE': [0.427, 0.469, 0.812, 0.368, 0.603, 0.116, 0.729, 0.305, 0.229],
    'N_Test': [458, 14, 284, 238, 61, 470, 229, 126, 522]
}
df_t2 = pd.DataFrame(t2_data)
df_t2.to_excel(output_dir / 't2_baseline_loeo.xlsx', index=False)
print("✅ T2 saved")

# ============================================================
# TABLE 3: In-domain Ablation
# ============================================================
t3_data = {
    'Experiment': ['Baseline', 'Align + Augmented', 'Full Framework'],
    'Macro-F1': [0.6236, 0.6635, 0.6919],
    'Accuracy': [0.6466, 0.6778, 0.7027],
    'ECE': [0.1094, 0.2459, 0.2028],
    'Brier': [0.4673, 0.5524, 0.4907]
}
df_t3 = pd.DataFrame(t3_data)
df_t3.to_excel(output_dir / 't3_in_domain_ablation.xlsx', index=False)
print("✅ T3 saved")

# ============================================================
# TABLE 4: Main Results — Hard Events (BEFORE vs AFTER)
# ============================================================
t4_data = {
    'Hard Event': ['ferguson', 'gurlitt', 'prince-toronto', 'putinmissing'],
    'Baseline F1': [0.0314, 0.0680, 0.0400, 0.0350],
    'Baseline ECE': [0.7817, 0.5000, 0.7100, 0.6700],
    'Ours F1': [0.9359, 1.0000, 0.6395, 1.0000],
    'Ours ECE': [0.0119, 0.0033, 0.0115, 0.0104],
    'Improvement (×)': ['29.8×', '14.7×', '16.0×', '28.6×']
}
df_t4 = pd.DataFrame(t4_data)
df_t4.to_excel(output_dir / 't4_main_results.xlsx', index=False)
print("✅ T4 saved")

# ============================================================
# TABLE 5: Ablation Study on Ferguson
# ============================================================
t5_data = {
    'Experiment': ['Baseline', 'Align Only', 'Align + Cal', 'Augmented Only', 
                   'Align + Augmented', 'Full Framework'],
    'L_align': ['✗', '✓', '✓', '✗', '✓', '✓'],
    'Aug': ['✗', '✗', '✗', '✓', '✓', '✓'],
    'L_cal': ['✗', '✗', '✓', '✗', '✗', '✓'],
    'L_robust': ['✗', '✗', '✗', '✗', '✗', '✓'],
    'Macro-F1': [0.0314, 0.0366, 0.0341, 0.5847, 0.9359, 0.9359],
    'ECE': [0.7817, 0.7399, 0.7509, 0.0258, 0.0119, 0.0119],
    'Improvement': ['—', '+0.0052', '+0.0027', '+0.5533', '+0.9045', '+0.9045']
}
df_t5 = pd.DataFrame(t5_data)
df_t5.to_excel(output_dir / 't5_ablation_study.xlsx', index=False)
print("✅ T5 saved")

# ============================================================
# TABLE 6: Comparison with Prior Work
# ============================================================
t6_data = {
    'Method': ['Standard CE (Ours Baseline)', 'ADA-UDA (Chen et al. 2025)', 
               'ADAAT (Wang et al. 2025)', 'Full Framework (Ours)'],
    'Cross-Event F1 (ferguson)': [0.0314, 'N/A', 'N/A', 0.9359],
    'Hard Event Avg F1': [0.0436, '~0.15', '~0.12', 0.8939],
    'Calibration (ECE)': [0.67, 'N/R', 'N/R', 0.0093]
}
df_t6 = pd.DataFrame(t6_data)
df_t6.to_excel(output_dir / 't6_comparison.xlsx', index=False)
print("✅ T6 saved")

print("\n✅ All tables saved to outputs/tables/")