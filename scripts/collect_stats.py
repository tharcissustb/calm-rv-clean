"""
Collect results from multiple seeds and compute mean ± std and p-values
"""

import json
from pathlib import Path
import numpy as np
from scipy import stats

def collect_results(exp_name_pattern, metric='macro_f1'):
    """Collect results from multiple seeds"""
    results = []
    for seed in [42, 123, 456]:
        path = Path(f"outputs/runs/{exp_name_pattern}_seed{seed}/results.json")
        if path.exists():
            with open(path) as f:
                data = json.load(f)
                results.append(data[metric])
    return np.mean(results), np.std(results), results

def paired_bootstrap(baseline_list, calmrv_list, n_resamples=10000):
    """Compute p-value for improvement"""
    diffs = np.array(calmrv_list) - np.array(baseline_list)
    n = len(diffs)
    bootstrap_diffs = []
    for _ in range(n_resamples):
        sample = np.random.choice(diffs, n, replace=True)
        bootstrap_diffs.append(np.mean(sample))
    p_value = np.mean(np.array(bootstrap_diffs) <= 0)
    return p_value

# Example usage
baseline = [0.0314, 0.0320, 0.0308]  # Replace with actual seed results
calmrv = [0.9359, 0.9340, 0.9370]     # Replace with actual seed results

mean_b, std_b, list_b = np.mean(baseline), np.std(baseline), baseline
mean_c, std_c, list_c = np.mean(calmrv), np.std(calmrv), calmrv
p_value = paired_bootstrap(list_b, list_c)

print(f"Baseline: {mean_b:.4f} ± {std_b:.4f}")
print(f"CALM-RV: {mean_c:.4f} ± {std_c:.4f}")
print(f"p-value: {p_value:.4f}")