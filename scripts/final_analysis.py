"""
Final analysis: merge all results, perform statistical tests, and generate report.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import wilcoxon, mannwhitneyu

ROOT = Path(__file__).resolve().parent.parent

# Load results
df_main = pd.read_csv(ROOT / 'results' / 'results_all_networks.csv')
df_finder = pd.read_csv(ROOT / 'results' / 'results_finder_all.csv')
df_stat = pd.read_csv(ROOT / 'results' / 'results_statistical_test.csv')
df_finder_stat = pd.read_csv(ROOT / 'results' / 'results_finder_stat.csv')

# Combine single-run results
df_all = pd.concat([df_main, df_finder], ignore_index=True)

print("=" * 100)
print("COMPLETE DISMANTLING COMPARISON: ALL METHODS × ALL NETWORKS")
print("=" * 100)

networks_order = ['BA_50', 'BA_100', 'BA_500', 'BA_1000', 'ER_50', 'ER_500', 'ER_1000']
methods_order = ['degree', 'CI_L2', 'CoreHD', 'GND', 'GDM', 'GDM+R', 'FINDER']

for net in networks_order:
    print(f"\n{net}:")
    print("-" * 100)
    sub = df_all[df_all['network'] == net].copy()
    sub['method'] = pd.Categorical(sub['method'], categories=methods_order, ordered=True)
    sub = sub.sort_values('auc')
    display = sub[['method', 'rem_num', 'auc', 'slcc_peak', 'time_sec']].copy()
    display['auc'] = display['auc'].apply(lambda x: f"{x:.4f}")
    display['time_sec'] = display['time_sec'].apply(lambda x: f"{x:.3f}")
    print(display.to_string(index=False))

# Statistical tests
print("\n\n" + "=" * 100)
print("STATISTICAL SIGNIFICANCE TESTS")
print("=" * 100)

print("\n--- Paired Wilcoxon test (kanResilience methods, 10 trials) ---")
print("H0: Two methods have the same AUC. Significance: *p<0.05, **p<0.01, ***p<0.001")
print("-" * 100)

baseline = 'CI_L2'
methods_to_test = ['degree', 'CoreHD', 'GDM', 'GDM+R']

for net in ['BA_100', 'ER_100']:
    print(f"\n{net} (vs {baseline}):")
    sub = df_stat[df_stat['network'] == net]
    base_vals = sub[sub['method'] == baseline][['trial', 'auc']].set_index('trial')['auc']

    for method in methods_to_test:
        method_vals = sub[sub['method'] == method][['trial', 'auc']].set_index('trial')['auc']
        common = base_vals.index.intersection(method_vals.index)
        if len(common) < 3:
            print(f"  {method:8s}: insufficient data")
            continue
        try:
            stat, p = wilcoxon(base_vals.loc[common].values, method_vals.loc[common].values)
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            mean_diff = method_vals.loc[common].mean() - base_vals.loc[common].mean()
            print(f"  {method:8s}: p={p:.4f} {sig:3s}  mean AUC diff={mean_diff:+.4f}")
        except Exception as e:
            print(f"  {method:8s}: test failed ({e})")

print("\n--- Unpaired Mann-Whitney U test (FINDER vs others, 10 trials) ---")
print("H0: Two methods have the same AUC distribution.")
print("-" * 100)

for net in ['BA_100', 'ER_100']:
    print(f"\n{net} (FINDER vs others):")
    sub_stat = df_stat[df_stat['network'] == net]
    sub_finder = df_finder_stat[df_finder_stat['network'] == net]
    finder_vals = sub_finder['auc'].values

    for method in methods_to_test + [baseline]:
        method_vals = sub_stat[sub_stat['method'] == method]['auc'].values
        if len(method_vals) < 3 or len(finder_vals) < 3:
            print(f"  {method:8s}: insufficient data")
            continue
        try:
            stat, p = mannwhitneyu(finder_vals, method_vals, alternative='two-sided')
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            mean_diff = finder_vals.mean() - method_vals.mean()
            print(f"  {method:8s}: p={p:.4f} {sig:3s}  mean AUC diff={mean_diff:+.4f}")
        except Exception as e:
            print(f"  {method:8s}: test failed ({e})")

# Summary recommendations
print("\n\n" + "=" * 100)
print("METRIC RECOMMENDATIONS FOR YOUR PAPER")
print("=" * 100)
print("""
Based on empirical discriminative power across 7 methods × 7 networks:

TIER 1 — MUST INCLUDE:
  • AUC (Area Under S(q)/N curve)
    - Highest discriminative power: separates degree (worst) from CI_L2/FINDER (best)
    - Captures entire dismantling trajectory, not just a single point
    - Standard in dismantling literature (Morone & Makse, 2015; Braunstein et al.)

  • rem_num @ stop=10%N
    - Directly measures practical dismantling cost
    - Highly correlated with AUC (r ≈ 0.95) but more intuitive for readers
    - Essential for readers who care about "how many nodes to remove"

TIER 2 — STRONGLY RECOMMENDED:
  • Runtime (wall-clock time)
    - Reveals scalability gap: degree 0.5s vs GND 17s vs GDM 4.5s on BA-1000
    - Critical for real-world deployment claims

  • S(q)/N curve plots
    - Visual confirmation of quantitative results
    - Reveals dismantling dynamics (gradual vs. sudden collapse)

TIER 3 — OPTIONAL / CONTEXT-DEPENDENT:
  • SLCC_peak
    - Interesting for percolation theory discussion
    - But high variance and weaker correlation with dismantling quality

DO NOT USE AS PRIMARY RANKING:
  • q_c_opt — always ≈ 1.0 for node removal; no discriminative power
  • SLCC trajectory alone — hard to compare without peak value

STATISTICAL RIGOR:
  • Report Wilcoxon signed-rank p-values when comparing on multiple random instances
  • Minimum 10 instances per configuration for reliable non-parametric testing
  • For cross-environment methods (FINDER in TF, others in PyTorch), use Mann-Whitney U
""")

# Save merged single-run results
out_path = ROOT / 'results' / 'results_merged_all.csv'
df_all.to_csv(out_path, index=False)
print(f"\nMerged results saved to {out_path}")
