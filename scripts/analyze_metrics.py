"""
Merge and analyze dismantling metrics across all methods.
"""
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Load results
df_main = pd.read_csv(ROOT / 'results' / 'results_metrics_v2.csv')
df_finder = pd.read_csv(ROOT / 'results' / 'results_finder_metrics.csv')

# Combine
df = pd.concat([df_main, df_finder], ignore_index=True)

# Select relevant columns
cols = ['network', 'method', 'rem_num', 'auc', 'q_c_opt', 'slcc_peak', 'time_sec']
df = df[cols]

print("=" * 90)
print("DISMANTLING METHODS COMPARISON ON BA NETWORKS")
print("=" * 90)

for net in ['BA_50', 'BA_100']:
    print(f"\n{net} (stop = 10% of nodes):")
    print("-" * 90)
    sub = df[df['network'] == net].copy()
    sub = sub.sort_values('auc')
    print(sub.to_string(index=False))

print("\n" + "=" * 90)
print("METRIC ANALYSIS: WHICH ONES ARE MOST USEFUL?")
print("=" * 90)

print("""
1. AUC (Area Under the S(q)/N curve)
   ----------------------------------
   • Definition: Integral of max-connected-component ratio vs. removal fraction
   • Interpretation: LOWER = better (faster dismantling)
   • Pros: Comprehensive, captures entire dismantling trajectory
   • Cons: Sensitive to tail behavior; methods with same rem_num can differ in AUC
   • Verdict: ⭐⭐⭐⭐⭐ PRIMARY METRIC — standard in the dismantling literature

2. rem_num (removal count to reach stop_condition)
   ------------------------------------------------
   • Definition: Number of nodes removed until LCC ≤ 10% N
   • Interpretation: SMALLER = better
   • Pros: Directly measures practical dismantling cost
   • Cons: Depends on stop_condition choice
   • Verdict: ⭐⭐⭐⭐⭐ PRIMARY METRIC — intuitive and task-relevant

3. q_c_opt (critical removal fraction for LCC=0)
   ----------------------------------------------
   • Definition: Minimum q where LCC drops to 0
   • Interpretation: SMALLER = better
   • Pros: Theoretical significance in percolation
   • Cons: In practice, all methods require q ≈ 1.0 to eliminate ALL components
           (isolated nodes remain until the very end)
   • Verdict: ⭐⭐  LOW DISCRIMINATIVE POWER for node-removal dismantling

4. SLCC_peak (peak of second-largest component)
   ---------------------------------------------
   • Definition: Maximum size of 2nd-LCC during dismantling
   • Interpretation: Related to percolation phase transition
   • Pros: Theoretically meaningful; large peak = sharp transition
   • Cons: High variance across networks; less directly tied to dismantling quality
   • Verdict: ⭐⭐⭐  SUPPLEMENTARY — use for percolation analysis, not ranking

5. time_sec (runtime)
   -------------------
   • Definition: Wall-clock time for full dismantling
   • Interpretation: SMALLER = better scalability
   • Pros: Practical relevance for large networks
   • Cons: Implementation-dependent
   • Verdict: ⭐⭐⭐⭐  IMPORTANT for real-world deployment

6. S(q)/N curve (LCC trajectory)
   ------------------------------
   • Definition: Plot of max-CC ratio vs. removal fraction
   • Pros: Visual, reveals dismantling dynamics
   • Cons: Hard to compare quantitatively
   • Verdict: ⭐⭐⭐  USE FOR FIGURES, not as primary ranking
""")

print("=" * 90)
print("RECOMMENDED METRIC SET FOR YOUR PAPER")
print("=" * 90)
print("""
Primary (for ranking methods):
  • AUC      — overall dismantling efficiency
  • rem_num  — practical removal cost @ target LCC

Secondary (for characterization):
  • Runtime  — scalability assessment
  • S(q)/N curve plots — visual comparison

Optional (for theoretical depth):
  • SLCC_peak and SLCC trajectory — percolation transition analysis

Avoid as primary ranking:
  • q_c_opt  — lacks discriminative power in this setting
""")
