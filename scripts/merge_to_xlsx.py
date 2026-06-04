"""
Merge repeated experiment results and compute mean ± std.
Output: dismantling_results.xlsx with 4 decimal places.
"""
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Load results
df_main = pd.read_csv(ROOT / 'results' / 'results_repeated_main.csv')
df_finder = pd.read_csv(ROOT / 'results' / 'results_repeated_finder.csv')

# Combine
df = pd.concat([df_main, df_finder], ignore_index=True)

# Compute mean and std for each (network_type, n, method) group
group_cols = ['network_type', 'n', 'method']
metric_cols = ['auc', 'rem_num', 'slcc_peak', 'time_sec']

agg_dict = {}
for col in metric_cols:
    agg_dict[f'{col}_mean'] = (col, 'mean')
    agg_dict[f'{col}_std'] = (col, 'std')

summary = df.groupby(group_cols).agg(**agg_dict).reset_index()

# Round to 4 decimal places
for col in summary.columns:
    if 'mean' in col or 'std' in col:
        summary[col] = summary[col].round(4)

# Create Excel writer
out_file = ROOT / 'results' / 'dismantling_results.xlsx'
with pd.ExcelWriter(out_file, engine='openpyxl') as writer:
    # Sheet 1: Full summary (mean ± std)
    summary.to_excel(writer, sheet_name='Summary_Mean_Std', index=False)

    # Sheet 2: Mean only (clean table for papers)
    mean_cols = [c for c in summary.columns if '_mean' in c]
    mean_df = summary[group_cols + mean_cols].copy()
    mean_df.columns = [c.replace('_mean', '') for c in mean_df.columns]
    mean_df.to_excel(writer, sheet_name='Mean_Only', index=False)

    # Sheet 3: Std only
    std_cols = [c for c in summary.columns if '_std' in c]
    std_df = summary[group_cols + std_cols].copy()
    std_df.columns = [c.replace('_std', '_stddev') for c in std_df.columns]
    std_df.to_excel(writer, sheet_name='Stddev_Only', index=False)

    # Sheet 4: Raw data
    df.to_excel(writer, sheet_name='Raw_Data', index=False)

    # Sheet 5: Pivot table (methods as rows, networks as columns, AUC mean)
    pivot = summary.pivot_table(
        index='method',
        columns=['network_type', 'n'],
        values='auc_mean',
        aggfunc='first'
    )
    pivot.to_excel(writer, sheet_name='AUC_Pivot')

    # Sheet 6: Pivot table for rem_num
    pivot2 = summary.pivot_table(
        index='method',
        columns=['network_type', 'n'],
        values='rem_num_mean',
        aggfunc='first'
    )
    pivot2.to_excel(writer, sheet_name='RemNum_Pivot')

print(f"Results saved to {out_file}")
print(f"Total experiments: {len(df)}")
print(f"Groups: {len(summary)}")

# Print preview
print("\nPreview (AUC mean, top 10 rows):")
preview = summary[['network_type', 'n', 'method', 'auc_mean', 'auc_std', 'rem_num_mean', 'rem_num_std']].head(10)
print(preview.to_string(index=False))

print("\nBest method per network (by AUC mean):")
best = summary.loc[summary.groupby(['network_type', 'n'])['auc_mean'].idxmin()]
print(best[['network_type', 'n', 'method', 'auc_mean', 'rem_num_mean']].to_string(index=False))
