"""
Evaluate all non-FINDER methods on BA and ER networks of various scales.
Part 1: Single-run comparison on large networks
Part 2: Multi-run (10 instances) for statistical testing on BA-100 / ER-100
"""
import time
from pathlib import Path
import numpy as np
import networkx as nx
import pandas as pd
from scipy.stats import wilcoxon
from network_dismantling.unified_interface import dismantle

np.random.seed(42)


def evaluate_sequence(G, seq, stop_condition=0.1):
    n = G.number_of_nodes()
    stop = max(1, int(stop_condition * n))

    G_tmp = G.copy()
    trace_lcc = []
    trace_slcc = []
    rem_num = n
    removed = set()

    for i, node in enumerate(seq):
        if node in G_tmp:
            G_tmp.remove_node(node)
            removed.add(node)

        if G_tmp.number_of_nodes() == 0:
            trace_lcc.append(0.0)
            trace_slcc.append(0)
            if rem_num == n:
                rem_num = i + 1
            break

        components = sorted((len(c) for c in nx.connected_components(G_tmp)), reverse=True)
        lcc = components[0] if len(components) > 0 else 0
        slcc = components[1] if len(components) > 1 else 0

        trace_lcc.append(lcc / n)
        trace_slcc.append(slcc)

        if lcc <= stop and rem_num == n:
            rem_num = i + 1

    # Complete with remaining nodes
    remaining = [v for v in G.nodes() if v not in removed]
    if remaining:
        deg_map = dict(G.degree(remaining))
        remaining.sort(key=lambda v: -deg_map[v])
        for node in remaining:
            if node in G_tmp:
                G_tmp.remove_node(node)
            if G_tmp.number_of_nodes() == 0:
                trace_lcc.append(0.0)
                trace_slcc.append(0)
                break
            components = sorted((len(c) for c in nx.connected_components(G_tmp)), reverse=True)
            lcc = components[0] if len(components) > 0 else 0
            slcc = components[1] if len(components) > 1 else 0
            trace_lcc.append(lcc / n)
            trace_slcc.append(slcc)

    while len(trace_lcc) < n:
        trace_lcc.append(0.0)
        trace_slcc.append(0)

    q_vals = np.arange(1, len(trace_lcc) + 1) / n
    auc = np.trapz(trace_lcc, q_vals)

    q_c_opt = 1.0
    for i, lcc in enumerate(trace_lcc):
        if lcc == 0:
            q_c_opt = (i + 1) / n
            break

    slcc_peak = max(trace_slcc) if trace_slcc else 0
    q_at_slcc_peak = (trace_slcc.index(slcc_peak) + 1) / n if trace_slcc else 0

    return {
        'rem_num': rem_num,
        'auc': auc,
        'q_c_opt': q_c_opt,
        'slcc_peak': slcc_peak,
        'q_at_slcc_peak': q_at_slcc_peak,
    }


def run_single_instance(G, method, stop_condition=0.1):
    t0 = time.time()
    kwargs = {'seed': 42} if method in ['random', 'EGND'] else {}
    seq = dismantle(G, method, stop_condition=stop_condition, **kwargs)
    t1 = time.time()
    metrics = evaluate_sequence(G, seq)
    metrics['time_sec'] = t1 - t0
    return metrics


def part1_single_run():
    """Single-run comparison on large networks."""
    methods = ['degree', 'CI_L2', 'CoreHD', 'GND', 'GDM', 'GDM+R']
    configs = [
        ('BA_50', lambda: nx.barabasi_albert_graph(50, 4, seed=42)),
        ('BA_100', lambda: nx.barabasi_albert_graph(100, 4, seed=42)),
        ('BA_500', lambda: nx.barabasi_albert_graph(500, 4, seed=42)),
        ('BA_1000', lambda: nx.barabasi_albert_graph(1000, 4, seed=42)),
        ('ER_50', lambda: nx.erdos_renyi_graph(50, 0.08, seed=42)),
        ('ER_500', lambda: nx.erdos_renyi_graph(500, 0.016, seed=42)),  # ~avg deg 8
        ('ER_1000', lambda: nx.erdos_renyi_graph(1000, 0.008, seed=42)),  # ~avg deg 8
    ]

    rows = []
    for net_name, gen_fn in configs:
        G = gen_fn()
        print(f"\n=== {net_name} (n={G.number_of_nodes()}, m={G.number_of_edges()}) ===")
        for method in methods:
            try:
                metrics = run_single_instance(G, method)
                metrics.update({'network': net_name, 'method': method, 'trial': 0})
                rows.append(metrics)
                print(f"  {method:8s}: rem={metrics['rem_num']:4d}, AUC={metrics['auc']:.4f}, "
                      f"SLCC={metrics['slcc_peak']:3d}, time={metrics['time_sec']:.3f}s")
            except Exception as e:
                print(f"  {method:8s}: ERROR - {e}")

    df = pd.DataFrame(rows)
    out_path = Path(__file__).resolve().parent.parent / 'results' / 'results_all_networks.csv'
    df.to_csv(out_path, index=False)
    print(f"\nPart 1 results saved to {out_path}")
    return df


def part2_statistical_test(n_trials=10):
    """Multi-run for Wilcoxon test on BA-100 and ER-100."""
    methods = ['degree', 'CI_L2', 'CoreHD', 'GDM', 'GDM+R']
    # Exclude GND from statistical test due to slow speed
    configs = [
        ('BA_100', lambda s: nx.barabasi_albert_graph(100, 4, seed=s)),
        ('ER_100', lambda s: nx.erdos_renyi_graph(100, 0.08, seed=s)),
    ]

    rows = []
    for net_name, gen_fn in configs:
        print(f"\n=== Statistical test: {net_name} ({n_trials} trials) ===")
        for trial in range(n_trials):
            G = gen_fn(1000 + trial)
            for method in methods:
                try:
                    metrics = run_single_instance(G, method)
                    metrics.update({'network': net_name, 'method': method, 'trial': trial})
                    rows.append(metrics)
                except Exception as e:
                    print(f"  {method} trial {trial}: ERROR - {e}")

    df = pd.DataFrame(rows)
    out_path = Path(__file__).resolve().parent.parent / 'results' / 'results_statistical_test.csv'
    df.to_csv(out_path, index=False)
    print(f"\nPart 2 results saved to {out_path}")

    # Wilcoxon signed-rank tests
    print("\n" + "=" * 80)
    print("WILCOXON SIGNED-RANK TEST RESULTS")
    print("=" * 80)
    print("H0: Two methods have the same dismantling efficiency (AUC)")
    print("Significance level: alpha = 0.05")
    print("-" * 80)

    baseline = 'CI_L2'
    for net_name in ['BA_100', 'ER_100']:
        print(f"\n{net_name} (comparing against {baseline}):")
        sub = df[df['network'] == net_name]
        base_vals = sub[sub['method'] == baseline][['trial', 'auc']].set_index('trial')['auc']

        for method in methods:
            if method == baseline:
                continue
            method_vals = sub[sub['method'] == method][['trial', 'auc']].set_index('trial')['auc']
            common = base_vals.index.intersection(method_vals.index)
            if len(common) < 3:
                print(f"  {method:8s}: insufficient data (n={len(common)})")
                continue
            try:
                stat, p = wilcoxon(base_vals.loc[common].values, method_vals.loc[common].values)
                sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
                mean_diff = method_vals.loc[common].mean() - base_vals.loc[common].mean()
                print(f"  {method:8s}: p={p:.4f} {sig:3s}  mean AUC diff = {mean_diff:+.4f} "
                      f"({method} {'worse' if mean_diff < 0 else 'better'})")
            except Exception as e:
                print(f"  {method:8s}: test failed - {e}")

    return df


def main():
    print("PART 1: Single-run comparison on all network scales")
    print("=" * 80)
    df1 = part1_single_run()

    print("\n\nPART 2: Statistical testing (10 trials)")
    print("=" * 80)
    df2 = part2_statistical_test(n_trials=10)

    # Summary table for Part 1
    print("\n\n" + "=" * 80)
    print("SUMMARY: Single-run results (sorted by AUC, lower = better)")
    print("=" * 80)
    for net in df1['network'].unique():
        print(f"\n{net}:")
        sub = df1[df1['network'] == net].sort_values('auc')
        print(sub[['method', 'rem_num', 'auc', 'slcc_peak', 'time_sec']].to_string(index=False))


if __name__ == '__main__':
    main()
