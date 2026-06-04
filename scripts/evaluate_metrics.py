"""
Evaluate dismantling methods with multiple metrics on BA-50 and BA-100.
"""
import time
import numpy as np
import networkx as nx
import pandas as pd
from network_dismantling.unified_interface import dismantle, METHOD_REGISTRY

np.random.seed(42)

METRICS = ['rem_num', 'auc', 'q_c_opt', 'slcc_peak', 'q_at_slcc_peak', 'time_sec']


def evaluate_method(G, method, stop_condition=0.1):
    n = G.number_of_nodes()
    stop = max(1, int(stop_condition * n))

    # Run dismantling
    t0 = time.time()
    kwargs = {'seed': 42} if method in ['random', 'EGND'] else {}
    seq = dismantle(G, method, stop_condition=stop, **kwargs)
    t1 = time.time()

    # Simulate removal and record trace
    G_tmp = G.copy()
    trace_lcc = []
    trace_slcc = []
    rem_num = n

    for i, node in enumerate(seq):
        if node in G_tmp:
            G_tmp.remove_node(node)

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

    # Pad traces if stopped early
    while len(trace_lcc) < n:
        trace_lcc.append(0.0)
        trace_slcc.append(0)

    # AUC: trapezoidal integration of S(q)/N vs q
    q_vals = np.arange(1, len(trace_lcc) + 1) / n
    auc = np.trapz(trace_lcc, q_vals)

    # q_c_opt: minimum q where LCC = 0
    q_c_opt = 1.0
    for i, lcc in enumerate(trace_lcc):
        if lcc == 0:
            q_c_opt = (i + 1) / n
            break

    # SLCC peak
    slcc_peak = max(trace_slcc) if trace_slcc else 0
    q_at_slcc_peak = (trace_slcc.index(slcc_peak) + 1) / n if trace_slcc else 0

    return {
        'rem_num': rem_num,
        'auc': auc,
        'q_c_opt': q_c_opt,
        'slcc_peak': slcc_peak,
        'q_at_slcc_peak': q_at_slcc_peak,
        'time_sec': t1 - t0,
    }


def main():
    methods = ['degree', 'CI_L2', 'CoreHD', 'GND', 'GDM', 'GDM+R']
    networks = {
        'BA_50': nx.barabasi_albert_graph(50, 4, seed=42),
        'BA_100': nx.barabasi_albert_graph(100, 4, seed=42),
    }

    rows = []
    for net_name, G in networks.items():
        print(f"\n=== {net_name} (n={G.number_of_nodes()}, m={G.number_of_edges()}) ===")
        for method in methods:
            print(f"  Running {method}...")
            metrics = evaluate_method(G, method)
            row = {'network': net_name, 'method': method}
            row.update(metrics)
            rows.append(row)
            print(f"    rem_num={metrics['rem_num']}, AUC={metrics['auc']:.4f}, "
                  f"q_c_opt={metrics['q_c_opt']:.4f}, SLCC_peak={metrics['slcc_peak']}, "
                  f"time={metrics['time_sec']:.3f}s")

    df = pd.DataFrame(rows)
    from pathlib import Path
    out_path = Path(__file__).resolve().parent.parent / 'results' / 'results_metrics.csv'
    df.to_csv(out_path, index=False)
    print(f"\nResults saved to {out_path}")

    # Print summary table
    print("\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)
    for net_name in networks:
        print(f"\n{net_name}:")
        sub = df[df['network'] == net_name].copy()
        sub = sub.sort_values('auc')
        print(sub[['method', 'rem_num', 'auc', 'q_c_opt', 'slcc_peak', 'time_sec']].to_string(index=False))


if __name__ == '__main__':
    main()
