"""
Repeated dismantling experiment: 3 network types × 9 scales × 20 trials × 6 methods.
Incremental save to prevent data loss on interruption.
Output: results_repeated_main.csv
"""
import os
import sys
import time
from pathlib import Path
import numpy as np
import networkx as nx
import pandas as pd
from network_dismantling.unified_interface import dismantle

np.random.seed(42)

METHODS = ['degree', 'CI_L2', 'CoreHD', 'GND', 'GDM', 'GDM+R']
NETWORKS = ['BA', 'ER', 'WS']
SIZES = [50, 100, 200, 300, 400, 500, 1000]
N_TRIALS = 20
STOP = 0.1
OUT_FILE = Path(__file__).resolve().parent.parent / 'results' / 'results_repeated_main.csv'
CHECKPOINT_EVERY = 50  # save every N rows


def generate_network(net_type, n, seed):
    if net_type == 'BA':
        return nx.barabasi_albert_graph(n, 4, seed=seed)
    elif net_type == 'ER':
        p = 8.0 / (n - 1)
        return nx.erdos_renyi_graph(n, p, seed=seed)
    elif net_type == 'WS':
        return nx.watts_strogatz_graph(n, 8, 0.3, seed=seed)
    else:
        raise ValueError(f"Unknown network type: {net_type}")


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
    slcc_peak = max(trace_slcc) if trace_slcc else 0
    return auc, rem_num, slcc_peak


def main():
    total_tasks = len(NETWORKS) * len(SIZES) * N_TRIALS * len(METHODS)
    completed = 0
    rows = []

    # Resume from existing file if present
    if os.path.exists(OUT_FILE):
        df_existing = pd.read_csv(OUT_FILE)
        rows = df_existing.to_dict('records')
        completed = len(rows)
        print(f"Resumed from {OUT_FILE}: {completed}/{total_tasks} already done.")

    # Build set of completed keys
    done_keys = set()
    for r in rows:
        key = (r['network_type'], r['n'], r['trial'], r['method'])
        done_keys.add(key)

    print(f"Running {total_tasks} experiments. Methods: {METHODS}")
    print(f"Networks: {NETWORKS}, Sizes: {SIZES}, Trials: {N_TRIALS}")
    print("=" * 70)

    t_start = time.time()
    for net_type in NETWORKS:
        for n in SIZES:
            for trial in range(N_TRIALS):
                G = generate_network(net_type, n, 10000 + trial)
                for method in METHODS:
                    key = (net_type, n, trial, method)
                    if key in done_keys:
                        continue

                    t0 = time.time()
                    kwargs = {'seed': 10000 + trial} if method in ['random', 'EGND'] else {}
                    seq = dismantle(G, method, stop_condition=STOP, **kwargs)
                    t1 = time.time()

                    auc, rem_num, slcc_peak = evaluate_sequence(G, seq)
                    rows.append({
                        'network_type': net_type,
                        'n': n,
                        'trial': trial,
                        'method': method,
                        'auc': auc,
                        'rem_num': rem_num,
                        'slcc_peak': slcc_peak,
                        'time_sec': t1 - t0,
                    })
                    completed += 1

                    if completed % CHECKPOINT_EVERY == 0:
                        pd.DataFrame(rows).to_csv(OUT_FILE, index=False)
                        elapsed = time.time() - t_start
                        eta = elapsed / completed * (total_tasks - completed)
                        print(f"  Progress: {completed}/{total_tasks} ({100*completed/total_tasks:.1f}%) | "
                              f"Elapsed: {elapsed/60:.1f}min | ETA: {eta/60:.1f}min")

    # Final save
    pd.DataFrame(rows).to_csv(OUT_FILE, index=False)
    print(f"\nAll done! Results saved to {OUT_FILE}")
    print(f"Total time: {(time.time() - t_start)/60:.1f} minutes")


if __name__ == '__main__':
    main()
