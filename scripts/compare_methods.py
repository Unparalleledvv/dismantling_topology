"""
Compare all dismantling methods on benchmark networks.
Produces a table of rem_num (nodes removed until LCC <= 0.1*n) and r_auc.
"""
import time
import numpy as np
import networkx as nx
import pandas as pd
from network_dismantling.unified_interface import dismantle, METHOD_REGISTRY


def compute_r_auc(lcc_trace, n):
    """Compute relative AUC: area under the LCC fraction curve normalized by max area."""
    if len(lcc_trace) == 0:
        return 0.0
    # LCC fraction after each removal step
    lcc_frac = np.array(lcc_trace) / n
    # Trapezoidal integration
    auc = np.trapz(lcc_frac, dx=1.0)
    # Normalize by max possible area (n * 1.0)
    max_auc = len(lcc_trace)
    if max_auc == 0:
        return 0.0
    r_auc = auc / max_auc
    return r_auc


def evaluate_method(G, method, stop_condition, seed=42, repeats=1):
    """Run dismantling and collect metrics."""
    n = G.number_of_nodes()
    rem_nums = []
    r_aucs = []
    times = []

    for r in range(repeats):
        kwargs = {}
        if method in ["random", "EGND", "CoreHD", "GND", "EI_s1", "EI_s2"]:
            kwargs["seed"] = seed + r

        t0 = time.time()
        try:
            seq = dismantle(G, method=method, stop_condition=stop_condition, **kwargs)
        except Exception as e:
            print(f"  ERROR {method}: {e}")
            return None
        t1 = time.time()

        # Progressive LCC trace
        G_tmp = G.copy()
        lcc_trace = []
        for node in seq:
            if node in G_tmp:
                G_tmp.remove_node(node)
            if G_tmp.number_of_nodes() > 0:
                components = list(nx.connected_components(G_tmp))
                lcc = max(len(c) for c in components) if components else 0
            else:
                lcc = 0
            lcc_trace.append(lcc)

        # Find rem_num: first step where LCC <= stop_condition
        rem_num = n
        for i, lcc in enumerate(lcc_trace):
            if lcc <= stop_condition:
                rem_num = i + 1
                break

        r_auc = compute_r_auc(lcc_trace, n)
        rem_nums.append(rem_num)
        r_aucs.append(r_auc)
        times.append(t1 - t0)

    return {
        "rem_num": np.mean(rem_nums),
        "r_auc": np.mean(r_aucs),
        "time": np.mean(times),
    }


def compare_all():
    # Define benchmark networks
    networks = {
        "BA_50": nx.barabasi_albert_graph(50, 3, seed=1),
        "BA_100": nx.barabasi_albert_graph(100, 3, seed=2),
        "ER_50": nx.erdos_renyi_graph(50, 0.08, seed=3),
        "ER_100": nx.erdos_renyi_graph(100, 0.08, seed=4),
        "WS_50": nx.watts_strogatz_graph(50, 6, 0.3, seed=5),
    }

    # Exclude brute_force (too slow) and methods that need special handling
    methods = [m for m in METHOD_REGISTRY.keys() if m != "brute_force"]
    # Random methods get more repeats
    stochastic = {"random", "EGND"}

    records = []

    for net_name, G in networks.items():
        n = G.number_of_nodes()
        stop_condition = max(1, int(0.10 * n))
        print(f"\n=== Network: {net_name} (n={n}, stop={stop_condition}) ===")

        for method in methods:
            repeats = 5 if method in stochastic else 1
            print(f"  Running {method} (repeats={repeats})...", end="", flush=True)
            result = evaluate_method(G, method, stop_condition, repeats=repeats)
            if result is not None:
                print(f" rem_num={result['rem_num']:.1f}, r_auc={result['r_auc']:.4f}, time={result['time']:.3f}s")
                records.append({
                    "network": net_name,
                    "method": method,
                    "rem_num": result["rem_num"],
                    "r_auc": result["r_auc"],
                    "time_s": result["time"],
                })
            else:
                print(" FAILED")

    df = pd.DataFrame(records)

    # Pivot tables
    print("\n\n" + "="*80)
    print("RESULTS: rem_num (lower is better)")
    print("="*80)
    pivot_rem = df.pivot(index="method", columns="network", values="rem_num")
    print(pivot_rem.round(1).to_string())

    print("\n" + "="*80)
    print("RESULTS: r_auc (lower is better)")
    print("="*80)
    pivot_auc = df.pivot(index="method", columns="network", values="r_auc")
    print(pivot_auc.round(4).to_string())

    print("\n" + "="*80)
    print("RESULTS: time (seconds)")
    print("="*80)
    pivot_time = df.pivot(index="method", columns="network", values="time_s")
    print(pivot_time.round(3).to_string())

    # Save
    df.to_csv("method_comparison.csv", index=False)
    print("\nSaved detailed results to method_comparison.csv")


if __name__ == "__main__":
    compare_all()
