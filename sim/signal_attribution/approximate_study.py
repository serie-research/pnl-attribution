"""Reproducible synthetic comparison of cached coalition-value estimators.

Run with ``python -m sim.signal_attribution.approximate_study OUT``.
Settings are written before any simulation. Exact tables use one solver call
per nonempty coalition; repeated estimators replay those deterministic values
through independent empty caches, retaining the actual distinct-query cost.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import json
import hashlib
from pathlib import Path
import time

import numpy as np

from .attribution import shapley
from .desk import Desk, simulate
from .optimiser import pnl, solve
from .units import IC6, NAMES6, RHO6

SEEDS = [0, 1, 2, 3, 4]
REPEATS = 200
DRAWS = [1, 2, 4, 8, 16, 32, 64, 128, 256]


def desks():
    base = Desk()
    six = base.with_(ic=IC6, rho=RHO6, names=NAMES6)
    eight = base.with_(ic=IC6 + (0.04, 0.03), rho=RHO6 + (0.9, 0.9),
                      names=NAMES6 + ("mid-slow A", "mid-slow B"))
    return {4: base, 6: six, 8: eight}


def settings():
    return dict(seeds=SEEDS, repeats=REPEATS, draws=DRAWS,
                desks={str(k): asdict(d) for k, d in desks().items()},
                games=["target", "gamma"],
                groups="Last word of signal name: family A or family B",
                metric="0.5 * sum(abs(estimate-exact)) / abs(full cumulative net P&L)",
                band="5th to 95th percentiles over 200 independent draws, per history",
                history="Median pairwise half-L1 distance between exact shares over five seeds",
                useful="First tested draw count whose worst-history q95 error is below history/3",
                timing="One solver call per nonempty coalition. Empty coalition is known zero. Fixed gamma requires the full target run as common setup.",
                sampling="Independent fresh cache per repetition; seeded generators. Cached deterministic table avoids re-running identical expensive histories across repetitions.",
                date="2026-10-29")


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def exact_case(job):
    k, seed, directory = job
    out = Path(directory)
    d = desks()[k]
    started = time.perf_counter()
    market = simulate(d, np.random.default_rng(seed))
    t = time.perf_counter()
    full = solve(d, market.g.sum(0)[None], sigma_target=np.array([d.risk_target]))
    setup = time.perf_counter() - t
    values = {g: np.zeros(2**k) for g in ("target", "gamma")}
    times = {g: np.zeros(2**k) for g in values}
    values["target"][-1] = pnl(d, full, market.r).sum()
    times["target"][-1] = setup
    for mask in range(1, 2**k):
        G = market.g[[i for i in range(k) if mask & (1 << i)]].sum(0)[None]
        for game in values:
            if game == "target" and mask == 2**k - 1:
                continue
            kw = ({"sigma_target": np.array([d.risk_target])} if game == "target"
                  else {"gamma_path": full.gamma})
            t = time.perf_counter()
            values[game][mask] = pnl(d, solve(d, G, **kw), market.r).sum()
            times[game][mask] = time.perf_counter() - t
    result = dict(k=k, seed=seed, full_target_setup_seconds=setup,
                  elapsed_seconds=time.perf_counter() - started,
                  games={g: dict(values=v.tolist(), coalition_seconds=times[g].tolist(),
                                  exact=shapley(v[:, None], k)[:, 0].tolist())
                         for g, v in values.items()})
    write_json(out / f"exact-k{k}-seed{seed}.json", result)
    return result


def quantiles(values):
    return np.quantile(values, [.05, .5, .95]).tolist()


def evaluate(case, game):
    from .approximate import CachedOracle, kernel_shap, order_two, owen, permutation
    k, seed = case["k"], case["seed"]
    table = np.asarray(case["games"][game]["values"])
    times = np.asarray(case["games"][game]["coalition_seconds"])
    exact = np.asarray(case["games"][game]["exact"])
    groups = [[i for i, n in enumerate(desks()[k].names) if n.endswith(f)] for f in ("A", "B")]
    rows, summaries = [], []
    for method in ("permutation", "antithetic", "kernel", "order_two", "owen"):
        for m in (DRAWS if method in ("permutation", "antithetic", "kernel") else [0]):
            subset = []
            for repeat in range(REPEATS if m else 1):
                oracle = CachedOracle(k, lambda mask: table[mask])
                t = time.perf_counter()
                draw_seed = 1_000_000 * seed + 10_000 * k + repeat
                if method == "permutation":
                    result = permutation(oracle, m, seed=draw_seed)
                elif method == "antithetic":
                    result = permutation(oracle, m, seed=draw_seed, antithetic=True)
                elif method == "kernel":
                    result = kernel_shap(oracle, m, seed=draw_seed)
                elif method == "order_two":
                    result = order_two(oracle)
                else:
                    result = owen(oracle, groups)
                overhead = time.perf_counter() - t
                error = np.asarray(result.values) - exact
                # Do not silently normalise a non-efficient approximation.
                l1 = float(0.5 * np.abs(error).sum() / abs(table[-1]))
                keys = list(oracle.cache)
                query_seconds = float(times[keys].sum())
                if game == "gamma":
                    query_seconds += case["full_target_setup_seconds"]
                row = dict(k=k, seed=seed, game=game, method=method, draws=m,
                           repeat=repeat, coalitions=int(result.coalitions),
                           nonempty_backtests=sum(mask != 0 for mask in keys),
                           distance=l1, max_absolute=float(np.abs(error).max()),
                           efficiency_residual=float(np.asarray(result.values).sum()-table[-1]),
                           rank=getattr(result, "rank", None),
                           replay_seconds=overhead,
                           equivalent_backtest_seconds=query_seconds,
                           equivalent_total_seconds=query_seconds+overhead)
                rows.append(row)
                subset.append(row)
            summaries.append(dict(k=k, seed=seed, game=game, method=method, draws=m,
                                  **{name: quantiles([r[name] for r in subset]) for name in
                                     ("coalitions", "distance", "max_absolute", "equivalent_total_seconds", "efficiency_residual")},
                                  rank_deficient=sum(r["rank"] is not None and r["rank"] < k-1 for r in subset),
                                  repeats=len(subset)))
    return rows, summaries


def report(out, cases, summaries):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullFormatter, ScalarFormatter
    histories = []
    recommendations = []
    lines = ["# Approximate Shapley: empirical tables", "", "Synthetic data; all five histories (seeds 0–4) retained. R=200 draws per random method and draw count. Bands are p05–p95, not confidence intervals for the median.", "", "Shares and distance use the full portfolio cumulative net P&L denominator, matching article §3.5. Max absolute error is in cumulative synthetic P&L units. Owen distance compares conventions; it is not sampling error. Order-2 need not reconcile the original game.", ""]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharey=False)
    styles = {"permutation":"o", "antithetic":"s", "kernel":"^", "order_two":"D", "owen":"X"}
    for col, k in enumerate((4, 6, 8)):
        for ri, game in enumerate(("target", "gamma")):
            relevant = [c for c in cases if c["k"] == k]
            shares = [np.asarray(c["games"][game]["exact"])/c["games"][game]["values"][-1] for c in relevant]
            pairwise = [float(.5*np.abs(a-b).sum()) for i,a in enumerate(shares) for b in shares[i+1:]]
            history = float(np.median(pairwise))
            histories.append(dict(k=k, game=game, median_pairwise_distance=history,
                                  all_pairwise_distances=pairwise,
                                  exact_share_min=np.min(shares,axis=0).tolist(),
                                  exact_share_max=np.max(shares,axis=0).tolist()))
            lines.extend([f"## K={k}, {game}", "", f"History spread: median pairwise distance {history:.6f}; range {min(pairwise):.6f}–{max(pairwise):.6f}. Exact cost: {2**k} distinct coalition values, {2**k-1} nonempty solver backtests.", "", "| Method | Draws | Median unique cost (seed range) | Median distance (seed range) | Worst seed p95 distance | Median max abs (seed range) |", "|---|---:|---:|---:|---:|---:|"])
            ax=axes[ri,col]
            for method, marker in styles.items():
                xx, yy, low, high, deficient=[],[],[],[],[]
                for m in (DRAWS if method in ("permutation", "antithetic", "kernel") else [0]):
                    rs=[r for r in summaries if r["k"]==k and r["game"]==game and r["method"]==method and r["draws"]==m]
                    deficient.append(sum(r["rank_deficient"] for r in rs))
                    costs=[r["coalitions"][1] for r in rs]
                    med=[r["distance"][1] for r in rs]
                    maxabs=[r["max_absolute"][1] for r in rs]
                    xx.append(float(np.median(costs)));yy.append(float(np.median(med)))
                    low.append(min(r["distance"][0] for r in rs));high.append(max(r["distance"][2] for r in rs))
                    lines.append(f"| {method} | {m or 'exact convention'} | {np.median(costs):.1f} ({min(costs):.1f}–{max(costs):.1f}) | {np.median(med):.6f} ({min(med):.6f}–{max(med):.6f}) | {high[-1]:.6f} | {np.median(maxabs):.4f} ({min(maxabs):.4f}–{max(maxabs):.4f}) |")
                ax.plot(xx, np.maximum(yy,1e-16), marker=marker,label=method)
                ax.fill_between(xx,np.maximum(low,1e-16),np.maximum(high,1e-16),alpha=.12)
                if method in ("permutation", "antithetic", "kernel"):
                    good=next((i for i,h in enumerate(high) if h<history/3 and deficient[i]==0), None)
                    recommendations.append(dict(k=k,game=game,method=method,
                                                draws=None if good is None else DRAWS[good],
                                                median_unique_cost=None if good is None else xx[good],
                                                worst_seed_p95=None if good is None else high[good],
                                                threshold=history/3))
            ax.axvline(2**k,color="black",ls=":",label="exact cost")
            ax.axhline(history,color="gray",ls="--",label="history spread")
            ax.set(xscale="log",yscale="log",title=f"K={k}, {game}",xlabel="Distinct coalition values",ylabel="Half L1 share distance")
            ax.set_xticks({4: [4, 8, 16], 6: [4, 8, 16, 32, 64], 8: [4, 16, 64, 256]}[k])
            ax.xaxis.set_major_formatter(ScalarFormatter())
            ax.xaxis.set_minor_formatter(NullFormatter())
            ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8)
    fig.suptitle("Synthetic coalition games: median errors and across-history p05–p95 envelope")
    fig.tight_layout()
    fig.savefig(out/"error-budget.png",dpi=180)
    fig.savefig(out/"error-budget.pdf")
    plt.close(fig)
    lines.extend(["", "## Measured backtest timings", "", "Three independent histories were processed concurrently. Times below are measured wall times under that load, not isolated CPU benchmarks. The fixed-gamma setup is the full target backtest; it is included once in each fixed-gamma estimator equivalent time. Repeats replay cached deterministic values, so equivalent total time sums measured costs of queried nonempty coalitions and measured estimator overhead. It is not the actual elapsed time of re-running every repetition.", "", "| K | Seed | Game | Backtest seconds p05 / median / p95 | All coalition solver seconds | Full-target setup seconds |", "|---:|---:|---|---|---:|---:|"])
    for case in cases:
        for game, data in case["games"].items():
            tt=np.asarray(data["coalition_seconds"])[1:]
            q=" / ".join(f"{v:.6g}" for v in quantiles(tt))
            lines.append(f"| {case['k']} | {case['seed']} | {game} | {q} | {tt.sum():.6g} | {case['full_target_setup_seconds']:.6g} |")
    lines.extend(["", "## Per-history tables", "", "Each tuple below is p05 / median / p95 over repeated draws. Deterministic methods have identical quantiles.", "", "| K | Game | Seed | Method | Draws | Unique cost | Distance | Max absolute | Equivalent total seconds | Rank-deficient draws |", "|---:|---|---:|---|---:|---|---|---|---|---:|"])
    for r in summaries:
        fmt=lambda field: " / ".join(f"{v:.6g}" for v in r[field])
        lines.append(f"| {r['k']} | {r['game']} | {r['seed']} | {r['method']} | {r['draws']} | {fmt('coalitions')} | {fmt('distance')} | {fmt('max_absolute')} | {fmt('equivalent_total_seconds')} | {r['rank_deficient']} |")
    (out/"study-tables.md").write_text("\n".join(lines)+"\n")
    write_json(out/"history-and-useful-budgets.json",dict(history=histories,useful_budgets=recommendations))


def public_summary(out, destination):
    """Export declared settings and all summaries, omitting large raw draws."""
    sources = [Path(__file__), Path(__file__).with_name("approximate.py"),
               Path(__file__).with_name("desk.py"), Path(__file__).with_name("optimiser.py")]
    payload = dict(
        settings=json.loads((out/"study-settings.json").read_text()),
        summaries=json.loads((out/"sampling-summary.json").read_text()),
        history_and_useful_budgets=json.loads((out/"history-and-useful-budgets.json").read_text()),
        timing=json.loads((out/"study-timing.json").read_text()),
        exact_cases=[json.loads(p.read_text()) for p in sorted(out.glob("exact-k*-seed*.json"))],
        provenance={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        notes=["Useful-budget recommendations also require zero rank-deficient repeats; deficient draws remain in all error summaries.",
               "Wall backtest timings measured with three histories concurrently; equivalent estimator totals reconstruct queried costs plus replay overhead.",
               "Owen is an exact grouped convention, not an approximation of Shapley.",
               "Empty coalition is known zero and counts as one distinct value but no optimiser backtest.",
               "Only K=4,6,8 measured. Five histories are an exploratory spread comparison, not a confidence interval for population uncertainty."])
    destination.parent.mkdir(parents=True,exist_ok=True)
    write_json(destination,payload)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("out",type=Path)
    p.add_argument("--workers",type=int,default=3)
    p.add_argument("--public-json",type=Path,help="Optional portable per-seed summary for the page data directory")
    p.add_argument("--reuse-exact",action="store_true",help="Reuse exact tables matching this run's immutable settings")
    args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    config=settings()
    settings_path=args.out/"study-settings.json"
    if settings_path.exists():
        if json.loads(settings_path.read_text()) != config:
            raise ValueError("Existing declaration differs; select a new output directory")
    else:
        write_json(settings_path,config)
    t=time.perf_counter()
    cases=[]
    jobs=[]
    for k in (4,6,8):
        for seed in SEEDS:
            path=args.out/f"exact-k{k}-seed{seed}.json"
            if args.reuse_exact and path.exists():
                cases.append(json.loads(path.read_text()))
            else:
                jobs.append((k,seed,str(args.out)))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for case in pool.map(exact_case,jobs):
            cases.append(case)
            print(f"Exact K={case['k']} seed={case['seed']}: {case['elapsed_seconds']:.2f}s",flush=True)
    cases.sort(key=lambda c:(c["k"],c["seed"]))
    rows=[]; summaries=[]
    for case in cases:
        for game in ("target","gamma"):
            rr,ss=evaluate(case,game)
            rows.extend(rr);summaries.extend(ss)
            print(f"Sampled K={case['k']} seed={case['seed']} {game}",flush=True)
    write_json(args.out/"sampling-raw.json",rows)
    write_json(args.out/"sampling-summary.json",summaries)
    report(args.out,cases,summaries)
    write_json(args.out/"study-timing.json",dict(total_wall_seconds=time.perf_counter()-t,
                                              reused_exact=args.reuse_exact,
                                              computed_exact_cases=len(jobs)))
    if args.public_json:
        public_summary(args.out,args.public_json)


if __name__ == "__main__":
    main()
