"""Figures of S1. Reads every parquet under ``results`` (any subset of units)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

from .experiment import ITERATED_VARIANTS  # noqa: E402
from .units import BASE_AXES  # noqa: E402

plt.rcParams.update({
    "font.size": 13, "axes.titlesize": 13, "axes.labelsize": 13, "legend.fontsize": 12,
    "xtick.labelsize": 12, "ytick.labelsize": 12, "axes.spines.top": False,
    "axes.spines.right": False,
})

METHOD_STYLE = {
    "durin": ("Durin (exact)", "black", "o"),
    "grinold_easton": ("Grinold-Easton", "tab:purple", "s"),
    "iterated": ("Iterated stand-alone proxies (NNLS, α=1)", "tab:red", "D"),
    "regression": ("regression, 1 step", "tab:orange", "v"),
    "shapley_target": ("Shapley, risk target", "tab:blue", "^"),
    "shapley_gamma": ("Shapley fixed γ", "tab:cyan", "<"),
    "singleton_target": ("stand-alone, risk target", "tab:gray", "x"),
    "singleton_gamma": ("stand-alone, fixed γ", "tab:olive", "+"),
}
AXIS_LABEL = {
    "corr": "innovation correlation between signals",
    "box": "position bound (equal-risk position units)",
    "spread": "spread cost λ₀ (std of G)",
    "k": "number of signals K",
}
SIGNAL_COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:brown", "tab:pink"]


def _load(results: Path) -> pl.DataFrame:
    files = sorted(Path(results).rglob("*.parquet"))
    if not files:
        return pl.DataFrame()
    return pl.concat([pl.read_parquet(f) for f in files], how="vertical_relaxed")


def _stamp(fig, df: pl.DataFrame, configs: list[str] | None = None):
    d = df if configs is None else df.filter(pl.col("config").is_in(configs))
    n_u = d["unit"].n_unique() if len(d) else 0
    n_s = d.select("unit", "seed").unique().height if len(d) else 0
    fig.text(0.995, 0.005, f"{n_u} units, {n_s} seeds", ha="right", va="bottom", fontsize=12, color="0.4")


def _save(fig, figdir: Path, name: str) -> list[Path]:
    figdir.mkdir(parents=True, exist_ok=True)
    out = []
    for ext in ("png", "svg"):
        p = figdir / f"{name}.{ext}"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        if ext == "svg":
            p.write_text("\n".join(line.rstrip() for line in p.read_text().splitlines()) + "\n")
        out.append(p)
    plt.close(fig)
    return out


def shares(df: pl.DataFrame) -> pl.DataFrame:
    """Attributed cumulative P&L / total cumulative P&L, per unit, seed, method, signal."""
    cum = df.filter((pl.col("metric") == "cum") & (pl.col("t") == -1))
    tot = cum.filter(pl.col("method") == "total").select("unit", "seed", total=pl.col("value"))
    return (
        cum.filter(pl.col("method") != "total")
        .join(tot, on=["unit", "seed"])
        .with_columns(share=pl.col("value") / pl.col("total"))
    )


def _axis_rows(df: pl.DataFrame, axis: str) -> pl.DataFrame:
    """Rows of a sweep axis, with the base configuration placed at its value."""
    a = df.filter(pl.col("axis") == axis)
    b = df.filter(pl.col("axis") == "base").with_columns(
        axis=pl.lit(axis), axis_value=pl.lit(float(BASE_AXES[axis]))
    )
    return pl.concat([a, b], how="vertical_relaxed")


def _axis_positions(values: np.ndarray, axis: str):
    """Categorical positions; the box axis is ordered from loose to tight."""
    vals = sorted(set(values.tolist()), reverse=(axis == "box"))
    labels = ["∞" if np.isinf(v) else f"{v:g}" for v in vals]
    return {v: i for i, v in enumerate(vals)}, labels


def _mean_sd(d: pl.DataFrame, by: list[str], col: str) -> pl.DataFrame:
    return d.group_by(by).agg(m=pl.col(col).mean(), sd=pl.col(col).std(), n=pl.len()).sort(by)


# ---------------------------------------------------------------------------


def fig_b2(df, figdir):
    d = df.filter((pl.col("config") == "b2") & (pl.col("metric") == "cum_path"))
    if d.is_empty():
        return []
    seed = d["seed"].min()
    d = d.filter(pl.col("seed") == seed)
    names = [n for n in d.filter(pl.col("method") == "durin")["signal"].unique(maintain_order=True)]

    def path(meth, sig):
        p = d.filter((pl.col("method") == meth) & (pl.col("signal") == sig)).sort("t")
        return p["t"].to_numpy(), p["value"].to_numpy()

    t, total = path("total", "_all")
    panels = [
        ("grinold_easton", "Grinold-Easton: signals + cost and constraint portfolios",
         names + ["spread", "constraint"]),
        ("durin", "Durin: constraints and spread inside each signal", names),
        ("iterated", "Iterated stand-alone proxies (NNLS, α=1)", names + ["_resid"]),
        ("regression", "regression on stand-alone P&L, one step", names + ["_resid"]),
    ]
    colors = dict(zip(names, SIGNAL_COLORS)) | {"spread": "tab:green", "constraint": "tab:red", "_resid": "0.5"}
    label = {"_resid": "residual", "spread": "spread portfolio", "constraint": "constraint portfolio"}
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), sharex=True)
    for ax, (meth, title, sigs) in zip(axes.flat, panels):
        ax.plot(t, total, color="black", lw=2.2, label="total")
        for s in sigs:
            tt, y = path(meth, s)
            if len(y):
                ax.plot(tt, y, color=colors[s], lw=1.6, ls="--" if s == "_resid" else "-",
                        label=label.get(s, s))
        if meth in ("grinold_easton", "durin"):
            ssum = sum(path(meth, s)[1] for s in sigs)
            err = np.abs(ssum - total).max()
            ax.text(0.02, 0.96, f"max |sum − total| = {err:.1e}", transform=ax.transAxes,
                    va="top", fontsize=12, color="0.3")
        ax.set_title(title, fontsize=12.5)
        ax.axhline(0, color="0.8", lw=0.8)
        ax.legend(loc="lower left" if meth == "grinold_easton" else "upper left",
                  fontsize=12, frameon=False, ncol=2 if meth == "grinold_easton" else 1,
                  bbox_to_anchor=(0, 0) if meth == "grinold_easton" else (0, 0.9))
    for ax in axes[1]:
        ax.set_xlabel("period")
    for ax in axes[:, 0]:
        ax.set_ylabel("cumulative net P&L (σ* units)")
    _stamp(fig, df.filter((pl.col("config") == "b2") & (pl.col("seed") == seed)))
    fig.tight_layout()
    return _save(fig, figdir, "B2-durin-split")


def fig_band(df, figdir):
    d = df.filter((pl.col("config") == "b2") & (pl.col("method") == "band"))
    if d.is_empty():
        return []
    w = d.pivot(on="metric", index=["seed", "signal", "t"], values="value")
    a, b, tr = w["a"].to_numpy(), w["b"].to_numpy(), w["trade"].to_numpy() > 0
    c = w["c"].to_numpy() if "c" in w.columns else np.zeros_like(a)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8), gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axes[0]
    lim = max(3.0, np.quantile(np.abs(np.r_[a, b]), 0.995))
    xx = np.array([-lim, lim])
    ax.fill_between(xx, -1 - xx, 1 - xx, color="0.88", label="no-trade band |a + b| ≤ 1")
    ax.scatter(a[~tr], b[~tr], s=7, color="0.35", label="no trade", alpha=0.6)
    ax.scatter(a[tr], b[tr], s=7, color="tab:red", label="trade", alpha=0.6)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("fast signal / λ₀")
    ax.set_ylabel("(slow signal − γσ²x₋₁) / λ₀")
    ax.legend(loc="upper right", fontsize=12, framealpha=0.9, markerscale=3)
    # vetoes: no trade although one signal alone would have traded
    # each signal alone, same previous position: |g^k - gamma sigma^2 x_{t-1}| > lambda_0
    nt = ~tr
    fast_alone = np.abs(a - c) > 1
    slow_alone = np.abs(b) > 1
    cats = ["neither alone", "fast alone", "slow alone", "both alone"]
    cnt = [
        (nt & ~fast_alone & ~slow_alone).sum(), (nt & fast_alone & ~slow_alone).sum(),
        (nt & ~fast_alone & slow_alone).sum(), (nt & fast_alone & slow_alone).sum(),
    ]
    axes[1].bar(cats, np.array(cnt) / max(nt.sum(), 1), color=["0.6", "tab:blue", "tab:orange", "tab:purple"])
    axes[1].set_ylabel("share of no-trade periods")
    axes[1].set_title("no-trade periods: which signal alone\nwould have cleared the spread", fontsize=12.5)
    axes[1].tick_params(axis="x", rotation=20)
    _stamp(fig, df.filter(pl.col("config") == "b2"))
    fig.tight_layout()
    return _save(fig, figdir, "B6-no-trade-band")


def fig_sweep_agreement(df, figdir):
    sh = shares(df)
    rv = df.filter((pl.col("metric") == "var_share") & (pl.col("signal") == "_resid"))
    axes_present = [a for a in AXIS_LABEL if not _axis_rows(sh, a).filter(pl.col("axis") == a).is_empty()
                    and df.filter(pl.col("axis") == a).height]
    if not axes_present:
        return []
    meths = ["regression", "iterated", "shapley_target", "shapley_gamma"]
    fig, axs = plt.subplots(len(axes_present), 3, figsize=(15, 3.6 * len(axes_present)), squeeze=False)
    for row, axis in enumerate(axes_present):
        s = _axis_rows(sh, axis)
        pos, labels = _axis_positions(s["axis_value"].to_numpy(), axis)
        # (a) residual variance share
        r = _mean_sd(_axis_rows(rv, axis), ["axis_value", "method"], "value")
        ax = axs[row, 0]
        for meth in ["regression", "iterated", "singleton_gamma"]:
            q = r.filter(pl.col("method") == meth)
            if q.is_empty():
                continue
            lab, col, mk = METHOD_STYLE[meth]
            if meth == "singleton_gamma":
                lab = "Σ stand-alone (fixed γ) = interaction"
            x = [pos[v] for v in q["axis_value"]]
            ax.errorbar(x, q["m"], q["sd"], color=col, marker=mk, label=lab, capsize=3)
        ax.set_ylabel("residual variance / total")
        # (b) distance to Durin: 1/2 sum_k |share_M,k - share_D,k|
        dur = s.filter(pl.col("method") == "durin").select("unit", "seed", "signal", d=pl.col("share"))
        dist = (
            s.filter(pl.col("method").is_in(meths)).join(dur, on=["unit", "seed", "signal"])
            .group_by("axis_value", "unit", "seed", "method")
            .agg(dist=0.5 * (pl.col("share") - pl.col("d")).abs().sum())
        )
        dd = _mean_sd(dist, ["axis_value", "method"], "dist")
        ax = axs[row, 1]
        for meth in meths:
            q = dd.filter(pl.col("method") == meth)
            lab, col, mk = METHOD_STYLE[meth]
            x = [pos[v] for v in q["axis_value"]]
            ax.errorbar(x, q["m"], q["sd"], color=col, marker=mk, label=lab, capsize=3)
        ax.set_ylabel("distance to Durin shares")
        # (c) stability: across-seed std of the share, averaged over signals
        st = (
            s.filter(pl.col("method").is_in(meths + ["durin"]))
            .group_by("axis_value", "method", "signal").agg(sd=pl.col("share").std())
            .group_by("axis_value", "method").agg(sd=pl.col("sd").mean()).sort("axis_value")
        )
        ax = axs[row, 2]
        for meth in ["durin"] + meths:
            q = st.filter(pl.col("method") == meth)
            lab, col, mk = METHOD_STYLE[meth]
            x = [pos[v] for v in q["axis_value"]]
            ax.plot(x, q["sd"], color=col, marker=mk, label=lab)
        ax.set_ylabel("seed spread of shares")
        for ax in axs[row]:
            ax.set_xticks(range(len(labels)), labels)
            ax.set_xlabel(AXIS_LABEL[axis])
            ax.set_ylim(bottom=0)
    axs[0, 0].legend(fontsize=12, frameon=False)
    axs[0, 2].legend(fontsize=12, frameon=False)
    _stamp(fig, df.filter(pl.col("axis").is_in(axes_present + ["base"])))
    fig.tight_layout()
    return _save(fig, figdir, "S1-sweep-agreement")


def fig_sweep_shares(df, figdir):
    sh = shares(df)
    axes_present = [a for a in AXIS_LABEL if df.filter(pl.col("axis") == a).height]
    if not axes_present:
        return []
    meths = ["durin", "iterated", "shapley_target", "shapley_gamma", "regression"]
    fig, axs = plt.subplots(1, len(axes_present), figsize=(4.6 * len(axes_present), 4.8), squeeze=False)
    for ax, axis in zip(axs[0], axes_present):
        s = _axis_rows(sh, axis)
        pos, labels = _axis_positions(s["axis_value"].to_numpy(), axis)
        for sig, ls in (("fast A", "-"), ("slow A", "--")):
            q = _mean_sd(s.filter((pl.col("signal") == sig) & pl.col("method").is_in(meths)),
                         ["axis_value", "method"], "share")
            for j, meth in enumerate(meths):
                qq = q.filter(pl.col("method") == meth)
                lab, col, mk = METHOD_STYLE[meth]
                x = np.array([pos[v] for v in qq["axis_value"]]) + (j - 2) * 0.05
                ax.errorbar(x, qq["m"], qq["sd"], color=col, marker=mk, ls=ls, capsize=2,
                            label=lab if sig == "fast A" else None)
        ax.set_xticks(range(len(labels)), labels)
        ax.set_xlabel(AXIS_LABEL[axis])
        ax.axhline(0, color="0.8", lw=0.8)
    axs[0, 0].set_ylabel("share of total P&L\n(solid: fast A, dashed: slow A)")
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.12),
               ncol=len(meths), fontsize=12, frameon=False)
    _stamp(fig, df.filter(pl.col("axis").is_in(axes_present + ["base"])))
    fig.tight_layout()
    return _save(fig, figdir, "S1-sweep-shares")


def fig_known(df, figdir):
    designs = [("zeroic", "zero-IC signal"), ("identical", "two identical signals"),
               ("valmom", "value + momentum, long only")]
    designs = [(c, t) for c, t in designs if df.filter(pl.col("config") == c).height]
    if not designs:
        return []
    sh = shares(df)
    meths = ["durin", "grinold_easton", "shapley_gamma", "shapley_target", "singleton_gamma",
             "iterated", "regression"]
    fig, axs = plt.subplots(1, len(designs), figsize=(5.6 * len(designs), 5.4), squeeze=False)
    for ax, (cfg, title) in zip(axs[0], designs):
        s = sh.filter(pl.col("config") == cfg)
        sigs = [x for x in s.filter(pl.col("method") == "durin")["signal"].unique(maintain_order=True)]
        q = _mean_sd(s, ["method", "signal"], "share")
        width = 0.8 / len(meths)
        for j, meth in enumerate(meths):
            lab, col, _ = METHOD_STYLE[meth]
            vals = [q.filter((pl.col("method") == meth) & (pl.col("signal") == g)) for g in sigs]
            m = [v["m"][0] if v.height else np.nan for v in vals]
            e = [v["sd"][0] if v.height else 0 for v in vals]
            ax.bar(np.arange(len(sigs)) + (j - len(meths) / 2 + 0.5) * width, m, width, yerr=e,
                   color=col, label=lab, capsize=2)
        ax.set_xticks(range(len(sigs)), sigs)
        ax.axhline(0, color="0.5", lw=0.8)
        r = df.filter((pl.col("config") == cfg) & (pl.col("metric") == "var_share")
                      & pl.col("method").is_in(["regression", "iterated"]))
        txt = "\n".join(f"{'1-step NNLS' if m == 'regression' else 'Iterated stand-alone proxies (α=1)'}: {r.filter(pl.col('method') == m)['value'].mean():.2f}"
                        for m in ["regression", "iterated"] if r.filter(pl.col("method") == m).height)
        ax.set_title(f"{title}\nresidual variance share\n{txt}", fontsize=12)
    axs[0, 0].set_ylabel("share of total P&L")
    h, lab = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=len(meths), fontsize=12, frameon=False,
               bbox_to_anchor=(0.5, -0.06))
    _stamp(fig, df.filter(pl.col("config").is_in([c for c, _ in designs])))
    fig.tight_layout()
    return _save(fig, figdir, "S1-known-answers")


def fig_residual_interaction(df, figdir):
    axes_present = [a for a in ("box", "corr", "spread") if df.filter(pl.col("axis") == a).height]
    if not axes_present:
        return []
    sh = shares(df)
    fig, axs = plt.subplots(2, len(axes_present), figsize=(5.2 * len(axes_present), 8), squeeze=False)
    for c, axis in enumerate(axes_present):
        rv = _axis_rows(df.filter((pl.col("metric") == "var_share") & (pl.col("signal") == "_resid")), axis)
        pos, labels = _axis_positions(rv["axis_value"].to_numpy(), axis)
        r = _mean_sd(rv, ["axis_value", "method"], "value")
        ax = axs[0, c]
        for meth, lab in [("regression", "regression, 1 step"), ("iterated", "Iterated stand-alone proxies (NNLS, α=1)"),
                          ("singleton_gamma", "v(N) − Σ v({k}), fixed γ")]:
            q = r.filter(pl.col("method") == meth)
            _, col, mk = METHOD_STYLE[meth]
            ax.errorbar([pos[v] for v in q["axis_value"]], q["m"], q["sd"], color=col, marker=mk,
                        label=lab, capsize=3)
        ax.set_ylabel("residual variance / total")
        # interaction mass sum_k |phi_k - v({k})| / |v(N)|
        s = _axis_rows(sh, axis)
        ax = axs[1, c]
        for conv, col in (("gamma", "tab:cyan"), ("target", "tab:blue")):
            phi = s.filter(pl.col("method") == f"shapley_{conv}").select("unit", "seed", "signal", "axis_value", phi="share")
            v1 = s.filter(pl.col("method") == f"singleton_{conv}").select("unit", "seed", "signal", v1="share")
            im = (phi.join(v1, on=["unit", "seed", "signal"])
                  .group_by("axis_value", "unit", "seed").agg(mass=(pl.col("phi") - pl.col("v1")).abs().sum()))
            q = _mean_sd(im, ["axis_value"], "mass")
            ax.errorbar([pos[v] for v in q["axis_value"]], q["m"], q["sd"], color=col, marker="o",
                        label=f"Shapley − stand-alone, {'fixed γ' if conv == 'gamma' else 'risk target'}",
                        capsize=3)
        ax.set_ylabel("Σₖ |φₖ − v({k})| / |total|")
        for ax in axs[:, c]:
            ax.set_xticks(range(len(labels)), labels)
            ax.set_xlabel(AXIS_LABEL[axis])
            ax.set_ylim(bottom=0)
    axs[0, 0].legend(fontsize=12, frameon=False)
    axs[1, 0].legend(fontsize=12, frameon=False)
    _stamp(fig, df.filter(pl.col("axis").is_in(axes_present + ["base"])))
    fig.tight_layout()
    return _save(fig, figdir, "S1-residual-vs-interaction")


def fig_iterated(df, figdir):
    d = df.filter(pl.col("config") == "switches")
    if d.is_empty():
        return []
    fig, axs = plt.subplots(1, 2, figsize=(22, 6.8), gridspec_kw={"width_ratios": [1, 1.6]})
    # (a) beta trajectories of the base variant
    b = d.filter((pl.col("method") == "iterated") & (pl.col("metric") == "beta_iter"))
    sigs = list(b["signal"].unique(maintain_order=True))
    ax = axs[0]
    for seed in sorted(b["seed"].unique())[:6]:
        for j, sname in enumerate(sigs):
            q = b.filter((pl.col("seed") == seed) & (pl.col("signal") == sname)).sort("t")
            ax.plot(q["t"] + 1, q["value"], color=SIGNAL_COLORS[j], lw=1.2, alpha=0.8,
                    label=sname if seed == b["seed"].min() else None)
    ax.axhline(1, color="0.5", lw=0.8)
    ax.set_xlabel("iteration")
    ax.set_title("Iterated stand-alone proxies (NNLS, α=1)")
    ax.set_ylabel("regression coefficient βₖ")
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=12, frameon=False)
    # (b) shares per switch variant, Durin as reference
    sh = shares(d)
    variants = ["durin"] + [v for v in ITERATED_VARIANTS if v in d["method"].unique() and v not in ("iterated:damped", "iterated:ridge", "iterated:ridge_damped")]
    q = _mean_sd(sh.filter(pl.col("method").is_in(variants)), ["method", "signal"], "share")
    conv = (d.filter(pl.col("metric") == "converged").group_by("method").agg(c=pl.col("value").mean()))
    nit = (d.filter(pl.col("metric") == "n_iter").group_by("method").agg(n=pl.col("value").median()))
    ax = axs[1]
    width = 0.8 / len(sigs)
    for j, sname in enumerate(sigs):
        m = [q.filter((pl.col("method") == v) & (pl.col("signal") == sname))["m"] for v in variants]
        e = [q.filter((pl.col("method") == v) & (pl.col("signal") == sname))["sd"] for v in variants]
        ax.bar(np.arange(len(variants)) + (j - len(sigs) / 2 + 0.5) * width,
               [x[0] if len(x) else np.nan for x in m], width,
               yerr=[x[0] if len(x) else 0 for x in e], color=SIGNAL_COLORS[j], capsize=2, label=sname)
    lab = []
    for v in variants:
        if v == "durin":
            lab.append("Durin")
            continue
        c = conv.filter(pl.col("method") == v)["c"]
        n = nit.filter(pl.col("method") == v)["n"]
        lab.append(f"{v.replace('iterated:', '').replace('iterated', 'NNLS α=1')}\n"
                   f"conv {c[0]:.0%}\n{n[0]:.0f} it")
    ax.set_xticks(range(len(variants)), lab, fontsize=12, rotation=30)
    ax.legend(fontsize=12, frameon=False, ncol=len(sigs))
    ax.set_ylabel("share of total P&L")
    ax.axhline(0, color="0.5", lw=0.8)
    _stamp(fig, d)
    fig.tight_layout()
    return _save(fig, figdir, "S1-iterated-switches")


def fig_stabilisers(df, figdir):
    d = df.filter(pl.col("method").is_in(["iterated", "iterated:damped", "iterated:ridge", "iterated:ridge_damped"]))
    failed = d.filter((pl.col("method") == "iterated") & (pl.col("metric") == "converged") & (pl.col("value") == 0))["config"].unique()
    cfgs = sorted(d.filter((pl.col("method") == "iterated:ridge") & pl.col("config").is_in(failed.implode()))["config"].unique())
    if not cfgs:
        return []
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))
    for method, label in [("iterated", "Iterated stand-alone proxies (NNLS, α=1)"), ("iterated:damped", "NNLS, α=0.5"),
                          ("iterated:ridge", "ridge to 1, α=1"), ("iterated:ridge_damped", "ridge to 1, α=0.5")]:
        for ax, metric in zip(axs, ["converged", "var_share"]):
            q = d.filter((pl.col("method") == method) & (pl.col("metric") == metric))
            vals = [q.filter(pl.col("config") == c)["value"].mean() for c in cfgs]
            ax.plot(range(len(cfgs)), vals, marker="o", label=label)
    for ax in axs:
        ax.set_xticks(range(len(cfgs)), cfgs, rotation=30)
        ax.legend(fontsize=12)
    axs[0].set_title("Optional remedies where undamped NNLS fails")
    axs[0].set_ylabel("fraction converged on live proxies")
    axs[1].set_ylabel("residual variance / total net variance")
    _stamp(fig, df, cfgs)
    fig.tight_layout()
    return _save(fig, figdir, "S1-stabilisers")


def fig_decoupled(df, figdir):
    d = df.filter(pl.col("config") == "base")
    if d.is_empty():
        return []
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))
    sh = shares(d)
    methods = ["durin", "regression", "iterated", "shapley_target", "shapley_gamma"]
    for j, suffix in enumerate(["", ":decoupled"]):
        vals = []
        for method in methods:
            q = sh.filter((pl.col("method") == method + suffix) & (pl.col("signal") != "_resid"))
            ref = sh.filter(pl.col("method") == "durin").select("unit", "seed", "signal", ref="share")
            a = q.join(ref, on=["unit", "seed", "signal"]).group_by("unit", "seed").agg(v=0.5*(pl.col("share")-pl.col("ref")).abs().sum())
            vals.append(a["v"].mean())
        axs[0].bar(np.arange(len(methods)) + (j-0.5)*0.35, vals, 0.35, label="net fit" if not suffix else "gross + separate costs")
    axs[0].set_xticks(range(len(methods)), ["Durin", "1 step", "Iterated stand-alone proxies (α=1)", "Shapley target", "Shapley γ"], rotation=20)
    axs[0].set_ylabel("distance to Durin net shares (base)")
    axs[0].legend(fontsize=12)
    q = df.filter(pl.col("metric") == "cost_gross_vol_ratio").group_by("config").agg(v=pl.col("value").mean()).sort("v")
    axs[1].bar(range(len(q)), q["v"])
    axs[1].set_xticks(range(len(q)), q["config"], rotation=90)
    axs[1].set_ylabel("cost volatility / gross P&L volatility")
    _stamp(fig, df)
    fig.tight_layout()
    return _save(fig, figdir, "S1-gross-cost")


def make_figures(results: Path, figdir: Path, tier: str) -> list[Path]:
    df = _load(Path(results))
    if df.is_empty():
        return []
    figdir = Path(figdir)
    out: list[Path] = []
    for f in (fig_b2, fig_band, fig_sweep_agreement, fig_sweep_shares, fig_known,
              fig_residual_interaction, fig_iterated, fig_stabilisers, fig_decoupled):
        out += f(df, figdir)
    return out
