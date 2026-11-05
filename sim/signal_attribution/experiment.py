"""One seed of one configuration: run the desk and every attribution method."""

from __future__ import annotations

import numpy as np

from . import attribution as A
from . import optimiser as O
from .desk import Desk, simulate

# Iterated's switches (README, "Iterated's method: switches"); the first is the base.
ITERATED_BASE = dict(intercept=False, init="full", negative="flip", adjust="risk")
ITERATED_VARIANTS = {
    "iterated": {},
    "iterated:ols_flip": dict(estimator="ols"),
    "iterated:damped": dict(alpha=0.5),
    "iterated:ridge": {},
    "iterated:ridge_damped": dict(alpha=0.5),
    "iterated:intercept": dict(intercept=True, estimator="ols"),
    "iterated:init_sqrtK": dict(init="sqrtK", estimator="ols"),
    "iterated:neg_floor": dict(negative="floor", estimator="ols"),
    "iterated:neg_stop": dict(negative="stop", estimator="ols"),
    "iterated:adjust_box": dict(adjust="risk_box", estimator="ols"),
}


class Rows:
    """Accumulates tidy rows: method, signal, metric, t, value."""

    def __init__(self):
        self.rows: list[tuple[str, str, str, int, float]] = []

    def add(self, method, signal, metric, value, t=-1):
        self.rows.append((method, signal, metric, int(t), float(value)))

    def add_signals(self, method, names, metric, values):
        for n, v in zip(names, values):
            self.add(method, n, metric, v)


def _corr(a, b):
    sa, sb = a.std(), b.std()
    if sa == 0 or sb == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def run_seed(
    desk: Desk,
    seed: int,
    variants: tuple[str, ...] = ("iterated",),
    keep_paths: bool = False,
) -> Rows:
    """Simulate one path and attribute its P&L with every method."""
    rng = np.random.default_rng(seed)
    m = simulate(desk, rng)
    g, r = m.g, m.r
    K, T, N = g.shape
    names = list(desk.names[:K])
    out = Rows()
    solver = O.DenseSolver(desk) if desk.cov != "diag" else None

    def run(G, **kw):
        return O.solve(desk, G, solver=solver, **kw)

    def run_pnl(G, sigma, lo, hi):
        return O.pnl(desk, run(G, sigma_target=sigma, lo=lo, hi=hi), r)

    # the full run
    full = run(g.sum(0)[None], sigma_target=np.array([desk.risk_target]))
    total = O.pnl(desk, full, r)[0]
    gross = (full.x[0] * r).sum(-1)
    costs = total - gross  # signed cost contribution
    out.add("desk", "_all", "cost_gross_vol_ratio", costs.std() / gross.std())
    out.add("total", "_all", "cum", total.sum())
    out.add("total", "_all", "var", total.var())
    out.add("desk", "_all", "bound_frac", full.at_bound.mean())
    out.add("desk", "_all", "notrade_frac", (full.dx == 0).mean())
    # share of periods where sigma* is reached (gamma above its floor)
    out.add("desk", "_all", "target_reached_frac", (full.gamma[0] > 1e-6 * desk.gamma_ref()).mean())
    out.add("desk", "_all", "risk_mean", np.sqrt(np.einsum("tn,nm,tm->t", full.x[0], desk.covariance(), full.x[0])).mean())
    out.add("desk", "_all", "stationary_corr_mean", desk.stationary_corr()[np.triu_indices(K, 1)].mean() if K > 1 else 0.0)
    paths = {"total": {"_all": total}}

    def record_parts(name, labels, gp, cp):
        net = gp + cp
        residual = total - net.sum(0)
        out.add_signals(name, labels, "gross_cum", gp.sum(1))
        out.add_signals(name, labels, "cost_cum", cp.sum(1))
        out.add_signals(name, labels, "cum", net.sum(1))
        out.add(name, "_resid", "cum", residual.sum())
        out.add(name, "_resid", "var_share", residual.var() / total.var())
        paths[name] = dict(zip(labels, net)) | {"_resid": residual}

    # method 3: Durin
    xk, dxk = A.durin_positions(desk, full, g)
    durin = A.component_pnl(desk, xk, dxk, full, r)
    out.add_signals("durin", names, "cum", durin.sum(1))
    out.add_signals("durin", names, "risk_share", A.risk_shares(desk, xk, full))
    out.add("durin", "_resid", "max_abs", np.abs(total - durin.sum(0)).max())
    paths["durin"] = dict(zip(names, durin))

    dg = (xk * r).sum(-1)
    record_parts("durin:decoupled", names, dg, durin - dg)

    # method 2: Grinold-Easton
    xg, dxg = A.grinold_easton_positions(desk, full, g)
    ge = A.component_pnl(desk, xg, dxg, full, r)
    ge_names = names + ["spread", "constraint"]
    out.add_signals("grinold_easton", ge_names, "cum", ge.sum(1))
    out.add("grinold_easton", "_resid", "max_abs", np.abs(total - ge.sum(0)).max())
    paths["grinold_easton"] = dict(zip(ge_names, ge))

    gg = (xg * r).sum(-1)
    record_parts("grinold_easton:decoupled", ge_names, gg, ge - gg)

    # method 5: Shapley, two risk conventions for a coalition
    GS = A.coalition_signals(g)
    single = A.singleton_index(K)
    for conv in ("target", "gamma"):
        if conv == "target":
            sol = run(GS, sigma_target=np.full(len(GS), desk.risk_target))
        else:
            sol = run(GS, gamma_path=np.broadcast_to(full.gamma, (len(GS), T)))
        v = np.vstack([np.zeros(T), O.pnl(desk, sol, r)])
        phi = A.shapley(v, K)
        vg = np.vstack([np.zeros(T), (sol.x * r).sum(-1)])
        # Shapley of gross is linear in coalition positions. Apply Durin's
        # Eqs. 14 and 22 to these signal positions/trades against the full trade.
        sx = A.shapley(np.concatenate([np.zeros((1, T, N)), sol.x]).reshape(2**K, -1), K).reshape(K, T, N)
        sdx = np.diff(sx, axis=1, prepend=0.0)
        sg = A.shapley(vg, K)
        sc = A.component_pnl(desk, sx, sdx, full, r) - sg
        record_parts(f"shapley_{conv}:decoupled", names, sg, sc)
        out.add_signals(f"shapley_{conv}", names, "cum", phi.sum(1))
        out.add_signals(f"singleton_{conv}", names, "cum", v[single].sum(1))
        out.add(f"shapley_{conv}", "_resid", "max_abs", np.abs(v[-1] - phi.sum(0)).max())
        out.add(f"shapley_{conv}", "_all", "vN_minus_total", np.abs(v[-1] - total).max())
        # residual of the stand-alone sum = total interaction, sum_k (phi_k - v_k)
        st_resid = v[-1] - v[single].sum(0)
        out.add(f"singleton_{conv}", "_resid", "cum", st_resid.sum())
        out.add(f"singleton_{conv}", "_resid", "var_share", st_resid.var() / total.var())
        paths[f"shapley_{conv}"] = dict(zip(names, phi))
        paths[f"singleton_{conv}"] = dict(zip(names, v[single]))
        if conv == "target":
            # method 1: regression on stand-alone P&Ls at the same risk target
            beta, att, res = A.regress(total, v[single])
            out.add_signals("regression", names, "cum", att.sum(1))
            out.add_signals("regression", names, "beta", beta)
            out.add("regression", "_resid", "cum", res.sum())
            out.add("regression", "_resid", "var_share", res.var() / total.var())
            bg, ag, _ = A.regress(gross, vg[single])
            record_parts("regression:decoupled", names, ag, bg[:, None] * (v[single] - vg[single]))
            paths["regression"] = dict(zip(names, att)) | {"_resid": res}

    # method 4: Iterated's iteration, with the requested switch variants
    for name in variants:
        opts = ITERATED_BASE | ITERATED_VARIANTS[name]
        if "ridge" in name:
            # Fixed across iterations: 1% of full gross squared norm per signal.
            opts["ridge"] = 0.01 * np.dot(gross, gross) / K
        em = A.iterated(desk, g, total, run_pnl, **opts)
        out.add_signals(name, names, "cum", em.attributed.sum(1))
        out.add_signals(name, names, "beta", em.beta)
        out.add_signals(name, names, "sigma", em.sign * em.sigma)
        out.add(name, "_resid", "cum", em.resid.sum())
        out.add(name, "_resid", "var_share", em.resid.var() / total.var())
        out.add(name, "_all", "converged", float(em.converged))
        out.add(name, "_all", "n_iter", em.n_iter)
        out.add(name, "_all", "zero_beta_count", (em.beta == 0).sum())
        out.add(name, "_all", "no_live_proxy", em.status == "no_live_proxy")
        for it, (b, s) in enumerate(zip(em.beta_hist, em.sigma_hist)):
            for n, bk, sk in zip(names, b, s):
                out.add(name, n, "beta_iter", bk, t=it)
                out.add(name, n, "sigma_iter", sk, t=it)
        paths[name] = dict(zip(names, em.attributed)) | {"_resid": em.resid}

    last_cost = None

    def run_gross(G, sigma, lo, hi):
        nonlocal last_cost
        sol = run(G, sigma_target=sigma, lo=lo, hi=hi)
        gp = (sol.x * r).sum(-1)
        last_cost = O.pnl(desk, sol, r) - gp
        return gp

    eg = A.iterated(desk, g, gross, run_gross)
    cp = np.zeros_like(eg.attributed)
    # Final callback includes precisely the positive-risk proxies.
    if np.any(eg.sigma > 0):
        cp[eg.sigma > 0] = eg.beta[eg.sigma > 0, None] * last_cost
    record_parts("iterated:decoupled", names, eg.attributed, cp)
    out.add("iterated:decoupled", "_all", "converged", eg.converged)
    out.add("iterated:decoupled", "_all", "n_iter", eg.n_iter)

    # agreement of every method with Durin, per signal (per-period correlation)
    for meth, d in paths.items():
        if meth in ("total", "durin"):
            continue
        for n in names:
            if n in d:
                out.add(meth, n, "corr_durin", _corr(d[n], paths["durin"][n]))

    if keep_paths:
        for meth, d in paths.items():
            for n, p in d.items():
                c = np.cumsum(p)
                for t in range(T):
                    out.add(meth, n, "cum_path", c[t], t=t)
        if K == 2:
            _band_points(desk, full, g, out)
    return out


def _band_points(desk, full, g, out, n_assets=5):
    """Points for the no-trade band (Durin Eq. 51, p. 17), two signals.

    Coordinates a = g^1 / lambda_0 and b = (g^2 - gamma sigma^2 x_{t-1}) / lambda_0:
    no trade iff |a + b| <= 1 when the box does not bind.
    """
    lam0 = desk.lam0()
    if lam0 == 0:
        return
    sig2 = desk.sigma2()
    x = full.x[0]
    xp = np.vstack([np.zeros(x.shape[1]), x[:-1]])
    gm = full.gamma[0][:, None]
    a = g[0] / lam0
    b = (g[1] - gm * sig2 * xp) / lam0
    c = gm * sig2 * xp / lam0  # position term, to test each signal alone
    for i in range(n_assets):
        for t in range(x.shape[0]):
            if full.at_bound[0, t, i]:
                continue
            out.add("band", f"asset{i}", "a", a[t, i], t=t)
            out.add("band", f"asset{i}", "b", b[t, i], t=t)
            out.add("band", f"asset{i}", "c", c[t, i], t=t)
            out.add("band", f"asset{i}", "trade", float(full.dx[0, t, i] != 0), t=t)
