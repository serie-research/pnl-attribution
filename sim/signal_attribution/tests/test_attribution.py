"""Properties that make the S1 figures true."""

from __future__ import annotations

import re

import numpy as np
import polars as pl
import pytest

from sim.signal_attribution import attribution as A
from sim.signal_attribution import optimiser as O
from sim.signal_attribution.desk import Desk, simulate
from sim.signal_attribution.units import STUDY, configs, run_unit, units

SMALL = Desk(n_assets=8, n_periods=60, box=1.0, spread_rel=0.8)


def _run(desk, seed=0):
    m = simulate(desk, np.random.default_rng(seed))
    sol = O.solve(desk, m.g.sum(0)[None], sigma_target=np.array([desk.risk_target]))
    return m, sol, O.pnl(desk, sol, m.r)[0]


@pytest.mark.parametrize("variant", [{}, {"long_only": True}, {"cov": "factor"}])
def test_durin_and_grinold_easton_sum_to_total(variant):
    desk = SMALL.with_(**variant)
    m, sol, total = _run(desk)
    assert sol.at_bound.any() and (sol.dx == 0).any()  # constraint and spread bind
    xk, dxk = A.durin_positions(desk, sol, m.g)
    np.testing.assert_allclose(xk.sum(0), sol.x[0], atol=1e-10)
    np.testing.assert_allclose(A.component_pnl(desk, xk, dxk, sol, m.r).sum(0), total, atol=1e-10)
    xg, dxg = A.grinold_easton_positions(desk, sol, m.g)
    np.testing.assert_allclose(A.component_pnl(desk, xg, dxg, sol, m.r).sum(0), total, atol=1e-9)


def test_durin_effective_multipliers_are_non_negative():
    """eta >= 0 (Durin p. 14): the box multiplier has the sign of the bound."""
    desk = SMALL
    _, sol, _ = _run(desk)
    x, nu, ab = sol.x[0], sol.nu[0], sol.at_bound[0]
    assert np.all(nu[ab] / x[ab] > 0)


def test_voting_rule():
    """Zero total trade gives zero signal trades; off the box, no trade iff
    |G - gamma sigma^2 x_{t-1}| <= lambda_0 (Durin Eq. 51, p. 17)."""
    desk = SMALL
    m, sol, _ = _run(desk)
    _, dxk = A.durin_positions(desk, sol, m.g)
    zero = sol.dx[0] == 0
    assert np.all(dxk[:, zero] == 0)
    xp = np.vstack([np.zeros(desk.n_assets), sol.x[0, :-1]])
    z = m.g.sum(0) - sol.gamma[0][:, None] * desk.sigma2() * xp
    free = ~sol.at_bound[0]
    np.testing.assert_array_equal(zero[free], (np.abs(z) <= desk.lam0())[free])


def test_risk_target_reached():
    desk = SMALL.with_(n_periods=200, box=np.inf)
    _, sol, _ = _run(desk)
    risk = np.sqrt((desk.sigma2() * sol.x[0] ** 2).sum(-1))
    reached = sol.gamma[0] > 1e-6 * desk.gamma_ref()  # early periods: target out of reach
    assert reached.mean() > 0.9
    np.testing.assert_allclose(risk[reached], desk.risk_target, rtol=1e-6)


def test_closed_form_matches_cvxpy_on_diagonal():
    desk = SMALL.with_(n_periods=30)
    m = simulate(desk, np.random.default_rng(3))
    G = m.g.sum(0)[None]
    sd = O.solve_diag(desk, G, sigma_target=np.array([1.0]))
    sc = O.solve_dense(desk, G, sigma_target=np.array([1.0]))
    np.testing.assert_allclose(sc.x, sd.x, atol=1e-5)
    bind = sd.gamma[0] > 1e-6 * desk.gamma_ref()
    np.testing.assert_allclose(sc.gamma[0, bind], sd.gamma[0, bind], rtol=1e-3)
    np.testing.assert_allclose(sc.nu, sd.nu, atol=1e-4)
    sg = O.solve_dense(desk, G, gamma_path=sd.gamma)
    np.testing.assert_allclose(sg.x, sd.x, atol=1e-5)


def test_shapley_efficiency_and_additive_game():
    """Sum of phi_k = v(N) = total. Without box and spread and with the full
    run's gamma path, positions are additive across coalitions; the P&L game is
    then additive up to the quadratic cost, whose pairwise cross terms Shapley
    splits equally, which is Durin's Euler split (Eq. 14): Shapley = Durin."""
    for desk, additive in [(SMALL, False), (SMALL.with_(box=np.inf, spread_rel=0.0), True)]:
        m, sol, total = _run(desk)
        K, T = m.g.shape[:2]
        GS = A.coalition_signals(m.g)
        cs = O.solve(desk, GS, gamma_path=np.broadcast_to(sol.gamma, (len(GS), T)))
        v = np.vstack([np.zeros(T), O.pnl(desk, cs, m.r)])
        np.testing.assert_allclose(v[-1], total, atol=1e-10)
        phi = A.shapley(v, K)
        np.testing.assert_allclose(phi.sum(0), total, atol=1e-10)
        if additive:
            xk, dxk = A.durin_positions(desk, sol, m.g)
            durin = A.component_pnl(desk, xk, dxk, sol, m.r)
            np.testing.assert_allclose(phi, durin, atol=1e-10)
            gross = (cs.x[A.singleton_index(K) - 1] * m.r).sum(-1)
            np.testing.assert_allclose(gross, (xk * m.r).sum(-1), atol=1e-10)


def test_identical_signals_get_identical_attribution():
    desk = configs("smoke")["identical"].desk.with_(n_assets=8, n_periods=60)
    m, sol, total = _run(desk)
    xk, dxk = A.durin_positions(desk, sol, m.g)
    durin = A.component_pnl(desk, xk, dxk, sol, m.r)
    np.testing.assert_allclose(durin[1], durin[2], atol=1e-12)
    GS = A.coalition_signals(m.g)
    v = np.vstack([np.zeros(desk.n_periods),
                   O.pnl(desk, O.solve(desk, GS, sigma_target=np.ones(len(GS))), m.r)])
    phi = A.shapley(v, 3)
    np.testing.assert_allclose(phi[1], phi[2], atol=1e-10)


@pytest.mark.parametrize("option", [{"init": "ful"}, {"negative": "flipp"}, {"adjust": "box"}])
def test_iterated_rejects_unknown_options(option):
    """A misspelt option must fail before any proxy is run, not select another variant."""
    desk = SMALL.with_(n_periods=20)

    def run_pnl(*_):
        raise AssertionError("no proxy should run")

    with pytest.raises(ValueError, match=next(iter(option))):
        A.iterated(desk, np.zeros((desk.n_signals, 20, desk.n_assets)), np.zeros(20), run_pnl, **option)


@pytest.mark.parametrize("frictions", [{"box": 2.0, "spread_rel": 0.3},
                                      {"box": np.inf, "spread_rel": 0.0}])
def test_iterated_fixed_point_and_singleton_link(frictions):
    """At convergence beta = 1 and the attribution is the proxies; the first
    iteration from sigma* is the stand-alone regression (method 1)."""
    desk = SMALL.with_(n_periods=200, **frictions)
    m, sol, total = _run(desk)

    def run_pnl(G, sigma, lo, hi):
        return O.pnl(desk, O.solve(desk, G, sigma_target=sigma, lo=lo, hi=hi), m.r)

    em = A.iterated(desk, m.g, total, run_pnl, max_iter=100, tol=1e-6)
    assert em.converged
    np.testing.assert_allclose(em.beta, 1.0, atol=1e-6)
    np.testing.assert_allclose(em.attributed.sum(0) + em.resid, total, atol=1e-10)
    K = desk.n_signals
    single = run_pnl(m.g, np.full(K, 1.0), *[np.broadcast_to(b, (K, desk.n_assets)) for b in desk.bounds()])
    beta1, _, _ = A.regress(total, single)
    np.testing.assert_allclose(em.beta_hist[0], beta1, rtol=1e-10)


def test_units_and_run_unit(tmp_path):
    ids = units("smoke")
    assert len(ids) == len(set(ids))
    assert all(re.fullmatch(r"[a-z0-9][a-z0-9_-]*", u) for u in ids)
    assert len(units("full")) > len(ids)
    out = tmp_path / "u.parquet"
    run_unit("smoke-k_2-b00", out)
    df = pl.read_parquet(out)
    assert set(df["config"]) == {"k_2"} and df["value"].is_finite().any()
    assert STUDY == "signal_attribution"


def test_nnls_matches_scipy_and_is_not_floored_ols():
    from scipy.optimize import nnls
    rng = np.random.default_rng(8)
    proxies = rng.normal(size=(4, 100))
    proxies[1] += proxies[0]
    total = proxies[0] - proxies[1] + 0.1 * rng.normal(size=100)
    beta, att, resid = A.regress(total, proxies)
    assert (beta >= 0).all()
    np.testing.assert_allclose(beta, nnls(proxies.T, total)[0])
    assert not np.allclose(beta, np.maximum(A.regress(total, proxies, estimator="ols")[0], 0))
    np.testing.assert_allclose(att.sum(0) + resid, total)


@pytest.mark.parametrize("alpha", [0.5, 1.0])
def test_additive_proxy_iteration(alpha):
    # A genuinely additive, homogeneous proxy map, no box/spread. Risk-targeted
    # optimiser paths need not be additive even when these two frictions vanish.
    desk = SMALL.with_(box=np.inf, spread_rel=0)
    rng = np.random.default_rng(9)
    proxies = rng.normal(size=(4, 60))
    weights = np.array([0.2, 0.4, 0.7, 0.3])
    def run(G, sigma, lo, hi):
        return G[:, :, 0] * sigma[:, None]
    g = np.repeat(proxies[:, :, None], 8, axis=2)
    em = A.iterated(desk, g, weights @ proxies, run, alpha=alpha, tol=1e-8)
    assert em.converged
    np.testing.assert_allclose(em.beta, 1, atol=1e-8)
    np.testing.assert_allclose(em.sigma, weights, atol=1e-8)
    np.testing.assert_allclose(em.resid, 0, atol=1e-8)


def test_ridge_toward_one_matches_augmented_nnls():
    from scipy.optimize import nnls
    rng = np.random.default_rng(7)
    p = rng.normal(size=(3, 60))
    y = rng.normal(size=60)
    mu = 0.7
    expected = nnls(np.vstack([p.T, np.sqrt(mu) * np.eye(3)]),
                    np.r_[y, np.full(3, np.sqrt(mu))])[0]
    np.testing.assert_allclose(A.regress(y, p, ridge=mu)[0], expected)


def test_decoupled_parts_sum_to_net():
    from sim.signal_attribution.experiment import run_seed
    rows = run_seed(SMALL, 12).rows
    d = {(m, s, k): v for m, s, k, t, v in rows if t == -1}
    methods = {m for m, s, k in d if m.endswith(":decoupled")}
    assert len(methods) == 6
    for meth in methods:
        signals = [s for m, s, k in d if m == meth and k == "gross_cum"]
        for signal in signals:
            assert d[meth, signal, "gross_cum"] + d[meth, signal, "cost_cum"] == pytest.approx(d[meth, signal, "cum"])
        assert sum(d[meth, s, "cum"] for s in signals) + d[meth, "_resid", "cum"] == pytest.approx(d["total", "_all", "cum"])
        if meth.startswith(("durin", "shapley", "grinold")):
            assert abs(d[meth, "_resid", "cum"]) < 1e-8


def test_all_zero_coefficients_stop_without_empty_solver_call():
    desk = SMALL
    g = np.ones((4, 60, 8))
    def run(G, sigma, lo, hi):
        assert len(G) > 0
        return G[:, :, 0]
    em = A.iterated(desk, g, -np.ones(60), run)
    assert em.status == "no_live_proxy"
    assert not em.converged
    np.testing.assert_array_equal(em.beta, 0)


@pytest.mark.parametrize("convention", ["target", "gamma"])
def test_decoupled_shapley_paths_and_cost_conservation(convention):
    desk = SMALL
    m, sol, total = _run(desk)
    K, T, N = m.g.shape
    GS = A.coalition_signals(m.g)
    kw = {"sigma_target": np.ones(len(GS))} if convention == "target" else {"gamma_path": np.broadcast_to(sol.gamma, (len(GS), T))}
    coal = O.solve(desk, GS, **kw)
    sx = A.shapley(np.concatenate([np.zeros((1, T, N)), coal.x]).reshape(2**K, -1), K).reshape(K, T, N)
    sd = np.diff(sx, axis=1, prepend=0)
    gross = A.shapley(np.vstack([np.zeros(T), (coal.x * m.r).sum(-1)]), K)
    cost = -0.5 * desk.lam() * (sd * desk.sigma2() * sol.dx[0]).sum(-1) - desk.lam0() * (sd * np.sign(sol.dx[0])).sum(-1)
    net = A.component_pnl(desk, sx, sd, sol, m.r)
    np.testing.assert_allclose(net, gross + cost, atol=1e-10)
    np.testing.assert_allclose(gross.sum(0), (sol.x[0] * m.r).sum(-1), atol=1e-10)
    np.testing.assert_allclose(net.sum(0), total, atol=1e-10)


def test_iteration_limit_returns_fitted_risk_targets():
    desk = SMALL
    rng = np.random.default_rng(1)
    g = rng.normal(size=(4, 60, 8))
    def run(G, sigma, lo, hi):
        return sigma[:, None] * G[:, :, 0]
    total = np.array([0.2, 0.4, 0.1, 0.6]) @ g[:, :, 0]
    em = A.iterated(desk, g, total, run, max_iter=1)
    np.testing.assert_array_equal(em.sigma, np.ones(4))
    np.testing.assert_allclose(em.attributed, em.beta[:, None] * em.sigma[:, None] * g[:, :, 0])


def test_constant_nonzero_proxy_is_live_without_intercept():
    desk = SMALL
    def run(G, sigma, lo, hi):
        return sigma[:, None] * np.ones((len(G), 60))
    em = A.iterated(desk, np.ones((1, 60, 8)), np.full(60, 0.4), run)
    assert em.converged
    np.testing.assert_allclose(em.beta, 1)
    np.testing.assert_allclose(em.sigma, 0.4)
    np.testing.assert_allclose(em.resid, 0, atol=1e-12)


def test_smaller_stress_blocks_preserve_all_s1_seed_identities():
    from sim.signal_attribution.units import seed_identity
    for name in ['corr_0p6', 'corr_0p9', 'k_5', 'k_6', 'spread_1', 'spread_2', 'switches']:
        cfg = configs('full')[name]
        assert cfg.seeds_per_block == 2 and cfg.n_blocks == 10
        got = [seed_identity('full', name, b, i) for b in range(10) for i in range(2)]
        expected = [(f'full-{name}-b{b:02d}', i) for b in range(2) for i in range(10)]
        assert got == expected
    assert seed_identity('full', 'factor', 3, 4) == ('full-factor-b03', 4)
