"""The five attribution methods compared in S1.

Each returns a P&L path per signal (K, T) and, where it applies, a residual (T,).

1. ``regression``: total P&L regressed on stand-alone P&Ls at the same risk
   target (Durin 2014 p. 2-3 describes it and expects a large residual when
   constraints bind).
2. ``grinold_easton``: signal portfolios plus constraint and spread portfolios
   built from the multipliers (Durin Eq. 17 p. 7 and Eq. 21 p. 9).
3. ``durin``: constraints and spread folded into effective quadratic terms
   (Durin Eqs. 30-31 p. 12, Eq. 37 p. 14, Eq. 48 p. 17); exact, no residual.
4. ``iterated``: the iterated stand-alone proxy regression
   (article section 3.1, "Stand-alone portfolios as proxies").
5. ``shapley``: exact Shapley values of the game v(S) = P&L of A run on the
   signals of S (Shapley 1953; Lipovetsky & Conklin 2001 for the regression game).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import factorial
from typing import Callable

import numpy as np
from scipy.optimize import nnls

from .desk import Desk
from .optimiser import Solution, pnl_split


# ---------------------------------------------------------------------------
# Method 3: Durin's signal-wise split


def durin_positions(desk: Desk, sol: Solution, g: np.ndarray, b: int = 0):
    """Signal-wise positions x^k and trades dx^k of run ``b`` of ``sol``.

    Effective quadratic terms per asset (diagonal or dense Sigma):
    spread, 2 eta_s = lambda_0 / |dx| (Eq. 48, p. 17), added to Q only;
    box, 2 eta_b = nu / x (Eq. 37, p. 14, nu the solver's multiplier), added to
    Q and P (Eq. 31, p. 12). Then Qbar dx^k + Pbar x^k_{t-1} = g^k (Eq. 24).
    Infinite multipliers are taken as limits: a zero total trade gives zero
    signal trades (Eq. 54 and the voting reading p. 18); a position held at a
    zero bound gives zero signal positions (Eq. 44 with alpha -> 1, p. 16).
    """
    if desk.cov != "diag":
        return _durin_dense(desk, sol, g, b)
    K, T, N = g.shape
    sig2 = desk.sigma2()
    lam, lam0 = desk.lam(), desk.lam0()
    xk = np.zeros((K, T, N))
    xp = np.zeros((K, N))
    for t in range(T):
        x, dx, nu, ab, gm = sol.x[b, t], sol.dx[b, t], sol.nu[b, t], sol.at_bound[b, t], sol.gamma[b, t]
        zero_trade = dx == 0
        zero_pos = ab & (x == 0) & ~zero_trade
        free = ~zero_trade & ~zero_pos
        s = sol.s[b, t]
        with np.errstate(divide="ignore", invalid="ignore"):
            Ds = np.where(free, lam0 * s / dx, 0.0)  # 2 eta_s = lambda_0 / |dx|
            Db = np.where(free & ab, nu / x, 0.0)  # 2 eta_b
        Qbar = (lam + gm) * sig2 + Ds + Db  # Eq. 31
        Pbar = gm * sig2 + Db
        dk = (g[:, t] - Pbar * xp) / Qbar  # Eq. 11 with Qbar, Pbar
        dk = np.where(zero_trade, 0.0, np.where(zero_pos, -xp, dk))
        xk[:, t] = xp + dk
        xp = xk[:, t]
    dxk = np.diff(xk, axis=1, prepend=0.0)
    return xk, dxk


def _durin_dense(desk, sol, g, b):
    K, T, N = g.shape
    Sig = desk.covariance()
    lam, lam0 = desk.lam(), desk.lam0()
    xk = np.zeros((K, T, N))
    xp = np.zeros((K, N))
    for t in range(T):
        x, dx, nu, ab, gm = sol.x[b, t], sol.dx[b, t], sol.nu[b, t], sol.at_bound[b, t], sol.gamma[b, t]
        zero_trade = dx == 0
        zero_pos = ab & (x == 0) & ~zero_trade
        free = ~zero_trade & ~zero_pos
        f = np.flatnonzero(free)
        Ds = np.zeros(N)
        Db = np.zeros(N)
        Ds[f] = lam0 * sol.s[b, t, f] / dx[f]  # lambda_0 / |dx| (Eq. 48)
        fb = np.flatnonzero(free & ab)
        Db[fb] = nu[fb] / x[fb]
        Qbar = (lam + gm) * Sig + np.diag(Ds + Db)
        Pbar = gm * Sig + np.diag(Db)
        dk = np.where(zero_pos, -xp, 0.0)  # fixed components (K, N)
        rhs = g[:, t, f] - xp @ Pbar[f].T - dk @ Qbar[f].T
        dk[:, f] = np.linalg.solve(Qbar[np.ix_(f, f)], rhs.T).T
        xk[:, t] = xp + dk
        xp = xk[:, t]
    dxk = np.diff(xk, axis=1, prepend=0.0)
    return xk, dxk


# ---------------------------------------------------------------------------
# Method 2: Grinold-Easton split with separate constraint and spread portfolios


def grinold_easton_positions(desk: Desk, sol: Solution, g: np.ndarray, b: int = 0):
    """Positions of the K signal portfolios, the spread and the box portfolio.

    Sources of Q dx + P x_{t-1} = sum_k g^k - lambda_0 s - nu (Eqs. 17, 21),
    each run through the same linear recursion (Eqs. 10-12, p. 6).
    Returns (K + 2, T, N): signals, then spread, then constraint.
    """
    K, T, N = g.shape
    lam, lam0 = desk.lam(), desk.lam0()
    src = np.concatenate([g, -lam0 * sol.s[b][None], -sol.nu[b][None]], axis=0)
    C = K + 2
    xk = np.zeros((C, T, N))
    xp = np.zeros((C, N))
    dense = desk.cov != "diag"
    if dense:
        Sig = desk.covariance()
    else:
        sig2 = desk.sigma2()
    for t in range(T):
        gm = sol.gamma[b, t]
        if dense:
            Q = (lam + gm) * Sig
            dk = np.linalg.solve(Q, (src[:, t] - gm * xp @ Sig).T).T
        else:
            dk = (src[:, t] - gm * sig2 * xp) / ((lam + gm) * sig2)
        xk[:, t] = xp + dk
        xp = xk[:, t]
    dxk = np.diff(xk, axis=1, prepend=0.0)
    return xk, dxk


def component_pnl(desk, xk, dxk, sol: Solution, r, b: int = 0):
    """P&L per component with Durin's cost split (Eqs. 14, 22)."""
    return pnl_split(desk, xk, dxk, sol.dx[b][None], r[None])


def risk_shares(desk, xk, sol: Solution, b: int = 0):
    """Litterman (1996) / Durin Eq. 13: R_k = x^k' Sigma x, averaged over time."""
    Sig = desk.covariance()
    x = sol.x[b]
    Rk = np.einsum("ktn,nm,tm->k", xk, Sig, x)
    return Rk / Rk.sum()


# ---------------------------------------------------------------------------
# Methods 1 and 4: regressions on stand-alone proxies


def regress(total: np.ndarray, proxies: np.ndarray, intercept: bool = False,
            estimator: str = "nnls", ridge: float = 0.0):
    """No-intercept NNLS by default; ridge is mu * sum((beta - 1)**2).

    Ridge changes the estimator. An optional unpenalised intercept is removed
    by centring; it remains in the reported residual. OLS preserves S1.
    """
    X = proxies.T
    y = total
    if intercept:
        X = X - X.mean(0)
        y = y - y.mean()
    if ridge < 0 or estimator not in ("nnls", "ols"):
        raise ValueError("invalid regression option")
    if ridge:
        X = np.vstack([X, np.sqrt(ridge) * np.eye(proxies.shape[0])])
        y = np.r_[y, np.full(proxies.shape[0], np.sqrt(ridge))]
    beta = nnls(X, y)[0] if estimator == "nnls" else np.linalg.lstsq(X, y, rcond=None)[0]
    attributed = beta[:, None] * proxies
    return beta, attributed, total - attributed.sum(0)


RunFn = Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray]


@dataclass
class IteratedResult:
    beta: np.ndarray
    sigma: np.ndarray
    sign: np.ndarray
    attributed: np.ndarray  # (K, T)
    resid: np.ndarray  # (T,)
    n_iter: int
    converged: bool
    status: str
    beta_hist: list = field(default_factory=list)
    sigma_hist: list = field(default_factory=list)


def iterated(
    desk: Desk,
    g: np.ndarray,
    total: np.ndarray,
    run_pnl: RunFn,
    *,
    intercept: bool = False,
    estimator: str = "nnls",
    alpha: float = 1.0,
    ridge: float = 0.0,
    init: str = "full",
    negative: str = "flip",
    adjust: str = "risk",
    tol: float = 1e-3,
    max_iter: int = 50,
) -> IteratedResult:
    """Iterative stand-alone proxy attribution.

    Proxy k: run A on sign_k g^k alone at risk target sigma_k (other parameters
    as in the full run, or the box scaled by sigma_k / sigma* if
    ``adjust == "risk_box"``). Regress the total on the proxies; set
    sigma_k <- beta_k**alpha sigma_k; repeat until max |beta_k - 1| < tol.
    Default: NNLS, alpha=1, ridge=0 (Iterated). Ridge changes the estimator.
    ``init``: sigma_k = sigma* ("full") or sigma*/sqrt(K) ("sqrtK").
    ``negative``: a beta_k < 0 flips the proxy ("flip"), drops it ("floor"),
    or ends the iteration ("stop").
    """
    for name, value, allowed in (("init", init, ("full", "sqrtK")),
                                 ("negative", negative, ("flip", "floor", "stop")),
                                 ("adjust", adjust, ("risk", "risk_box"))):
        if value not in allowed:
            raise ValueError(f"{name} must be one of {allowed}, got {value!r}")
    K, T, N = g.shape
    s_star = desk.risk_target
    sigma = np.full(K, s_star if init == "full" else s_star / np.sqrt(K))
    sign = np.ones(K)
    lo0, hi0 = desk.bounds()
    status = "max_iter"
    bh, sh = [], []
    beta = np.zeros(K)
    proxies = np.zeros((K, T))
    for it in range(1, max_iter + 1):
        act = np.flatnonzero(sigma > 0)
        G = sign[act, None, None] * g[act]
        if adjust == "risk_box":
            scale = (sigma[act] / s_star)[:, None]
            lo, hi = lo0[None] * scale, hi0[None] * scale
        else:
            lo = np.broadcast_to(lo0, (len(act), N))
            hi = np.broadcast_to(hi0, (len(act), N))
        proxies = np.zeros((K, T))
        if len(act):
            proxies[act] = run_pnl(G, sigma[act], lo, hi)
        # a proxy that never trades (e.g. spread above the signal) carries no
        # information on its scale: beta = 0, sigma kept, out of the criterion
        act = act[np.any(proxies[act] != 0, axis=1)]
        beta = np.zeros(K)
        if len(act):
            beta[act], _, _ = regress(total, proxies[act], intercept, estimator, ridge)
        bh.append(beta.copy())
        sh.append((sign * sigma).copy())
        if not len(act):
            status = "no_live_proxy"
            break
        if np.max(np.abs(beta[act] - 1.0)) < tol:
            status = "converged"
            break
        neg = act[beta[act] < 0]
        if len(neg) and negative == "stop":
            status = "stopped_negative"
            break
        if it == max_iter:
            break  # return the risk targets that actually generated the final proxies
        if negative == "flip":
            sign[neg] *= -1
            sigma[act] *= np.abs(beta[act]) ** alpha
        else:  # "floor", or "stop" with no negative beta (it has already returned otherwise)
            sigma[act] *= np.maximum(beta[act], 0.0) ** alpha
    attributed = beta[:, None] * proxies
    resid = total - attributed.sum(0)
    return IteratedResult(
        beta=beta, sigma=sigma, sign=sign, attributed=attributed, resid=resid,
        n_iter=it, converged=status == "converged", status=status,
        beta_hist=bh, sigma_hist=sh,
    )


# ---------------------------------------------------------------------------
# Method 5: Shapley over signal coalitions


def coalition_signals(g: np.ndarray) -> np.ndarray:
    """Combined signal of every coalition, indexed by bitmask 1..2^K-1."""
    K = g.shape[0]
    masks = np.arange(1, 2**K)
    member = (masks[:, None] >> np.arange(K)) & 1  # (2^K-1, K)
    return np.einsum("mk,ktn->mtn", member.astype(float), g)


def shapley_weights(K: int):
    return np.array([factorial(s) * factorial(K - s - 1) / factorial(K) for s in range(K)])


def shapley(v: np.ndarray, K: int) -> np.ndarray:
    """Exact Shapley values per period.

    ``v``: (2^K, T) coalition values indexed by bitmask, v[0] = 0.
    phi_k = sum_{S not containing k} |S|!(K-|S|-1)!/K! [v(S + k) - v(S)].
    """
    w = shapley_weights(K)
    masks = np.arange(2**K)
    size = np.array([bin(m).count("1") for m in masks])
    phi = np.zeros((K, v.shape[1]))
    for k in range(K):
        S = masks[(masks >> k) & 1 == 0]
        phi[k] = (w[size[S]][:, None] * (v[S | (1 << k)] - v[S])).sum(0)
    return phi


def singleton_index(K: int) -> np.ndarray:
    """Bitmask of each singleton coalition."""
    return 1 << np.arange(K)


__all__ = [
    "durin_positions", "grinold_easton_positions", "component_pnl", "risk_shares",
    "regress", "iterated", "IteratedResult", "coalition_signals", "shapley",
    "shapley_weights", "singleton_index",
]
