"""The portfolio construction algorithm A and its KKT multipliers.

Static trading problem of Durin (2014) Eq. 5, p. 4, with the static-model
matrices of Eq. 7, p. 5 (Q = Lambda + gamma Sigma, P = gamma Sigma,
Lambda = lam Sigma), an L1 spread cost lambda_0 |dx| (Eq. 19, p. 9) and
position bounds lo <= x_{t-1} + dx <= hi (Eq. 15, p. 7):

    max_dx  dx.G - 1/2 dx.Q dx - dx.P x_{t-1} - lambda_0 |dx|_1.

Risk targeting: gamma_t is chosen each period so that the ex-ante risk
sqrt(x_t' Sigma x_t) equals sigma* (Durin p. 8: a risk constraint makes gamma
a time-dependent multiplier and leaves the optimality equation unchanged).

KKT (Eq. 17 and Eq. 21, p. 7-9), per asset in the diagonal case:

    q dx + p x_{t-1} = G - lambda_0 s - nu,

with s in the subdifferential of |dx| and nu the box multiplier (nu > 0 at an
upper bound, nu < 0 at a lower bound). In the diagonal case the problem is
separable and the solution is a soft threshold followed by a clip (one-dimensional
concave problem on an interval).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .desk import Desk


@dataclass
class Solution:
    """Path of the algorithm for a batch of B runs."""

    x: np.ndarray  # (B, T, N) positions after the trade at t
    dx: np.ndarray  # (B, T, N) trades
    gamma: np.ndarray  # (B, T) risk aversion used at t
    s: np.ndarray  # (B, T, N) spread subgradient (sign of dx where dx != 0)
    nu: np.ndarray  # (B, T, N) box multiplier
    at_bound: np.ndarray  # (B, T, N) bool, the box constraint binds


def _bcast_bounds(lo, hi, B, N):
    lo = np.broadcast_to(np.asarray(lo, float), (B, N))
    hi = np.broadcast_to(np.asarray(hi, float), (B, N))
    return lo, hi


def _trade_diag(G, xp, gamma, sig2, lam, lam0, lo, hi):
    """Closed-form optimal position for given gamma (B,) (Durin Eqs. 50-51, p. 17)."""
    g = gamma[:, None]
    q = (lam + g) * sig2
    z = G - g * sig2 * xp
    dstar = np.sign(z) * np.maximum(np.abs(z) - lam0, 0.0) / q  # soft threshold
    xu = xp + dstar
    return np.clip(xu, lo, hi), xu, z, q


def solve_diag(
    desk: Desk,
    G: np.ndarray,
    sigma_target: np.ndarray | None = None,
    gamma_path: np.ndarray | None = None,
    lo: np.ndarray | None = None,
    hi: np.ndarray | None = None,
    n_bisect: int = 32,
) -> Solution:
    """Run A on a batch of combined signals G (B, T, N), diagonal Sigma.

    Exactly one of ``sigma_target`` (B,) (risk target per run) and
    ``gamma_path`` (B, T) (fixed risk aversion per period) is given.
    """
    B, T, N = G.shape
    sig2 = desk.sigma2()
    lam, lam0 = desk.lam(), desk.lam0()
    if lo is None or hi is None:
        lo, hi = desk.bounds()
    lo, hi = _bcast_bounds(lo, hi, B, N)
    x = np.zeros((B, T, N))
    gam = np.zeros((B, T))
    xp = np.zeros((B, N))
    g_ref = desk.gamma_ref()
    g_min, g_max = 1e-8 * g_ref, 1e8 * g_ref
    g_prev = np.full(B, g_ref)
    if sigma_target is not None:
        tgt2 = np.broadcast_to(np.asarray(sigma_target, float), (B,)) ** 2

    def risk2(gm):
        xx = _trade_diag(G[:, t], xp, gm, sig2, lam, lam0, lo, hi)[0]
        return (sig2 * xx**2).sum(1)

    for t in range(T):
        if gamma_path is not None:
            gm = gamma_path[:, t]
        else:
            # bracket around last period's gamma, widen where needed;
            # risk is non-increasing in gamma
            a = np.maximum(g_prev / 30, g_min)
            b = np.minimum(g_prev * 30, g_max)
            for _ in range(10):
                low_a = risk2(a) < tgt2  # even at small gamma below target
                a = np.where(low_a & (a > g_min), np.maximum(a / 1e3, g_min), a)
                high_b = risk2(b) > tgt2
                b = np.where(high_b & (b < g_max), np.minimum(b * 1e3, g_max), b)
                if not ((low_a & (a > g_min)).any() or (high_b & (b < g_max)).any()):
                    break
            la, lb = np.log(a), np.log(b)
            for _ in range(n_bisect):
                m = 0.5 * (la + lb)
                above = risk2(np.exp(m)) > tgt2
                la = np.where(above, m, la)
                lb = np.where(above, lb, m)
            gm = np.exp(0.5 * (la + lb))
            # target out of reach at the smallest gamma: stay at g_min
            gm = np.where(risk2(a) < tgt2, a, gm)
            g_prev = gm
        x[:, t] = _trade_diag(G[:, t], xp, gm, sig2, lam, lam0, lo, hi)[0]
        gam[:, t] = gm
        xp = x[:, t]
    return _multipliers_diag(desk, G, x, gam, lo, hi)


def _multipliers_diag(desk, G, x, gam, lo, hi) -> Solution:
    """Recover s and nu from the KKT equation given positions and gamma."""
    sig2 = desk.sigma2()
    lam, lam0 = desk.lam(), desk.lam0()
    xp = np.concatenate([np.zeros_like(x[:, :1]), x[:, :-1]], axis=1)
    dx = x - xp
    g = gam[:, :, None]
    q = (lam + g) * sig2
    z = G - g * sig2 * xp
    dstar = np.sign(z) * np.maximum(np.abs(z) - lam0, 0.0) / q
    xu = xp + dstar
    at_bound = (xu > hi[:, None]) | (xu < lo[:, None])
    if lam0 > 0:
        s = np.where(dx != 0, np.sign(dx), np.clip(z / lam0, -1.0, 1.0))
    else:
        s = np.zeros_like(dx)
    nu = np.where(at_bound, z - q * dx - lam0 * s, 0.0)
    return Solution(x=x, dx=dx, gamma=gam, s=s, nu=nu, at_bound=at_bound)


def pnl(desk: Desk, sol: Solution, r: np.ndarray) -> np.ndarray:
    """Net P&L per period (B, T): x_t . r_{t+1} minus quadratic and spread costs."""
    return pnl_split(desk, sol.x, sol.dx, sol.dx, r)


def pnl_split(desk, xk, dxk, dx_total, r):
    """P&L of component positions xk with costs split as Durin Eqs. 14 and 22.

    Quadratic cost: 1/2 dx^k Lambda dx (Eq. 14, p. 7); spread cost:
    lambda_0 dx^k sgn(dx) (Eq. 22, p. 9). Components sum to the total.
    """
    if desk.cov != "diag":
        return pnl_split_dense(desk, xk, dxk, dx_total, r)
    lam, lam0 = desk.lam(), desk.lam0()
    sig2 = desk.sigma2()
    gross = (xk * r).sum(-1)
    quad = 0.5 * lam * (dxk * sig2 * dx_total).sum(-1)
    spread = lam0 * (dxk * np.sign(dx_total)).sum(-1)
    return gross - quad - spread


def pnl_split_dense(desk, xk, dxk, dx_total, r):
    """Same as ``pnl_split`` with Lambda = lam Sigma for a dense Sigma."""
    lam, lam0 = desk.lam(), desk.lam0()
    Sig = desk.covariance()
    gross = (xk * r).sum(-1)
    quad = 0.5 * lam * (dxk * (dx_total @ Sig)).sum(-1)
    spread = lam0 * (dxk * np.sign(dx_total)).sum(-1)
    return gross - quad - spread


# ---------------------------------------------------------------------------
# Dense covariance (one-factor variant): cvxpy, one QP/SOCP per period.


class DenseSolver:
    """Per-period problem with a dense Sigma = f f' + diag(s2), solved by Clarabel.

    Two forms share the cost model. Risk mode adds the constraint
    x' Sigma x <= sigma*^2, whose multiplier mu gives gamma = 2 mu; gamma mode
    uses a given gamma in the objective. Returns duals of the box constraints.
    """

    BIG = 1e6
    OPTS = dict(tol_gap_abs=1e-10, tol_gap_rel=1e-10, tol_feas=1e-10, tol_ktratio=1e-8)

    def __init__(self, desk: Desk):
        import cvxpy as cp

        self.cp = cp
        self.desk = desk
        self.n_fallback = 0
        N = desk.n_assets
        if desk.cov == "diag":
            f, s2 = np.zeros(N), desk.sigma2()
        else:
            f, s2 = desk.factor_model()
        self.f, self.s2 = f, s2
        self.Sigma = np.outer(f, f) + np.diag(s2)
        lam, lam0 = desk.lam(), desk.lam0()
        self.lin = cp.Parameter(N)
        self.qc = cp.Parameter(nonneg=True)
        self.xp = cp.Parameter(N)
        self.lo = cp.Parameter(N)
        self.hi = cp.Parameter(N)
        self.sig2 = cp.Parameter(nonneg=True)
        dx = cp.Variable(N)
        x = cp.Variable(N)
        sq = np.sqrt(s2)

        def quad(v):
            return cp.square(f @ v) + cp.sum_squares(cp.multiply(sq, v))

        obj = self.lin @ dx - 0.5 * self.qc * quad(dx) - lam0 * cp.norm1(dx)
        self.c_hi = x <= self.hi
        self.c_lo = x >= self.lo
        base = [x - dx == self.xp, self.c_hi, self.c_lo]
        self.c_risk = quad(x) <= self.sig2
        self.p_risk = cp.Problem(cp.Maximize(obj), base + [self.c_risk])
        self.p_gamma = cp.Problem(cp.Maximize(obj), base)
        self.dx, self.x = dx, x
        assert self.p_risk.is_dpp() and self.p_gamma.is_dpp()

    def _solve(self, prob):
        try:
            prob.solve(solver="CLARABEL", **self.OPTS)
        except self.cp.error.SolverError:
            # degenerate periods (previous position already optimal on the risk
            # sphere) can stall Clarabel's interior point; SCS is a fallback
            prob.solve(solver="SCS", eps_abs=1e-10, eps_rel=1e-10, max_iters=200_000)
            self.n_fallback += 1

    def step(self, G, xp, lo, hi, sigma=None, gamma=None):
        """Solve one period; returns x, dx, gamma, nu."""
        import warnings

        warnings.filterwarnings("ignore", message="Solution may be inaccurate")
        lam = self.desk.lam()
        self.xp.value = xp
        self.lo.value = np.maximum(lo, -self.BIG)
        self.hi.value = np.minimum(hi, self.BIG)
        if gamma is None:
            self.lin.value = G
            self.qc.value = lam
            self.sig2.value = sigma**2
            self._solve(self.p_risk)
            gm = 2.0 * float(np.asarray(self.c_risk.dual_value).ravel()[0])
        else:
            # 1/2 gamma x'Sigma x = 1/2 gamma dx'Sigma dx + gamma xp'Sigma dx + const
            self.lin.value = G - gamma * (self.Sigma @ xp)
            self.qc.value = lam + gamma
            self._solve(self.p_gamma)
            gm = gamma
        nu = self.c_hi.dual_value - self.c_lo.dual_value
        return self.x.value.copy(), self.dx.value.copy(), gm, nu


def solve_dense(
    desk: Desk,
    G: np.ndarray,
    sigma_target: np.ndarray | None = None,
    gamma_path: np.ndarray | None = None,
    lo: np.ndarray | None = None,
    hi: np.ndarray | None = None,
    solver: DenseSolver | None = None,
    tol: float = 1e-6,
) -> Solution:
    """Run A with cvxpy; same interface as ``solve_diag``."""
    B, T, N = G.shape
    solver = solver or DenseSolver(desk)
    if lo is None or hi is None:
        lo, hi = desk.bounds()
    lo, hi = _bcast_bounds(lo, hi, B, N)
    scale = desk.risk_target / np.sqrt(N)
    x = np.zeros((B, T, N))
    dx = np.zeros((B, T, N))
    nu = np.zeros((B, T, N))
    gam = np.zeros((B, T))
    for b in range(B):
        xp = np.zeros(N)
        for t in range(T):
            if gamma_path is not None:
                xt, d, gm, n = solver.step(G[b, t], xp, lo[b], hi[b], gamma=gamma_path[b, t])
            else:
                xt, d, gm, n = solver.step(G[b, t], xp, lo[b], hi[b], sigma=sigma_target[b])
            # snap solver noise: zero trades and positions on a bound
            d = np.where(np.abs(d) < tol * scale, 0.0, d)
            xt = xp + d
            on_hi = np.abs(xt - hi[b]) < tol * scale
            on_lo = np.abs(xt - lo[b]) < tol * scale
            xt = np.where(on_hi, hi[b], np.where(on_lo, lo[b], xt))
            d = xt - xp
            n = np.where(on_hi | on_lo, n, 0.0)
            x[b, t], dx[b, t], nu[b, t], gam[b, t] = xt, d, n, gm
            xp = xt
    at_bound = nu != 0
    # spread subgradient from the KKT residual where the trade is zero
    lam, lam0 = desk.lam(), desk.lam0()
    Sig = solver.Sigma
    # w = lambda_0 s + nu from the KKT equation; where the trade is zero the
    # split is not unique and we use the convention of the closed form: the
    # spread absorbs what it can, the box the rest
    # Off the box with a non-zero trade, s = w / lambda_0 equals sgn(dx) up to
    # solver error; taking it from w closes the KKT equation exactly, so the
    # linear recursions of Grinold-Easton and Durin do not accumulate that error.
    w = G - lam * dx @ Sig - gam[:, :, None] * (x @ Sig)
    if lam0 > 0:
        s = np.where(dx != 0, np.where(at_bound, np.sign(dx), w / lam0), np.clip(w / lam0, -1.0, 1.0))
    else:
        s = np.zeros_like(dx)
    nu = np.where(at_bound, w - lam0 * s, 0.0)
    at_bound = nu != 0
    return Solution(x=x, dx=dx, gamma=gam, s=s, nu=nu, at_bound=at_bound)


def solve(desk: Desk, G, **kw) -> Solution:
    """Dispatch to the closed form (diagonal) or to cvxpy (factor)."""
    if desk.cov == "diag":
        kw.pop("solver", None)
        return solve_diag(desk, G, **kw)
    return solve_dense(desk, G, **kw)
