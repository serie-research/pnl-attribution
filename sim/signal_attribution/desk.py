"""The simulated desk: signals, returns and the parameters of the optimiser.

Returns follow Durin (2014) Eqs. 1-4, p. 4: r_{t+1} = sum_k g^k_t + u_{t+1},
where g^k_t = b_k z^k_t is the return component predicted by signal k and z^k
is a unit-variance score. Scores are AR(1) per asset, the information-decay
model of Grinold (2011) Eqs. 5-8, p. 3: z_t = rho z_{t-1} + sqrt(1-rho^2) e_t.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np


@dataclass(frozen=True)
class Desk:
    """All parameters of one simulated desk (one configuration)."""

    n_assets: int = 50
    n_periods: int = 1000
    names: tuple[str, ...] = ("fast A", "slow A", "fast B", "slow B")
    # Per-signal coefficient b_k / sigma of the return model (multivariate IC).
    ic: tuple[float, ...] = (0.06, 0.05, 0.04, 0.03)
    # Coefficient the optimiser believes in; None means equal to ``ic``.
    ic_assumed: tuple[float, ...] | None = None
    # AR(1) persistence per signal: 0.5 is a one-period half-life (fast),
    # 0.95 a 13.5-period half-life (slow), Grinold (2011) Eq. 8, p. 3.
    rho: tuple[float, ...] = (0.5, 0.95, 0.5, 0.95)
    # Correlation of the AR(1) innovations between signals (scalar, all pairs),
    # or a full K x K matrix given as a tuple of tuples.
    corr: float | tuple[tuple[float, ...], ...] = 0.3
    # Index groups of signals that share the same innovations (identical signals).
    clones: tuple[tuple[int, int], ...] = ()
    cov: str = "diag"  # "diag" or "factor"
    factor_share: float = 0.3  # variance share of the factor in the factor variant
    risk_target: float = 1.0  # sigma*, ex-ante portfolio risk per period
    # Position bound, in units of the equal-risk position sigma*/(sigma sqrt N).
    box: float = 1.5
    long_only: bool = False
    # Quadratic cost Lambda = lam Sigma (Durin Eq. 7, p. 5), lam = lam_rel * gamma_ref.
    lam_rel: float = 1.0
    # Spread cost lambda_0 (Durin Eq. 19, p. 9) in units of std of G per asset.
    spread_rel: float = 0.5

    @property
    def n_signals(self) -> int:
        return len(self.ic)

    def with_(self, **kw) -> "Desk":
        return replace(self, **kw)

    def b_true(self) -> np.ndarray:
        return np.asarray(self.ic, float)

    def b_assumed(self) -> np.ndarray:
        return np.asarray(self.ic if self.ic_assumed is None else self.ic_assumed, float)

    def innovation_corr(self) -> np.ndarray:
        K = self.n_signals
        if isinstance(self.corr, (int, float)):
            C = np.full((K, K), float(self.corr))
            np.fill_diagonal(C, 1.0)
        else:
            C = np.asarray(self.corr, float)
        return C

    def stationary_corr(self) -> np.ndarray:
        """Correlation of the scores implied by AR(1) with correlated innovations."""
        rho = np.asarray(self.rho, float)
        C = self.innovation_corr()
        a = np.sqrt(1 - rho**2)
        S = C * np.outer(a, a) / (1 - np.outer(rho, rho))
        for i, j in self.clones:
            S[i, :] = S[j, :]
            S[:, i] = S[:, j]
        return S

    def asset_vol(self) -> np.ndarray:
        return np.ones(self.n_assets)

    def sigma2(self) -> np.ndarray:
        """Diagonal of Sigma (total variance per asset)."""
        return self.asset_vol() ** 2

    def factor_model(self):
        """One-factor covariance Sigma = f f^T + diag(s^2) with unit total variance."""
        N = self.n_assets
        beta = np.linspace(0.8, 1.2, N)
        f = np.sqrt(self.factor_share) * beta
        s2 = 1.0 - f**2
        return f, s2

    def covariance(self) -> np.ndarray:
        if self.cov == "diag":
            return np.diag(self.sigma2())
        f, s2 = self.factor_model()
        return np.outer(f, f) + np.diag(s2)

    def g_std(self) -> float:
        """Stationary standard deviation of G = sum_k g^k per asset (assumed model)."""
        b = self.b_assumed()
        return float(np.sqrt(b @ self.stationary_corr() @ b))

    def gamma_ref(self) -> float:
        """Risk aversion that hits sigma* with no costs and no bounds (diagonal).

        x = G / (gamma sigma^2) (Markowitz aim, Durin p. 13), so
        ||sigma x|| = sqrt(N) std(G) / gamma = sigma*.
        """
        return np.sqrt(self.n_assets) * self.g_std() / self.risk_target

    def lam(self) -> float:
        return self.lam_rel * self.gamma_ref()

    def lam0(self) -> float:
        return self.spread_rel * self.g_std()

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        M = self.box * self.risk_target / (self.asset_vol() * np.sqrt(self.n_assets))
        hi = np.full(self.n_assets, np.inf) if np.isinf(self.box) else M
        lo = np.zeros(self.n_assets) if self.long_only else -hi
        return lo, hi


@dataclass
class Market:
    """One simulated path: signal components and realised returns."""

    g: np.ndarray  # (K, T, N) signal components fed to the optimiser, return units
    r: np.ndarray  # (T, N) return from t to t+1 (realised after x_t is chosen)
    z: np.ndarray = field(repr=False, default=None)  # (K, T, N) unit scores


def simulate(desk: Desk, rng: np.random.Generator) -> Market:
    """Simulate scores, signal components and returns for one path."""
    K, T, N = desk.n_signals, desk.n_periods, desk.n_assets
    rho = np.asarray(desk.rho, float)[:, None]
    L = np.linalg.cholesky(desk.innovation_corr() + 1e-12 * np.eye(K))
    e = np.einsum("kl,ltn->ktn", L, rng.standard_normal((K, T, N)))
    for i, j in desk.clones:
        e[i] = e[j]
    z = np.empty((K, T, N))
    # start in the stationary distribution (marginally N(0,1); cross-corr of the
    # innovations, a small transient for mixed persistences)
    z[:, 0] = e[:, 0]
    a = np.sqrt(1 - rho**2)
    for t in range(1, T):
        z[:, t] = rho * z[:, t - 1] + a * e[:, t]  # Grinold (2011) Eq. 5, p. 3
    for i, j in desk.clones:
        z[i] = z[j]
    vol = desk.asset_vol()
    if desk.cov == "diag":
        u = rng.standard_normal((T, N)) * vol
    else:
        f, s2 = desk.factor_model()
        u = rng.standard_normal((T, 1)) * f + rng.standard_normal((T, N)) * np.sqrt(s2)
    b_true = desk.b_true()[:, None, None]
    b_ass = desk.b_assumed()[:, None, None]
    r = (b_true * z * vol).sum(0) + u  # Durin Eq. 4, p. 4
    g = b_ass * z * vol
    return Market(g=g, r=r, z=z)
