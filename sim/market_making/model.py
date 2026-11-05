"""Event-time GLFT quotes and exact benchmark-dependent accounting."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Parameters:
    # GLFT (2013), preprint p. 8 Fig. 1; Q from the plotted inventory range.
    sigma: float = 0.3
    a: float = 0.9
    k: float = 0.3
    gamma: float = 0.01
    horizon: float = 600.0
    bound: int = 30
    impact: float = 1.0  # Synthetic choice, ticks per fill.


def quotes(q: int, p: Parameters) -> tuple[float, float]:
    """Closed-form long-horizon approximation, bid and ask distances in ticks."""
    # GLFT p. 11, with adverse-selection extension pp. 14-15, 19-20.
    c = np.sqrt(
        p.sigma**2
        * p.gamma
        / (2 * p.k * p.a)
        * (1 + p.gamma / p.k) ** (1 + p.k / p.gamma)
    ) * np.exp(p.k * p.impact / 4)
    base = np.log1p(p.gamma / p.k) / p.gamma + p.impact / 2
    return base + (q + 0.5) * c, base - (q - 0.5) * c


def ledger(side, price, reference, terminal_mid):
    """Return spread, positioning, total; positive side means an MM purchase."""
    # Menkveld eqs. 1-3, PDF pp. 16-18, extended to terminal marked inventory.
    # sum d_i (r_i-p_i) + sum d_i (S_T-r_i) = -sum d_i p_i + q_T S_T.
    spread = np.dot(side, reference - price)
    positioning = np.dot(side, terminal_mid - reference)
    total = -np.dot(side, price) + side.sum() * terminal_mid
    return np.array([spread, positioning, total], dtype=float)


def simulate(rng, p=Parameters(), horizons=(0.0, 1.0, 5.0, 30.0)):
    """Simulate exact exponential waiting times and Brownian marks at query times."""
    t, q = 0.0, 0
    times, sides, distances = [], [], []
    while True:
        bid, ask = quotes(q, p)
        # GLFT p. 4: conditional Poisson intensities A exp(-k delta).
        rates = np.array(
            [
                p.a * np.exp(-p.k * bid) if q < p.bound else 0,
                p.a * np.exp(-p.k * ask) if q > -p.bound else 0,
            ]
        )
        t += rng.exponential(1 / rates.sum())
        if t > p.horizon:
            break
        buy = rng.random() < rates[0] / rates.sum()
        side = 1 if buy else -1
        times.append(t)
        sides.append(side)
        distances.append(bid if buy else ask)
        q += side
    times, side, distance = map(np.asarray, (times, sides, distances))
    queries = [np.minimum(times + h, p.horizon) for h in horizons]
    grid = np.unique(np.concatenate(([0, p.horizon], times, *queries)))
    brownian = np.cumsum(rng.normal(size=len(grid)) * np.sqrt(np.diff(grid, prepend=0)))
    event_index = np.searchsorted(grid, times)
    jumps = np.zeros(len(grid))
    # GLFT p. 14: dS = sigma dW - xi dN_bid + xi dN_ask.
    jumps[event_index] = -p.impact * side
    mids = 100 + p.sigma * brownian + np.cumsum(jumps)
    pre_mid = mids[event_index] + p.impact * side
    prices = pre_mid - side * distance
    values = []
    for h, query in zip(horizons, queries, strict=True):
        ref = pre_mid if h == 0 else mids[np.searchsorted(grid, query)]
        values.append(ledger(side, prices, ref, mids[-1]))
    # Independent chronological inventory revaluation, including each fill's jump.
    q_after = np.cumsum(side)
    inventory = 0.0
    if len(times):
        pre_jump_q = np.concatenate(([0], q_after[:-1]))
        between = pre_mid - np.concatenate(([100], mids[event_index[:-1]]))
        inventory = np.dot(pre_jump_q, between) - p.impact * np.dot(q_after, side)
        inventory += q_after[-1] * (mids[-1] - mids[event_index[-1]])
    return np.array(values), dict(
        fills=len(times),
        inventory_integral=float(inventory),
        max_inventory=int(np.max(np.abs(q_after), initial=0)),
    )
