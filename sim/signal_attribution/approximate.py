"""Cached scalar coalition games and estimators of their allocations.

Masks use bit i for player i. Costs include the empty coalition; use a fresh
oracle for each comparison. Random draw counts are not backtest budgets.
See approximate_study.py and its result for settings and references.
"""
from dataclasses import dataclass
from itertools import combinations
from math import comb
from time import perf_counter

import numpy as np


class CachedOracle:
    """Evaluate a deterministic scalar v(mask) once per distinct coalition."""

    def __init__(self, k, value):
        if not isinstance(k, int) or k < 1:
            raise ValueError("k must be a positive integer")
        self.k = k
        self.value = value
        self.cache = {}
        self.seconds = 0.0

    def __call__(self, mask):
        mask = int(mask)
        if not 0 <= mask < 1 << self.k:
            raise ValueError("coalition mask outside player set")
        if mask not in self.cache:
            start = perf_counter()
            value = float(self.value(mask))
            self.seconds += perf_counter() - start
            if not np.isfinite(value):
                raise ValueError("oracle value must be finite")
            self.cache[mask] = value
        return self.cache[mask]


@dataclass
class Result:
    values: np.ndarray
    coalitions: int
    seconds: float
    rank: int | None = None


def _result(oracle, values, start, rank=None):
    return Result(np.asarray(values), len(oracle.cache), perf_counter() - start, rank)


def _positive(m):
    if not isinstance(m, (int, np.integer)) or m < 1:
        raise ValueError("m must be a positive integer")


def permutation_marginal(oracle, order):
    """One telescoping allocation, also useful for exhaustive order checks."""
    order = tuple(int(i) for i in order)
    if sorted(order) != list(range(oracle.k)):
        raise ValueError("order must be a permutation of all players")
    out = np.zeros(oracle.k)
    mask, previous = 0, oracle(0)
    for i in order:
        mask |= 1 << i
        current = oracle(mask)
        out[i] = current - previous
        previous = current
    return out


def permutation(oracle, m, seed=0, antithetic=False):
    """Average m uniform orders; optionally average each with its reverse.

    Every order is efficient. Antithetic sampling uses 2m orders, not m.
    An order and its reverse are exactly Shapley for any order-two game.
    """
    _positive(m)
    start = perf_counter()
    rng = np.random.default_rng(seed)
    values = np.zeros(oracle.k)
    for _ in range(m):
        order = rng.permutation(oracle.k)
        values += permutation_marginal(oracle, order)
        if antithetic:
            values += permutation_marginal(oracle, order[::-1])
    values /= m * (2 if antithetic else 1)
    return _result(oracle, values, start)


def kernel_shap(oracle, m, seed=0):
    """Paired KernelSHAP, m iid interior masks plus their complements.

    Sample size s with probability proportional to 1/[s(k-s)], then a uniform
    size-s subset. This is the SHAP kernel distribution, so sampled rows have
    equal weight. Repeated rows retain multiplicity while oracle calls cache.
    Eliminate the last coefficient to impose sum(phi)=v(N)-v(empty) exactly.
    A rank-deficient sample returns a least-squares solution and exposes rank;
    it is not silently repaired by extra draws. Finite-sample regression can
    be biased. k=1 needs endpoints only.
    """
    _positive(m)
    start = perf_counter()
    k = oracle.k
    full = (1 << k) - 1
    baseline = oracle(0)
    total = oracle(full) - baseline
    if k == 1:
        return _result(oracle, [total], start, 0)
    rng = np.random.default_rng(seed)
    sizes = np.arange(1, k)
    probabilities = 1.0 / (sizes * (k - sizes))
    probabilities /= probabilities.sum()
    masks = []
    for _ in range(m):
        s = rng.choice(sizes, p=probabilities)
        mask = sum(1 << int(i) for i in rng.choice(k, size=s, replace=False))
        masks.extend((mask, full ^ mask))
    z = np.array([[(mask >> i) & 1 for i in range(k)] for mask in masks], float)
    y = np.array([oracle(mask) - baseline for mask in masks])
    reduced = z[:, :-1] - z[:, -1, None]
    beta, _, rank, _ = np.linalg.lstsq(reduced, y - z[:, -1] * total, rcond=None)
    return _result(oracle, np.r_[beta, total - beta.sum()], start, int(rank))


def order_two(oracle):
    """Shapley of the order-two Moebius truncation, not generally efficient.

    Cost 1+k+k(k-1)/2. Higher-order interactions are discarded, not rescaled.
    Fixed gamma, no bounds, no spread makes positions linear in signals and
    net P&L quadratic: this truncation is then exact (including quadratic cost).
    """
    start = perf_counter()
    k = oracle.k
    baseline = oracle(0)
    single = np.array([oracle(1 << i) - baseline for i in range(k)])
    values = single.copy()
    for i, j in combinations(range(k), 2):
        interaction = oracle((1 << i) | (1 << j)) - baseline - single[i] - single[j]
        values[i] += interaction / 2
        values[j] += interaction / 2
    return _result(oracle, values, start)


def _subsets(items):
    for size in range(len(items) + 1):
        yield from combinations(items, size)


def owen(oracle, groups):
    """Exact Owen value for a partition: a distinct allocation convention.

    Average marginal contributions over group orders and within-group orders
    via subset weights. Only coalitions compatible with that partition occur.
    """
    start = perf_counter()
    groups = tuple(tuple(g) for g in groups)
    if (not groups or any(not g for g in groups)
            or sorted(i for g in groups for i in g) != list(range(oracle.k))):
        raise ValueError("groups must partition all players into nonempty sets")
    masks = [sum(1 << i for i in g) for g in groups]
    values = np.zeros(oracle.k)
    q = len(groups)
    for a, group in enumerate(groups):
        others = [j for j in range(q) if j != a]
        for preceding in _subsets(others):
            union = sum(masks[j] for j in preceding)
            outer = 1 / (q * comb(q - 1, len(preceding)))
            for i in group:
                for inner in _subsets([j for j in group if j != i]):
                    mask = union | sum(1 << j for j in inner)
                    weight = outer / (len(group) * comb(len(group) - 1, len(inner)))
                    values[i] += weight * (oracle(mask | (1 << i)) - oracle(mask))
    return _result(oracle, values, start)
