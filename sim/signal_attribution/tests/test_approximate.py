"""Allocation identities, exact references, and actual oracle-call accounting."""
from itertools import permutations

import numpy as np
import pytest

from sim.signal_attribution.approximate import (
    CachedOracle, kernel_shap, order_two, owen, permutation, permutation_marginal,
)
from sim.signal_attribution.attribution import shapley, coalition_signals
from sim.signal_attribution.desk import Desk, simulate
from sim.signal_attribution import optimiser as O


def reference(v, k):
    return shapley(np.array([v(s) for s in range(1 << k)])[:, None], k)[:, 0]


def quadratic(mask):
    z = (mask >> np.arange(4)) & 1
    return 7 + z @ np.array([1., -2., 3., 4.]) + 5*z[0]*z[1] - 2*z[1]*z[3]


def test_uniform_orders_unbiased_and_each_order_efficient():
    # A nonsymmetric game with higher-order interactions and nonzero baseline.
    v = lambda s: quadratic(s) + 9 * ((s & 7) == 7)
    oracle = CachedOracle(4, v)
    estimates = [permutation_marginal(oracle, p) for p in permutations(range(4))]
    np.testing.assert_allclose(np.sum(estimates, axis=1), v(15) - v(0))
    np.testing.assert_allclose(np.mean(estimates, axis=0), reference(v, 4))
    assert len(oracle.cache) == 16


@pytest.mark.parametrize('method', [
    lambda o: permutation(o, 1), lambda o: permutation(o, 1, antithetic=True),
    lambda o: kernel_shap(o, 40), order_two,
    lambda o: owen(o, [[0, 1], [2, 3]]),
])
def test_additive_exact(method):
    weights = np.array([2., -4., 0., 7.])
    v = lambda s: 13 + ((s >> np.arange(4)) & 1) @ weights
    np.testing.assert_allclose(method(CachedOracle(4, v)).values, weights, atol=1e-12)


@pytest.mark.parametrize('method', [
    lambda o: permutation(o, 1, antithetic=True),
    lambda o: kernel_shap(o, 40), order_two,
    lambda o: owen(o, [[0, 1], [2, 3]]),
])
def test_order_two_exact(method):
    np.testing.assert_allclose(method(CachedOracle(4, quadratic)).values,
                               reference(quadratic, 4), atol=1e-12)


def test_plain_single_order_is_not_claimed_exact_on_quadratic_game():
    assert not np.allclose(permutation(CachedOracle(4, quadratic), 1).values,
                           reference(quadratic, 4))


def test_kernel_efficiency_and_deficient_rank_exposed():
    v = lambda s: quadratic(s) + 9 * ((s & 7) == 7)
    for m in [1, 2, 40]:
        out = kernel_shap(CachedOracle(4, v), m)
        np.testing.assert_allclose(out.values.sum(), v(15) - v(0))
        assert out.coalitions <= min(16, 2 + 2*m)
        if m == 1:
            assert out.rank < 3
        else:
            assert 0 <= out.rank <= 3


def test_pair_quadratic_exact_for_every_full_rank_sample():
    for seed in range(30):
        out = kernel_shap(CachedOracle(4, quadratic), 4, seed=seed)
        if out.rank == 3:
            np.testing.assert_allclose(out.values, reference(quadratic, 4), atol=1e-12)


def test_truncation_cost_and_missing_high_order_efficiency():
    calls = []
    def v(s):
        calls.append(s)
        return float(s == 15)
    out = order_two(CachedOracle(4, v))
    assert out.coalitions == len(calls) == len(set(calls)) == 11
    np.testing.assert_array_equal(out.values, np.zeros(4))
    assert out.values.sum() != v(15)


def test_owen_is_distinct_convention_and_matches_group_orders():
    v = lambda s: 2 + float((s & 7) == 7)
    oracle = CachedOracle(3, v)
    out = owen(oracle, [[0, 1], [2]])
    np.testing.assert_allclose(out.values, [.25, .25, .5])
    assert not np.allclose(out.values, reference(v, 3))
    orders = [(0, 1, 2), (1, 0, 2), (2, 0, 1), (2, 1, 0)]
    np.testing.assert_allclose(out.values,
        np.mean([permutation_marginal(oracle, p) for p in orders], axis=0))
    np.testing.assert_allclose(out.values.sum(), 1)
    for groups in [[[0], [1], [2]], [[0, 1, 2]]]:
        np.testing.assert_allclose(owen(CachedOracle(3, v), groups).values, reference(v, 3))


def test_oracle_cached_calls_and_single_player():
    calls = []
    def value(s):
        calls.append(s)
        return 3 + s*5
    oracle = CachedOracle(1, value)
    for method in [lambda o: permutation(o, 9), lambda o: kernel_shap(o, 3),
                   order_two, lambda o: owen(o, [[0]])]:
        out = method(oracle)
        np.testing.assert_allclose(out.values, [5])
        assert out.coalitions == 2
    assert calls == [0, 1]


@pytest.mark.parametrize('method', [permutation, kernel_shap])
def test_invalid_draw_counts(method):
    with pytest.raises(ValueError):
        method(CachedOracle(3, float), 0)


def test_invalid_partition():
    with pytest.raises(ValueError):
        owen(CachedOracle(3, float), [[0, 1], [1, 2]])


def test_fixed_gamma_no_bounds_no_spread_backtest_is_order_two():
    desk = Desk(n_assets=8, n_periods=60, box=np.inf, spread_rel=0)
    market = simulate(desk, np.random.default_rng(31))
    full = O.solve(desk, market.g.sum(0)[None], sigma_target=np.array([1.]))
    signals = coalition_signals(market.g)
    solution = O.solve(desk, signals, gamma_path=np.broadcast_to(full.gamma, (15, 60)))
    table = np.r_[0., O.pnl(desk, solution, market.r).sum(axis=1)]
    expected = reference(lambda s: table[s], 4)
    for method in [order_two, lambda o: permutation(o, 1, antithetic=True),
                   lambda o: kernel_shap(o, 40)]:
        np.testing.assert_allclose(method(CachedOracle(4, lambda s: table[s])).values,
                                   expected, atol=1e-10)


def test_kernel_full_weighted_design_recovers_high_order_shapley(monkeypatch):
    # K=4 kernel weights per mask: size 1/3 twice size 2. Listing each
    # mask with that multiplicity reproduces the exact regression objective.
    from sim.signal_attribution import approximate
    masks = [s for s in range(1, 15) for _ in range(2 if s.bit_count() in (1, 3) else 1)]
    class Design:
        def __init__(self):
            self.position = 0
        def choice(self, items, p=None, size=None, replace=None):
            mask = masks[self.position]
            if size is None:
                return mask.bit_count()
            self.position += 1
            return np.array([i for i in range(4) if mask & (1 << i)])
    monkeypatch.setattr(approximate.np.random, 'default_rng', lambda seed: Design())
    v = lambda s: quadratic(s) + 9 * ((s & 7) == 7) - 3*(s == 15)
    out = kernel_shap(CachedOracle(4, v), len(masks))
    assert out.rank == 3
    assert out.coalitions == 16
    np.testing.assert_allclose(out.values, reference(v, 4), atol=1e-12)
