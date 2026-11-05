"""Fixed-seed net P&L paths using the committed diagonal study, without edits.

Run: uv run python web/tools/data/f13-pnl-paths.py
LOO is v(N)-v(N\\{k}) from the same target-risk coalition runs as Shapley.
All checks use all 1000 periods before sampling; exported paths keep every
fifth period plus zero and the endpoint. Public paths are rounded to ten
decimals (below the 1e-8 conservation tolerance); simulation checks use originals.
"""
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from sim.signal_attribution import attribution as A, optimiser as O
from sim.signal_attribution.desk import simulate
from sim.signal_attribution.units import configs

SEED = 20261029
TOL = 1e-8  # absolute simulation P&L units, period and cumulative checks
SCENARIOS = [('base', 'Base configuration'), ('spread_2', 'Spread 2.0 std(G)'),
             ('box_0p75', 'Tight position limit'), ('corr_0p9', 'High innovation correlation')]
METHODS = [('regression', 'One-step regression'), ('iterated', 'Iterated stand-alone proxies'),
           ('loo', 'Leave-one-out'), ('grinold_easton', 'Grinold–Easton'),
           ('durin', 'Durin'), ('shapley_target', 'Shapley · risk target'),
           ('shapley_gamma', 'Shapley · full-portfolio γ')]


def build():
    published = json.loads((ROOT / 'sim-figures/signal_attribution/full/article-summary.json').read_text())
    published_ids = {c['id'] for c in published['configurations']}
    data = dict(seed=SEED, units='Simulation P&L units: asset volatility = 1; portfolio risk target = 1 per period.',
                sampling='Every fifth cumulative endpoint, plus period zero; checks cover every period.',
                tolerance=TOL, methods=[dict(id=k, label=v) for k, v in METHODS], scenarios=[])
    for config, label in SCENARIOS:
        assert config in published_ids
        desk = configs('full')[config].desk
        market = simulate(desk, np.random.default_rng(SEED))
        g, r = market.g, market.r
        K, T, _ = g.shape
        full = O.solve(desk, g.sum(0)[None], sigma_target=np.array([desk.risk_target]))
        total = O.pnl(desk, full, r)[0]
        parts = {}
        for name, positions in [('durin', A.durin_positions), ('grinold_easton', A.grinold_easton_positions)]:
            x, dx = positions(desk, full, g)
            parts[name] = A.component_pnl(desk, x, dx, full, r)
        GS = A.coalition_signals(g)
        for convention in ('target', 'gamma'):
            opts = dict(sigma_target=np.full(len(GS), desk.risk_target)) if convention == 'target' else dict(gamma_path=np.broadcast_to(full.gamma, (len(GS), T)))
            sol = O.solve(desk, GS, **opts)
            v = np.vstack([np.zeros(T), O.pnl(desk, sol, r)])
            parts['shapley_' + convention] = A.shapley(v, K)
            if convention == 'target':
                _, parts['regression'], _ = A.regress(total, v[A.singleton_index(K)])
                parts['loo'] = v[-1] - v[(2**K - 1) ^ A.singleton_index(K)]
        def run_pnl(G, sigma, lo, hi):
            return O.pnl(desk, O.solve(desk, G, sigma_target=sigma, lo=lo, hi=hi), r)
        em = A.iterated(desk, g, total, run_pnl)
        parts['iterated'] = em.attributed
        times = np.r_[0, np.arange(5, T + 1, 5)]
        def cumulative(values):
            return np.concatenate([np.zeros((*values.shape[:-1], 1)), np.cumsum(values, axis=-1)], axis=-1)
        ctotal = cumulative(total)
        scenario = dict(id=config, label=label, parameters=asdict(desk), periods=times.tolist(),
                        total=ctotal[times].tolist(), methods={}, checks={},
                        iteration=dict(status=em.status, converged=em.converged, n_iter=em.n_iter,
                                       beta=em.beta.tolist(), risk_targets=em.sigma.tolist()))
        for name, _ in METHODS:
            values = parts[name]
            labels = list(desk.names)
            exact = name in ('durin', 'grinold_easton', 'shapley_target', 'shapley_gamma')
            if name == 'grinold_easton':
                labels += ['spread cost', 'position limit']
            if not exact:
                values = np.vstack([values, total - values.sum(0)])
                labels += ['residual']
            cv = cumulative(values)
            period_error = float(np.max(np.abs(values.sum(0) - total)))
            cumulative_error = float(np.max(np.abs(cv.sum(0) - ctotal)))
            passed = max(period_error, cumulative_error) <= TOL
            assert passed, (config, name, period_error, cumulative_error)
            scenario['checks'][name] = dict(exact=exact, includes_residual=not exact,
                max_period_error=period_error, max_cumulative_error=cumulative_error, passed=passed)
            scenario['methods'][name] = [dict(label=l, values=p[times].tolist()) for l, p in zip(labels, cv)]
        data['scenarios'].append(scenario)
        print(config, scenario['iteration'], flush=True)
    data['exact_sum_check'] = all(v['passed'] for s in data['scenarios'] for v in s['checks'].values())
    data = compact(data)
    encoded = json.dumps(data, separators=(',', ':'), allow_nan=False) + '\n'
    assert len(encoded.encode()) < 1_500_000
    (ROOT / 'web/public/data/f13-pnl-paths.json').write_text(encoded)
    print(f'{len(encoded.encode())} bytes; all full-resolution checks passed')


def compact(data):
    """Trim unused metadata; keep displayed values and exported conservation."""
    for scenario in data['scenarios']:
        scenario['parameters'] = {k: scenario['parameters'][k] for k in
                                  ('spread_rel', 'box', 'corr', 'n_assets', 'n_periods')}
        scenario['iteration'] = {k: scenario['iteration'][k] for k in ('status', 'n_iter')}
        paths = [scenario['total']] + [c['values'] for m in scenario['methods'].values() for c in m]
        for path in paths:
            for i, value in enumerate(path):
                rounded = round(value, 10)
                # JS formats negative zero as "0.000"; keep tiny negative values.
                if value < 0 and rounded == 0:
                    rounded = value
                assert f'{value:.3f}' == f'{rounded:.3f}'
                path[i] = rounded
        for components in scenario['methods'].values():
            assert all(abs(sum(c['values'][t] for c in components) - total) <= data['tolerance']
                       for t, total in enumerate(scenario['total']))
    return data


if __name__ == '__main__':
    build()
