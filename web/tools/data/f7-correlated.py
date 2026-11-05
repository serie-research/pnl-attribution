"""Package the audited B3 study for figure 4.1: per-sample coefficients and Shapley values.

The default run reads only the committed summary, so the public archive can
rerun it. `--refresh-summary RESULTS` rebuilds the per-sample fields of that
summary from the ignored full-linear parquet files, after checking each file
against the SHA256 recorded in the summary's provenance.
"""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SUMMARY = ROOT / 'sim-figures/indicator_importance/full/article-summary.json'


def refresh(results: Path) -> None:
    import numpy as np
    import polars as pl

    summary = json.loads(SUMMARY.read_text())
    expected = {p['file']: p['sha256'] for p in summary['provenance']}
    for name, digest in expected.items():
        actual = hashlib.sha256((results / name).read_bytes()).hexdigest()
        assert actual == digest, f'{name}: SHA256 differs from the recorded provenance'
    for row in summary['rows']:
        tag = f"full-linear-r{round(100 * row['rho']):03d}-"
        files = sorted(name for name in expected if name.startswith(tag))
        frame = pl.concat([pl.read_parquet(results / name) for name in files]).filter(
            (pl.col('tier') == 'full') & pl.col('feature').is_in([0, 1]))
        wide = (frame.filter(pl.col('metric').is_in(['ols_beta', 'shapley_r2']))
                .with_columns((pl.col('metric') + '_' + pl.col('feature').cast(pl.String)).alias('key'))
                .pivot(on='key', index=['unit', 'rep'], values='value').sort(['unit', 'rep']))
        assert wide.height == len(row['pairs']) == summary['repeats']
        # The committed pairs are in (unit, rep) order; the coefficients must match exactly.
        beta = wide.select(['ols_beta_0', 'ols_beta_1']).to_numpy()
        assert np.array_equal(beta, [[p['beta1'], p['beta2']] for p in row['pairs']]), 'pair order changed'
        shapley = wide.select(['shapley_r2_0', 'shapley_r2_1']).to_numpy()
        for j, stat in enumerate(row['metrics']['shapley_r2']):
            assert abs(shapley[:, j].mean() - stat['mean']) < 1e-12
            assert abs(shapley[:, j].std(ddof=1) - stat['sd']) < 1e-12
        for pair, (s1, s2) in zip(row['pairs'], shapley.tolist()):
            pair['shapley1'], pair['shapley2'] = s1, s2
    SUMMARY.write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')


def package() -> None:
    data = json.loads(SUMMARY.read_text())
    data.pop('provenance')
    for row in data['rows']:
        assert all('shapley1' in p for p in row['pairs']), 'run --refresh-summary first'
    (ROOT / 'web/public/data/f7-correlated.json').write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--refresh-summary', type=Path, metavar='RESULTS')
    args = parser.parse_args()
    if args.refresh_summary:
        refresh(args.refresh_summary.resolve())
    package()
