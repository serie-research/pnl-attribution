"""Rebuild the public sweep from the audited compact S1b aggregates.

Provenance and input SHA256s are in sim-figures/signal_attribution/full/article-summary.json.
No ignored parquet or solver run is needed. Run with python from any directory.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
data = json.loads((ROOT / 'sim-figures/signal_attribution/full/article-summary.json').read_text())
data.pop('inputs', None)
for config in data['configurations']:
    for method in config['methods']:
        method['label'] = method['label'].replace('full desk', 'full portfolio')
# Sample counts and unused dispersion metadata are in the source aggregates.
# Preserve all plotted numbers verbatim, including the fixed domain extrema.
for config in data['configurations']:
    for method in config['methods']:
        for signal in method['signals']:
            signal.pop('n', None)
        for key in ('residual_variance', 'convergence'):
            if method.get(key):
                method[key] = {'mean': method[key]['mean']}
(ROOT / 'web/public/data/f5-signal-attribution.json').write_text(json.dumps(data, separators=(',', ':'), allow_nan=False) + '\n')
