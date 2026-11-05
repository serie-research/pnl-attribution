"""Constructed teaching game, independent of simulation runs.

The non-additive default distinguishes stand-alone, leave-one-out and Shapley.
The values are illustrative inputs, not fitted data or reference outputs.
"""
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
data = dict(source='Constructed synthetic teaching game; editable integer P&L units, not simulation estimates.',names=['A','B','C'],values=[0,4,3,8,4,6,9,12])
(ROOT/'web/public/data/f6-coalitions.json').write_text(json.dumps(data,indent=2)+'\n')
