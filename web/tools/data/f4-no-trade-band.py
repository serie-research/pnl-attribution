"""Durin static spread model, pp. 17–18, Eqs. 49–54.

Total trades use sim/signal_attribution/optimiser.py independently; splits
restate the browser closed form and therefore do not independently test it.
"""
import json
from pathlib import Path
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from sim.signal_attribution.optimiser import _trade_diag
points=[]
for g1,g2,spread,gamma,previous in [(1,1,.5,1,0),(1,-1,.5,1,0),(1,0,.5,1,.5),(-2,.5,.25,2,-.5),(.2,.1,0,.5,1)]:
    z=g1+g2-gamma*previous
    position=_trade_diag(np.array([[g1+g2]]),np.array([[previous]]),np.array([gamma]),np.array([1.]),1.,spread,-np.inf,np.inf)[0].item()
    trade=position-previous
    split=[0.,0.] if abs(trade)<1e-14 else [(g-gamma*previous/2)/(1+gamma+spread/abs(trade)) for g in [g1,g2]]
    points.append(dict(g1=g1,g2=g2,spread=spread,gamma=gamma,previous=previous,z=z,trade=trade,split=split))
data=dict(source='Synthetic static no-trade model. Reference total trades use an independent optimiser; splits use the displayed closed form.',variance=1,quadratic_cost=1,previous_split=[.5,.5],points=points)
(ROOT/'web/public/data/f4-no-trade-band.json').write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
