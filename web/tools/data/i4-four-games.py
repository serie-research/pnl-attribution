"""Worked example for the four games of infographic i4 (section 4.2).

Outcome: the daily P&L of a desk (synthetic, k EUR).
Indicators: volatility (annualised, %) and spread (bp), jointly lognormal, correlation near 0.6.
Model: a linear regression of the P&L on both indicators, fitted on a seeded synthetic history.
The P&L is slightly concave in volatility, so the linear model is an approximation.

The four games are evaluated on the model's own distribution (Gauss-Hermite quadrature,
no sampling noise); the regression is the population least-squares fit.
Writes web/tools/data/i4-four-games.json; its numbers are copied by hand into
web/public/data/w-i4-shared-information.json (rounded).
"""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
P = dict(
    seed=20261029, n_days=100_000, quad_nodes=80,
    log_vol_mu=np.log(20.0), log_vol_sd=0.40,      # volatility, % per year
    log_spr_mu=np.log(4.0), log_spr_sd=0.40,       # spread, bp
    rho_log=0.62,                                   # correlation of the logarithms
    pnl_const=-20.0, pnl_vol=2.5, pnl_spr=2.5,     # k EUR, k EUR per vol point, k EUR per bp
    pnl_curv=0.05,                                  # k EUR per (vol point)^2 lost beyond the median
    pnl_noise=10.0,                                 # k EUR, sd of the unexplained part
    day_vol=35.0, day_spr=3.0,                      # the explained day: high volatility
)


def indicators(z1, z2):
    return (np.exp(P['log_vol_mu'] + P['log_vol_sd'] * z1), np.exp(P['log_spr_mu'] + P['log_spr_sd'] * z2))


def mean_pnl(v, s):
    return P['pnl_const'] + P['pnl_vol'] * v + P['pnl_spr'] * s - P['pnl_curv'] * (v - 20.0) ** 2


# --- the fitted model: OLS on a seeded history ---
rng = np.random.default_rng(P['seed'])
r = P['rho_log']
z1 = rng.standard_normal(P['n_days'])
z2 = r * z1 + np.sqrt(1 - r * r) * rng.standard_normal(P['n_days'])
hv, hs = indicators(z1, z2)
hy = mean_pnl(hv, hs) + P['pnl_noise'] * rng.standard_normal(P['n_days'])
X = np.column_stack([np.ones_like(hv), hv, hs])
b = np.linalg.lstsq(X, hy, rcond=None)[0]
corr_vs = float(np.corrcoef(hv, hs)[0, 1])
b_sample = b

# --- quadrature on the true joint law ---
t, w = np.polynomial.hermite_e.hermegauss(P['quad_nodes'])
w = w / w.sum()
T1, T2 = np.meshgrid(t, t, indexing='ij')
W = np.outer(w, w)
Z1, Z2 = T1, r * T1 + np.sqrt(1 - r * r) * T2              # joint law, weights W
V, S = indicators(Z1, Z2)
M = mean_pnl(V, S)
# The model is the least-squares fit on a history of unbounded length (population moments),
# so the three global games agree exactly at v(N); the seeded 100,000-day fit is kept as a check.
Wf_, Vf_, Sf_, Mf_ = W.reshape(-1), V.reshape(-1), S.reshape(-1), M.reshape(-1)
A_ = np.stack([Vf_, Sf_]); mu_ = np.einsum('ij,j->i', A_, Wf_); Ac_ = A_ - mu_[:, None]
slope = np.linalg.solve(np.einsum('ij,kj,j->ik', Ac_, Ac_, Wf_), np.einsum('ij,j,j->i', Ac_, Wf_, Mf_ - (Wf_ * Mf_).sum()))
b = np.array([(Wf_ * Mf_).sum() - slope @ mu_, *slope])
f = lambda v, s: b[0] + b[1] * v + b[2] * s
F = f(V, S)
E = lambda a: float((W * a).sum())
var = lambda a: E(a * a) - E(a) ** 2
var_y = var(M) + P['pnl_noise'] ** 2

# E[f | one indicator] at any log-indicator value z (1-D quadrature over the other indicator)
def g_of(first, z):
    z = np.asarray(z, float)
    zo = r * z[..., None] + np.sqrt(1 - r * r) * t
    zz = np.broadcast_to(z[..., None], zo.shape)
    v_, s_ = indicators(zz, zo) if first == 'vol' else indicators(zo, zz)
    return (w * f(v_, s_)).sum(-1)

mean_f = E(F)
var1 = lambda g: float((w * g * g).sum() - (w * g).sum() ** 2)

# --- game 1: refitted R2 (population regression of the P&L on each subset) ---
def r2_subset(cols):
    if not cols:
        return 0.0
    A = np.stack([cols_[c] for c in cols])              # indicator values on the 2-D grid
    Am = A.reshape(len(cols), -1)
    Wf = W.reshape(-1)
    mu = np.einsum('ij,j->i', Am, Wf)
    Ac = Am - mu[:, None]
    C = np.einsum('ij,kj,j->ik', Ac, Ac, Wf)
    my = float((Wf * M.reshape(-1)).sum())
    c = np.einsum('ij,j,j->i', Ac, Wf, M.reshape(-1) - my)   # cov(indicators, noise-free P&L); noise is independent
    return float(c @ np.linalg.solve(C, c) / var_y)
cols_ = {'vol': V, 'spr': S}
R2 = {'0': 0.0, 'vol': r2_subset(['vol']), 'spr': r2_subset(['spr']), 'N': r2_subset(['vol', 'spr'])}

# --- game 2: variance of the model's conditional mean ---
CV = {'0': 0.0, 'vol': var1(g_of('vol', t)), 'spr': var1(g_of('spr', t)), 'N': var(F)}

# --- game 3: SAGE with squared error: MSE of the mean-only model minus MSE of E[f | subset] ---
mse0 = var_y + (E(M) - mean_f) ** 2
sage_mse = lambda G: P['pnl_noise'] ** 2 + E((M - G) ** 2)
SAGE = {'0': 0.0, 'vol': mse0 - sage_mse(g_of('vol', Z1)), 'spr': mse0 - sage_mse(g_of('spr', Z2)),
        'N': mse0 - sage_mse(F)}

# --- game 4: local conditional SHAP on one day ---
zday1 = (np.log(P['day_vol']) - P['log_vol_mu']) / P['log_vol_sd']
zday2 = (np.log(P['day_spr']) - P['log_spr_mu']) / P['log_spr_sd']
def cond_at(first_fixed, zfix):
    zo = r * zfix + np.sqrt(1 - r * r) * t
    v_, s_ = (P['day_vol'] * np.ones_like(zo), indicators(0, zo)[1]) if first_fixed == 'vol' else (indicators(zo, 0)[0], P['day_spr'] * np.ones_like(zo))
    return float((w * f(v_, s_)).sum())
SH = {'0': mean_f, 'vol': cond_at('vol', zday1), 'spr': cond_at('spr', zday2), 'N': float(f(P['day_vol'], P['day_spr']))}


def shapley(vv):
    d1 = 0.5 * ((vv['vol'] - vv['0']) + (vv['N'] - vv['spr']))
    d2 = 0.5 * ((vv['spr'] - vv['0']) + (vv['N'] - vv['vol']))
    return d1, d2


usual_spread = float(np.exp(P['log_spr_mu'] + P['log_spr_sd'] * (r * zday1) + 0.5 * (1 - r * r) * P['log_spr_sd'] ** 2))  # E[spread | day volatility]
out = {'usual_spread_given_day_volatility': usual_spread, 'parameters': {k: (float(v) if not isinstance(v, int) else v) for k, v in P.items()},
       'fitted_coefficients': [float(x) for x in b], 'sample_fit_check': [float(x) for x in b_sample], 'corr_vol_spread_history': corr_vs,
       'var_pnl': var_y, 'games': {}}
for key, vv in (('r2', R2), ('cv', CV), ('shap', SH), ('sage', SAGE)):
    phi = shapley(vv)
    total = vv['N'] - vv['0']
    out['games'][key] = {'v': {k: float(x) for k, x in vv.items()}, 'phi_vol': phi[0], 'phi_spr': phi[1],
                         'share_vol': phi[0] / total, 'share_spr': phi[1] / total, 'total': total}
(ROOT / 'web/tools/data/i4-four-games.json').write_text(json.dumps(out, indent=2) + '\n')
print(json.dumps(out, indent=2))
