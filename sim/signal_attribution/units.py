"""Study interface: units are one configuration x one block of seeds.

Unit id: ``<tier>-<config>-b<block>``, e.g. ``full-box_1p0-b01``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from sim.common.io import atomic_write_parquet, unit_seed

from .desk import Desk
from .experiment import ITERATED_VARIANTS, run_seed

STUDY = "signal_attribution"

IC6 = (0.06, 0.05, 0.04, 0.03, 0.05, 0.04)
RHO6 = (0.5, 0.95, 0.5, 0.95, 0.8, 0.8)
NAMES6 = ("fast A", "slow A", "fast B", "slow B", "mid A", "mid B")

# Base values of the four swept axes (the base desk sits on every sweep).
BASE_AXES = {"corr": 0.3, "box": 1.5, "spread": 0.5, "k": 4}
SWEEPS = {
    "full": {
        "corr": [-0.3, 0.0, 0.6, 0.9],
        # box <= 1: sigma* is out of reach (all positions at the bound give sigma*)
        "box": [float("inf"), 3.0, 2.0, 1.25, 1.1, 0.75],
        "spread": [0.0, 0.25, 1.0, 2.0],
        "k": [2, 3, 5, 6],
    },
    "smoke": {"corr": [0.9], "box": [float("inf"), 1.1], "spread": [2.0], "k": [2]},
}


@dataclass(frozen=True)
class Config:
    desk: Desk
    axis: str  # "base", a sweep axis, or "design"
    axis_value: float
    seeds_per_block: int
    n_blocks: int
    variants: tuple[str, ...] = ("iterated",)
    keep_paths: bool = False
    seed_block_size: int | None = None  # retain S1 seed identities after repartition


def _tag(v: float) -> str:
    if np.isinf(v):
        return "inf"
    s = f"{v:g}".replace("-", "m").replace(".", "p")
    return s


def _k_desk(base: Desk, k: int) -> Desk:
    return base.with_(ic=IC6[:k], rho=RHO6[:k], names=NAMES6[:k])


def configs(tier: str) -> dict[str, Config]:
    """Every configuration of a tier, by name."""
    if tier == "full":
        base = Desk()
        spb, nb = 10, 2  # 20 seeds per configuration
        small = Desk(n_assets=20, n_periods=250, cov="factor")
        fspb, fnb = 5, 4
    else:
        base = Desk(n_assets=20, n_periods=300)
        spb, nb = 2, 1
        small = Desk(n_assets=8, n_periods=40, cov="factor")
        fspb, fnb = 1, 1
    out: dict[str, Config] = {"base": Config(base, "base", 0.0, spb, nb)}
    for axis, values in SWEEPS[tier].items():
        for v in values:
            if axis == "corr":
                d = base.with_(corr=v)
            elif axis == "box":
                d = base.with_(box=v)
            elif axis == "spread":
                d = base.with_(spread_rel=v)
            else:
                d = _k_desk(base, int(v))
            stress = (axis == "corr" and v >= 0.6) or (axis == "k" and v >= 5) or (axis == "spread" and v >= 1)
            variants = ("iterated", "iterated:damped", "iterated:ridge", "iterated:ridge_damped") if stress else ("iterated",)
            out[f"{axis}_{_tag(v)}"] = Config(
                d, axis, float(v), 2 if stress and tier == "full" else spb,
                10 if stress and tier == "full" else nb, variants=variants,
                seed_block_size=10 if stress and tier == "full" else None,
            )
    # designs with a known answer
    c3 = ((1.0, 0.3, 0.0), (0.3, 1.0, 0.0), (0.0, 0.0, 1.0))
    out["zeroic"] = Config(
        base.with_(ic=(0.06, 0.05, 0.0), ic_assumed=(0.06, 0.05, 0.05), rho=(0.5, 0.95, 0.8),
                   corr=c3, names=("fast A", "slow A", "no-IC")),
        "design", 0.0, spb, nb,
    )
    out["identical"] = Config(
        base.with_(ic=(0.06, 0.04, 0.04), rho=(0.5, 0.95, 0.95), corr=0.3, clones=((2, 1),),
                   names=("fast A", "slow A", "slow A copy")),
        "design", 0.0, spb, nb,
    )
    # Durin's value and momentum long-only fund (p. 3, p. 13): anti-correlated
    # signals (stationary correlation -0.5), no-short constraint on the total
    out["valmom"] = Config(
        base.with_(ic=(0.05, 0.05), rho=(0.95, 0.8), corr=-0.64, long_only=True, box=2.0,
                   spread_rel=0.25, names=("value", "momentum")),
        "design", 0.0, spb, nb,
    )
    out["b2"] = Config(
        base.with_(ic=(0.06, 0.05), rho=(0.5, 0.95), corr=0.3, box=1.2, spread_rel=1.0,
                   names=("fast", "slow")),
        "design", 0.0, max(2, spb // 2), 1, keep_paths=True,
    )
    out["switches"] = Config(
        base, "design", 0.0, 2 if tier == "full" else spb, 10 if tier == "full" else nb,
        variants=tuple(v for v in ITERATED_VARIANTS if v not in
                       ("iterated:damped", "iterated:ridge", "iterated:ridge_damped")),
        seed_block_size=10 if tier == "full" else None,
    )
    out["factor"] = Config(small, "design", 0.0, fspb, fnb)
    return out


def units(tier: str) -> list[str]:
    return [
        f"{tier}-{name}-b{b:02d}"
        for name, c in configs(tier).items()
        for b in range(c.n_blocks)
    ]


def parse(unit: str) -> tuple[str, str, int]:
    tier, rest = unit.split("-", 1)
    name, block = rest.rsplit("-b", 1)
    if name not in configs(tier):
        raise KeyError(unit)
    return tier, name, int(block)


def seed_identity(tier: str, name: str, block: int, index: int) -> tuple[str, int]:
    """Map smaller execution blocks to the original S1 seed blocks."""
    cfg = configs(tier)[name]
    original_size = cfg.seed_block_size or cfg.seeds_per_block
    absolute_index = block * cfg.seeds_per_block + index
    return (f"{tier}-{name}-b{absolute_index // original_size:02d}",
            absolute_index % original_size)


def run_unit(unit: str, out: Path) -> None:
    tier, name, block = parse(unit)
    cfg = configs(tier)[name]
    frames = []
    for i in range(cfg.seeds_per_block):
        t0 = time.perf_counter()
        source_unit, source_index = seed_identity(tier, name, block, i)
        rows = run_seed(cfg.desk, [unit_seed(STUDY, source_unit), source_index], cfg.variants,
                        keep_paths=cfg.keep_paths and i == 0)
        rows.add("timing", "_all", "seconds", time.perf_counter() - t0)
        m, s, k, t, v = zip(*rows.rows)
        frames.append(pl.DataFrame({
            "method": m, "signal": s, "metric": k,
            "t": pl.Series(t, dtype=pl.Int32), "value": v,
        }).with_columns(seed=pl.lit(block * cfg.seeds_per_block + i, pl.Int32)))
    df = pl.concat(frames).with_columns(
        unit=pl.lit(unit), config=pl.lit(name), axis=pl.lit(cfg.axis),
        axis_value=pl.lit(cfg.axis_value), tier=pl.lit(tier),
        n_signals=pl.lit(cfg.desk.n_signals, pl.Int32),
    )
    atomic_write_parquet(df, out)
