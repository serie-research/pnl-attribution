"""Repartition completed S1b execution blocks without recomputing seed paths.

Only the unit identifier changes; seed ids and all numerical rows are retained.
The original results and timing records are kept.
"""
import argparse
from pathlib import Path
import polars as pl
from sim.common.io import atomic_write_parquet
from .units import configs, units


def main():
    p = argparse.ArgumentParser()
    p.add_argument('source', type=Path)
    p.add_argument('destination', type=Path)
    args = p.parse_args()
    frames = [pl.read_parquet(f) for f in sorted(args.source.glob('*.parquet'))]
    df = pl.concat(frames)
    cfgs = configs('full')
    written = []
    for unit in units('full'):
        _, name, block = unit.split('-')
        b = int(block[1:])
        cfg = cfgs[name]
        lo = b * cfg.seeds_per_block
        d = df.filter((pl.col('config') == name) & (pl.col('seed') >= lo) &
                      (pl.col('seed') < lo + cfg.seeds_per_block))
        assert d['seed'].n_unique() == cfg.seeds_per_block, unit
        d = d.with_columns(unit=pl.lit(unit))
        atomic_write_parquet(d, args.destination / f'{unit}.parquet')
        written.append(d)
    # No observations lost or added, including timings and trajectory rows.
    keys = [c for c in df.columns if c != 'unit']
    assert df.select(keys).sort(keys).equals(pl.concat(written).select(keys).sort(keys))
    print(f'{len(frames)} source blocks -> {len(written)} units; numerical rows identical')


if __name__ == '__main__':
    main()
