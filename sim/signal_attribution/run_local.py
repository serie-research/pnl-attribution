"""Run an isolated tier locally; results and timing records are kept."""
import argparse
import concurrent.futures
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument("tier", choices=["smoke", "full"])
    p.add_argument("out", type=Path)
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()
    from .units import units
    args.out.mkdir(parents=True, exist_ok=True)
    env = os.environ | {k: "1" for k in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS", "POLARS_MAX_THREADS"]}
    def run(unit):
        t = time.monotonic()
        with (args.out / f"{unit}.log").open("w") as log:
            proc = subprocess.run([sys.executable, "-m", "sim.cli", "run", "signal_attribution", unit,
                                   "--out", str(args.out / f"{unit}.parquet")], env=env, stdout=log, stderr=log)
        return unit, proc.returncode, time.monotonic() - t
    failed = False
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
        for unit, code, seconds in ex.map(run, units(args.tier)):
            print(f"{unit}\t{code}\t{seconds:.3f}", flush=True)
            failed |= code != 0
    raise SystemExit(int(failed))


if __name__ == "__main__":
    main()
