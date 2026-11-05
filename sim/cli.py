"""Command-line entry point: ``python -m sim.cli <command> <study> ...``.

Commands:
  units   <study> --tier smoke|full            print one unit id per line
  run     <study> <unit> --out FILE            run one unit, write FILE atomically
  figures <study> --results DIR --figdir DIR --tier smoke|full
"""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path


def _module(study: str, name: str):
    return importlib.import_module(f"sim.{study}.{name}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="sim.cli")
    sub = parser.add_subparsers(dest="command", required=True)

    p_units = sub.add_parser("units")
    p_units.add_argument("study")
    p_units.add_argument("--tier", choices=["smoke", "full"], required=True)

    p_run = sub.add_parser("run")
    p_run.add_argument("study")
    p_run.add_argument("unit")
    p_run.add_argument("--out", type=Path, required=True)

    p_fig = sub.add_parser("figures")
    p_fig.add_argument("study")
    p_fig.add_argument("--results", type=Path, required=True)
    p_fig.add_argument("--figdir", type=Path, required=True)
    p_fig.add_argument("--tier", choices=["smoke", "full"], required=True)

    args = parser.parse_args(argv)
    if args.command == "units":
        for unit in _module(args.study, "units").units(args.tier):
            print(unit)
    elif args.command == "run":
        _module(args.study, "units").run_unit(args.unit, args.out)
    elif args.command == "figures":
        for path in _module(args.study, "figures").make_figures(
            args.results, args.figdir, args.tier
        ):
            print(path)


if __name__ == "__main__":
    main()
