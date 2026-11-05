# Reproduce the synthetic examples

Extract reproduce.zip and run commands from its root. Python 3.13 and uv are
required. `uv sync --group dev` installs NumPy, SciPy, Polars, Matplotlib,
CVXPY and pytest, using the dependency declarations in pyproject.toml.
Times below are from one local run; they vary with hardware.

## Figure data

Each script in web/tools/data/ writes the data of one figure of the page:

- f4-no-trade-band.py: the no-trade band (section 3.3);
- f6-coalitions.py: the three-signal coalition game (section 3.4);
- f5-signal-attribution.py and f13-pnl-paths.py: the signal allocations and
  P&L paths (section 3.5);
- f7-correlated.py: the two correlated indicators (section 4.1);
- i4-four-games.py: the four games of the diagram at the start of section 4.

Run, for example, `uv run python web/tools/data/f4-no-trade-band.py`. Outputs
go to web/public/data/, except i4-four-games.py, which writes
web/tools/data/i4-four-games.json. The f4, f13 and i4 scripts recompute their
figures, and f13 can take minutes; f6 writes a constructed game. The f5 and f7
scripts export the summaries archived under sim-figures/ and do not rerun the
studies behind them. The simulation code of the signal study is in the archive
(next section); that of the indicator study behind f7 is not.

Browser figures f1, f3 and f14 (sections 2.1 and 2.3) compute in the page. The
embeds/ directory contains their exact HTML and extracted JavaScript, plus the
other figures of the page. They require D3 v7, the page's CSS variables and a
[data-embed] container whose data-url points to the matching JSON. The scripts
refer to document.currentScript and run inside that container. They are not
command-line programs. The published JSON files contain their initial
parameters.

## Section 3.5 tables and signal study

Run the complete sweep (107 units, 485 seeds, 20 seeds for each main
configuration; some variants use fewer seeds):

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 uv run python -m sim.signal_attribution.run_local full results --workers 8
uv run python -m sim.signal_attribution.summarise results
uv run python -m sim.cli figures signal_attribution --results results --figdir figures --tier full
```

The original full sweep took about 6 hours of compute, summed over study units,
not the elapsed time of the parallel command above. The smoke tier is a
shorter execution check: replace full with smoke and use a separate directory.

Run the tests, including closed-form/CVXPY comparisons and approximate Shapley:

```sh
uv run pytest sim/signal_attribution/tests
```

Run the approximate Shapley study described in section 3.4:

```sh
uv run python -m sim.signal_attribution.approximate_study --help
```

See `--help` for the output directory and study-size options. The archived
article-summary.json records the published table inputs. Floating-point
results can vary by numerical library.
