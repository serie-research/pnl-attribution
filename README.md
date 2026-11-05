# Trading P&L attribution

An English article on attributing trading P&L, with interactive synthetic
examples, an executed notebook, and a simulation package. Open `/attribution/`
after building the site. All browser assets are served locally. External paper
links are optional reading; reproduction does not fetch paper content.

## Reproduce

Use Python 3.12, Node 20 and Make. Create an environment and install the exact
Python dependencies (including transitive dependencies):

```sh
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
make all
make serve
```

Visit http://localhost:4321/attribution/. `make site` installs the exact npm
lock using `npm ci` before building. Installation requires network access;
all numerical checks and the built site work offline. To rebuild offline after
installation, run `make site-build` instead. npm dependency advisories from the
inherited template have not been remediated in this export.

Or build the image (base manifests and dependencies are pinned):

```sh
docker build -t pnl-attribution .
docker run --rm --network none pnl-attribution make all
# Local-only port for the built site:
docker run --rm -p 127.0.0.1:4321:4321 pnl-attribution make serve
```

The image installs dependencies and builds the site during image construction.
The numerical and site checks in `make all` use no network inside the image.
Outputs in a disposable container disappear at exit; mount a directory at
`/results` and set `OUTPUT=/results` to retain execution receipts.

## Targets and expected times

- `make notebook`: execute every code cell, including the embedded figure
  reference comparisons and solver/conservation assertions. Writes an executed
  copy to `build/`; does not overwrite the committed notebook. Allow 2–5 minutes.
  Output timing and solver diagnostics can differ across platforms; assertions
  have their original tolerances. Text output is not required to be byte equal.
- `make figures`: run every Python script in `web/tools/data/`, compare its JSON
  against the committed version, and restore the original bytes even on failure.
  Dictionary keys, array lengths, strings and booleans must match exactly;
  finite numbers use absolute tolerance 1e-6 or relative tolerance 1e-9: iterated fits differ by about 1e-8 between platforms; the page shows two decimals.
  This covers numerical solver/platform variation, not missing inputs.
  Allow 1–3 minutes. The four compact aggregate inputs are included; packaging
  them does not rerun their historical simulation campaigns. Static SVGs and
  data with no matching generator are shipped unchanged; TeX sources are
  retained for reference and are not part of `make figures`.
- `make study`: run all signal-attribution tests plus the two-seed base smoke
  unit (20 assets, 300 periods). Allow 1–2 minutes.
- `make study-full`: separate 107-unit campaign, normally 20 seeds per
  configuration (485 seeds overall, with documented special designs in code).
  Historical measured sum of unit wall times was 6.06 hours under contention;
  allow hours, depending on CPU and workers. `WORKERS=8` selects concurrency.
- `make site`: install npm dependencies and compile Astro/MDX; usually under
  a minute after download. `make site-build` skips installation.
- `make serve`: serve the existing site on port 4321 until interrupted.
- `make all`: figures, notebook, fast study, site; allow 5–10 minutes initially.

`PYTHON`, `OUTPUT` (default `build`) and `WORKERS` are configurable Make variables.
The notebook is self-contained; the package tests exercise the independent
study implementation. Re-execution retains solver warnings and discrepancies.

## Source

The article is `web/src/content/sessions/attribution.mdx`; its authoring partials
are in `web/src/content/parts/attribution/`. This export preserves their text.
Comparison routes are omitted. `/citations/` is included: it contains public
bibliographic metadata and reading relationships only. Bibliography attachment
paths were removed; scientific reference metadata are retained.

`NOTICE.md` lists every adapted and third-party component with its licence.

## Reuse

The text, figures, infographics and presentation are under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) (`LICENSE-CONTENT`);
the code (simulation package, figure exporters, notebook, page scripts, embeds and presentation view) is under [the MIT licence](https://opensource.org/license/mit) (`LICENSE`).
All figures and diagrams are original; numbers taken from published papers are cited in their captions.
Quoted passages, cited results and bibliographic metadata belong to their authors and publishers. Adapted and third-party components keep their own licences, listed in the notice file, NOTICE.md.

The page adapts Thibaud Frere's [research article template](https://huggingface.co/spaces/tfrere/research-article-template) and David Louapre's [NLP before LLMs course page](https://huggingface.co/spaces/dlouapre/nlp-before-llms),
both under CC BY 4.0; the upstream notice is kept in `web/LICENSE`.

## Citation

To cite this work:

> Sérié, E. (2026). Explaining a trading P&L: what finance knows about performance attribution, and what is open. https://attribution.serie-research.dev/attribution/

```bibtex
@misc{serie2026attribution,
  author = {S{\'e}ri{\'e}, Emmanuel},
  title = {Explaining a trading {P\&L}: what finance knows about performance attribution, and what is open},
  year = {2026},
  month = nov,
  url = {https://attribution.serie-research.dev/attribution/},
  note = {Code: https://github.com/serie-research/pnl-attribution}
}
```
