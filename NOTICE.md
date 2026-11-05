# Licences

Copyright (c) 2026 Emmanuel Sérié.

| What | Licence | File |
|---|---|---|
| Code: the simulation package (`sim/`), the figure exporters (`web/tools/`), the notebook, the page's scripts, components, layouts and embeds, including the presentation view (`/attribution/present/`) and its ink layer | MIT | `LICENSE` |
| Content: the article text, figures, infographics, notebook prose and the presentation view's content | Creative Commons Attribution 4.0 International (CC BY 4.0) | `LICENSE-CONTENT` |

To cite this work, see the Citation section of `README.md` or `CITATION.cff`.

Quoted passages, cited results and bibliographic metadata belong to their
authors and publishers; they are reproduced for comment and citation only.

## Adapted and third-party material

| Material | Origin | Licence |
|---|---|---|
| Design tokens `web/src/styles/_variables.css` (verbatim) | Thibaud Frere, [research article template](https://huggingface.co/spaces/tfrere/research-article-template) | CC BY 4.0; upstream notice kept in `web/LICENSE` |
| Components `Fold`, `Optional`, `PaperNote`, `ReferenceList`, `WhatToNotice` and `papers.ts` in `web/src/components/post/` (verbatim or lightly edited) | David Louapre, [NLP before LLMs](https://huggingface.co/spaces/dlouapre/nlp-before-llms), itself built on the template above | CC BY 4.0 |
| D3 | Mike Bostock | ISC |
| KaTeX | Khan Academy and contributors | MIT |
| Rough.js 4.6.6 (`web/public/vendor/`) | Preet Shihn | MIT, `rough-4.6.6.LICENSE.txt` |
| Patrick Hand font (`web/public/vendor/`) | Patrick Wagesreiter, distributed by Google Fonts | SIL Open Font License 1.1, `patrick-hand-OFL.txt` |
| Lab runtime, loaded at run time: Pyodide; NumPy and SciPy; CVXPY and Clarabel | their projects | MPL-2.0; BSD-3-Clause; Apache-2.0 |
| npm and Python dependencies | their projects | their own licences, as declared in `web/package-lock.json` and `uv.lock` |

These files remain under CC BY 4.0; this repository's changes to them are
offered under the same terms. The layouts, the other components, the styles
and the embeds were written for this page (compared on 2026-11-02 with both
sources at their latest commits, c75fc29 and b44980b).
