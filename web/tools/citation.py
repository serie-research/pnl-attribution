"""Reuse and Citation text for the READMEs and CITATION.cff, from web/src/content/citation.json."""
import json
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1] / 'src/content/citation.json'


def load():
    return json.loads(SOURCE.read_text())


def markdown():
    c = load()
    content, code = c['reuse']['content'], c['reuse']['code']
    return f"""## Reuse

{content['scope'][0].upper() + content['scope'][1:]} are under [{content['licence']}]({content['url']}) (`LICENSE-CONTENT`);
{code['scope']} is under [{code['licence']}]({code['url']}) (`LICENSE`).
{c['reuse']['figures']}
{c['reuse']['excluded']}

The page adapts {' and '.join(f"{a['author']}'s [{a['work']}]({a['url']})" for a in c['reuse']['adapted'])},
both under CC BY 4.0; the upstream notice is kept in `web/LICENSE`.

## Citation

To cite this work:

> {c['text']}

```bibtex
{c['bibtex']}
```
"""


def cff():
    c = load()
    authors = '\n'.join(f'  - family-names: "{a["family"]}"\n    given-names: "{a["given"]}"' for a in c['authors'])
    return f"""cff-version: 1.2.0
message: "If you use this work, please cite it as below."
title: "{c['title']}"
authors:
{authors}
date-released: {c['date']}
url: "{c['url']}"
repository-code: "{c['repository']}"
license: MIT
type: software
"""
