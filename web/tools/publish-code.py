"""Publish an explicit source allowlist; check bytes and confidentiality before release."""
import argparse
import io
import json
from pathlib import Path
import re
import shutil
import sys
import zipfile

import citation  # web/tools/citation.py

WEB = Path(__file__).resolve().parents[1]
ROOT = WEB.parent
# Redactions and restricted patterns live in publication-guard.json, which the
# standalone export omits: that repository is scanned when it is exported.
_GUARD = WEB / 'tools' / 'publication-guard.json'
if _GUARD.exists():
    _guard = json.loads(_GUARD.read_text())
else:
    # Only the standalone export lacks the guard; say so instead of passing silently.
    print('publish-code: no publication-guard.json, confidentiality scan skipped', file=sys.stderr)
    _guard = {'redactions': {}, 'patterns': {}}
REDACTIONS = {name: [tuple(pair) for pair in pairs] for name, pairs in _guard['redactions'].items()}
PATTERNS = _guard['patterns']

def scan(name, data):
    text = data.decode('utf8')
    for label, pattern in PATTERNS.items():
        match = re.search(pattern, text, re.I | re.M)
        if match:
            raise ValueError(f'Confidentiality scan: {name}, line {text[:match.start()].count(chr(10)) + 1}, rule {label}')


def scripts(html):
    return '\n'.join(re.findall(r'<script\b[^>]*>([\s\S]*?)</script>', html)).encode()


def sources():
    files = {}
    paths = [*sorted((WEB / 'tools/data').glob('*.py')),
             *sorted((ROOT / 'sim/signal_attribution').rglob('*.py')),
             ROOT / 'sim/__init__.py', ROOT / 'sim/cli.py',
             ROOT / 'sim/common/__init__.py', ROOT / 'sim/common/io.py',
             ROOT / 'pyproject.toml',
             *[ROOT / p for p in [
                 'sim-figures/signal_attribution/full/article-summary.json',
                 'sim-figures/indicator_importance/full/article-summary.json']]]
    # In the standalone export (no guard file), files it does not ship are skipped.
    paths = [path for path in paths if _GUARD.exists() or path.exists()]
    for path in paths:
        name = path.relative_to(ROOT).as_posix()
        data = path.read_bytes()
        for old, new in REDACTIONS.get(name, []):
            assert data.count(old.encode()) == 1, f'Redaction source changed: {name}'
            data = data.replace(old.encode(), new.encode())
        files[name] = data
    # Include all article embeds and their data, but not the template demo.
    for path in sorted((WEB / 'src/content/embeds').glob('*.html')):
        if path.name == 'demo.html':
            continue
        files[f'embeds/{path.stem}.js'] = scripts(path.read_text())
        files[f'embeds/{path.name}'] = path.read_bytes()
    for path in sorted((WEB / 'public/data').glob('*.json')):
        if path.name == 'demo.json':
            continue
        files[f'web/public/data/{path.name}'] = path.read_bytes()
    files['README.md'] = (WEB / 'tools/public-code-README.md').read_bytes() + b'\n' + citation.markdown().encode()
    # NOTICE.md points to CITATION.cff; ship it as the exported repository does.
    files['CITATION.cff'] = citation.cff().encode()
    # Licence files: licenses/ in the source repository, the root of the standalone export.
    for name in ('LICENSE', 'LICENSE-CONTENT', 'NOTICE.md'):
        source = ROOT / 'licenses' / name
        files[name] = (source if source.exists() else ROOT / name).read_bytes()
    for name, data in files.items():
        scan(name, data)
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as bundle:
        for name, data in sorted(files.items()):
            entry = zipfile.ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_DEFLATED
            entry.external_attr = 0o644 << 16
            bundle.writestr(entry, data)
    files['reproduce.zip'] = archive.getvalue()
    return files


def verify(directory):
    expected = sources()
    code = directory / 'code'
    actual = {p.relative_to(code).as_posix(): p.read_bytes() for p in code.rglob('*') if p.is_file()}
    assert actual == expected, 'Published code differs from the source allowlist (including archive)'
    for path in (directory / 'data').rglob('*'):
        if path.is_file():
            scan(str(path.relative_to(directory)), path.read_bytes())
            assert path.read_bytes() == (WEB / 'public' / path.relative_to(directory)).read_bytes(), f'Data differs: {path.name}'
    expected_data = {p.relative_to(WEB / 'public/data').as_posix(): p.read_bytes() for p in (WEB / 'public/data').rglob('*') if p.is_file()}
    actual_data = {p.relative_to(directory / 'data').as_posix(): p.read_bytes() for p in (directory / 'data').rglob('*') if p.is_file()}
    assert actual_data == expected_data, 'Published data set differs from public source data'
    with zipfile.ZipFile(io.BytesIO(actual['reproduce.zip'])) as bundle:
        for name in bundle.namelist():
            scan(name, bundle.read(name))
    print(f'Publication verified: {len(actual)} code files; data and archive scanned; source bytes match.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', type=Path)
    args = parser.parse_args()
    if args.check:
        verify(args.check.resolve())
        return
    expected = sources()  # Scan before writing any file.
    code = WEB / 'public/code'
    if code.exists():
        shutil.rmtree(code)
    for name, data in expected.items():
        target = code / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    verify(WEB / 'public')


if __name__ == '__main__':
    main()
