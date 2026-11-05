"""Offline execution checks; committed references are never updated."""
import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def compare(expected, actual, locus='$'):
    if isinstance(expected, bool) or expected is None or isinstance(expected, str):
        assert type(expected) is type(actual) and expected == actual, locus
    elif isinstance(expected, (int, float)):
        assert isinstance(actual, (int, float)) and not isinstance(actual, bool), locus
        assert math.isfinite(expected) and math.isfinite(actual), locus
        assert math.isclose(expected, actual, abs_tol=1e-6, rel_tol=1e-9), (locus, expected, actual)
    elif isinstance(expected, list):
        assert isinstance(actual, list) and len(expected) == len(actual), locus
        for i, (a, b) in enumerate(zip(expected, actual)):
            compare(a, b, f'{locus}[{i}]')
    else:
        assert isinstance(actual, dict) and expected.keys() == actual.keys(), locus
        for key in expected:
            compare(expected[key], actual[key], f'{locus}.{key}')


def figures(out):
    data = ROOT / 'web/public/data'
    originals = {p: p.read_bytes() for p in data.glob('*.json')}
    receipts = []
    try:
        for script in sorted((ROOT / 'web/tools/data').glob('*.py')):
            target = data / (script.stem + '.json')
            assert target in originals, f'No committed reference for {script.name}'
            start = time.monotonic()
            subprocess.run([sys.executable, str(script)], cwd=ROOT, check=True)
            compare(json.loads(originals[target]), json.loads(target.read_bytes()))
            receipts.append(dict(script=script.name, seconds=time.monotonic()-start, passed=True))
        (out / 'figures-check.json').write_text(json.dumps(receipts, indent=2) + '\n')
    finally:
        for p, raw in originals.items():
            p.write_bytes(raw)


def notebook(out):
    import nbformat
    from nbclient import NotebookClient
    from jupyter_client.kernelspec import KernelSpec
    from jupyter_client import KernelManager
    source = ROOT / 'web/notebooks/signal-attribution.ipynb'
    book = nbformat.read(source, as_version=4)
    # Use this environment's interpreter, independently of user kernel installs.
    manager = KernelManager(kernel_name='python3')
    manager._kernel_spec = KernelSpec(argv=[sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}'], display_name='Python 3', language='python')
    for cell in book.cells:
        if cell.cell_type == 'code':
            cell.outputs = []
            cell.execution_count = None
    NotebookClient(book, km=manager, timeout=600, resources={'metadata': {'path': str(ROOT)}}).execute()
    code = [c for c in book.cells if c.cell_type == 'code']
    assert code and all(c.execution_count is not None for c in code)
    assert not any(o.output_type == 'error' for c in code for o in c.outputs)
    nbformat.write(book, out / 'signal-attribution.executed.ipynb')
    (out / 'notebook-check.json').write_text(json.dumps(dict(executed_cells=len(code), passed=True)) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('target', choices=['figures', 'notebook'])
    parser.add_argument('--out', type=Path, default=Path('build'))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    globals()[args.target](args.out)
    print(f'{args.target}: PASS in {time.monotonic()-start:.3f}s')


if __name__ == '__main__':
    main()
