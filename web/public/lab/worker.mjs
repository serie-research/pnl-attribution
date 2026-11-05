const indexURL = 'https://cdn.jsdelivr.net/pyodide/v0.29.5/full/';
let python;
let activeCell = null;
self.onmessage = async ({ data }) => {
  try {
    if (data.type === 'init') {
      const start = performance.now();
      const { loadPyodide } = await import(indexURL + 'pyodide.mjs');
      python = await loadPyodide({ indexURL,
        stdout: text => postMessage({ type: 'stdout', index: activeCell, text }),
        stderr: text => postMessage({ type: 'stderr', index: activeCell, text }),
      });
      await python.loadPackage(['numpy', 'scipy', 'cvxpy-base', 'clarabel']);
      const versions = python.runPython(`import json, sys, numpy, scipy, cvxpy, clarabel
json.dumps(dict(python=sys.version, numpy=numpy.__version__, scipy=scipy.__version__, cvxpy=cvxpy.__version__, clarabel=clarabel.__version__, solvers=cvxpy.installed_solvers()))`);
      postMessage({ type: 'ready', versions: JSON.parse(versions), ms: performance.now() - start,
        wasmBytes: python._module.HEAP8.buffer.byteLength });
    } else if (data.type === 'run') {
      activeCell = data.index;
      const start = performance.now();
      let error = null;
      try {
        // exec matches notebook cell statement execution; no source transformations.
        python.globals.set('__lab_source', data.source);
        await python.runPythonAsync('exec(compile(__lab_source, "cell-' + data.index + '", "exec"), globals())');
      } catch (e) { error = String(e); }
      postMessage({ type: 'result', index: data.index, error, ms: performance.now() - start,
        wasmBytes: python._module.HEAP8.buffer.byteLength });
      activeCell = null;
    }
  } catch (e) { postMessage({ type: 'fatal', error: String(e) }); }
};
