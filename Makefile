PYTHON ?= python
OUTPUT ?= build
WORKERS ?= 8
export OPENBLAS_NUM_THREADS = 1
export OMP_NUM_THREADS = 1
export VECLIB_MAXIMUM_THREADS = 1
export MKL_NUM_THREADS = 1
.PHONY: all notebook figures study study-full site site-build serve
all: figures notebook study site
notebook:
	$(PYTHON) tools/reproduce.py notebook --out $(OUTPUT)
figures:
	$(PYTHON) tools/reproduce.py figures --out $(OUTPUT)
study:
	$(PYTHON) -m pytest -q sim/signal_attribution/tests
	$(PYTHON) -m sim.cli run signal_attribution smoke-base-b00 --out $(OUTPUT)/study/smoke-base-b00.parquet
study-full:
	$(PYTHON) -m sim.signal_attribution.run_local full $(OUTPUT)/study-full --workers $(WORKERS)
site:
	@if [ -z "$$DEPS_PREINSTALLED" ]; then cd web && npm ci --no-audit --no-fund; fi
	$(MAKE) site-build
site-build:
	cd web && npm run build
serve:
	$(PYTHON) -m http.server 4321 --bind 0.0.0.0 --directory web/dist
