PYTHON ?= python
MEMCAP ?=
CHROME ?= /Applications/Google Chrome.app/Contents/MacOS/Google Chrome
CLI = $(PYTHON) -m driftwatch.cli compare data/demo/reference.parquet
LABELS = --target MedHouseVal --prediction prediction

# Single-threaded BLAS/OpenMP: results are deterministic and the run stays light on RAM/CPU.
export OMP_NUM_THREADS ?= 1
export OPENBLAS_NUM_THREADS ?= 1
export MKL_NUM_THREADS ?= 1

.PHONY: install install-locked data run test lint screenshot clean

install:
	$(PYTHON) -m pip install -e ".[dev]"

# Exact package versions that produced the committed reports/ (see requirements.lock).
install-locked:
	$(PYTHON) -m pip install -r requirements.lock
	$(PYTHON) -m pip install -e . --no-deps

# Download California Housing (~0.4 MB) into data/ once; later runs use the cache.
data:
	$(PYTHON) -c "from driftwatch.data import load_california; print(load_california('data').shape)"

# Full reproduction: study -> reports/metrics.json + reports/figures/*.png, then the CLI
# gates on a real shift (must exit 1) and on same-region held-out data (must exit 0).
run: data
	$(MEMCAP) $(PYTHON) -m driftwatch.experiments --out reports --data data
	$(CLI) data/demo/north.parquet $(LABELS) --out reports/sample_report.html \
		--json reports/sample_report.json \
		--title "California Housing: Northern/Central vs Southern California"; test $$? -eq 1
	$(CLI) data/demo/holdout.parquet $(LABELS) --quiet

test:
	$(PYTHON) -m pytest -q

lint:
	$(PYTHON) -m ruff check .

# Optional: PNG screenshot of the sample report for the README (needs Google Chrome).
screenshot:
	"$(CHROME)" --headless=new --disable-gpu --hide-scrollbars --force-device-scale-factor=1 \
		--user-data-dir="$(CURDIR)/build/chrome-profile" --window-size=1180,1480 \
		--screenshot="$(CURDIR)/reports/sample_report.png" "file://$(CURDIR)/reports/sample_report.html"

clean:
	rm -rf .pytest_cache .ruff_cache build dist src/*.egg-info
