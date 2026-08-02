.PHONY: rebuild-south-star1 test-south-star1-fast test-south-star1-slow-one test-south-star1-slow-shard test-south-star1-slow qualify-south-star1 slow-south-star1

PYTHON ?= python3
MATURIN ?= $(dir $(PYTHON))maturin
SLOW_SHARDS := zero-h-adjacent remote-a remote-b
SLOW_DIAGNOSTIC_LAYERS := count-dag-build count-dag-validate support-artifact-build support-artifact-live offline-complete
SLOW_LAYERS_REMOTE := public-build public-certify public-runtime public-recertification public-proofs-0 public-proofs-1 public-proofs-2 public-proofs-3 support-reparse continuation stereo-audit
SLOW_LAYERS_ZERO_H := public-build public-certify public-runtime public-recertification offline-zero-h offline-adjacent support-zero-h support-adjacent support-reparse continuation stereo-audit
SLOW_ASSET_ROOT := .south-star1-qualification/$(shell git rev-parse HEAD)

rebuild-south-star1:
	$(MATURIN) develop --release --skip-install

test-south-star1-fast:
	PYTHONPATH=python:. $(PYTHON) -m unittest tests.run_south_star_semantics -q

test-south-star1-slow-one:
	@test -n "$(SLOW_SHARD)" || (echo "SLOW_SHARD is required" >&2; exit 2)
	@test -n "$(SLOW_LAYER)" || (echo "SLOW_LAYER is required" >&2; exit 2)
	@echo "SOUTH_STAR1_RUN_SLOW=1 SOUTH_STAR1_SLOW_SHARD=$(SLOW_SHARD) SOUTH_STAR1_SLOW_LAYER=$(SLOW_LAYER) SOUTH_STAR1_SLOW_ASSET_ROOT=$(SLOW_ASSET_ROOT) PYTHONPATH=python:. $(PYTHON) -m tests.run_south_star1_slow"
	SOUTH_STAR1_RUN_SLOW=1 SOUTH_STAR1_SLOW_SHARD=$(SLOW_SHARD) SOUTH_STAR1_SLOW_LAYER=$(SLOW_LAYER) SOUTH_STAR1_SLOW_ASSET_ROOT=$(SLOW_ASSET_ROOT) PYTHONPATH=python:. $(PYTHON) -m tests.run_south_star1_slow

test-south-star1-slow-shard:
	@test -n "$(SLOW_SHARD)" || (echo "SLOW_SHARD is required" >&2; exit 2)
	@for layer in $(if $(filter zero-h-adjacent,$(SLOW_SHARD)),$(SLOW_LAYERS_ZERO_H),$(SLOW_LAYERS_REMOTE)); do \
		$(MAKE) --no-print-directory test-south-star1-slow-one PYTHON="$(PYTHON)" SLOW_SHARD="$(SLOW_SHARD)" SLOW_LAYER="$$layer"; \
	done

test-south-star1-slow:
	@for shard in $(SLOW_SHARDS); do \
		$(MAKE) --no-print-directory test-south-star1-slow-shard PYTHON="$(PYTHON)" SLOW_SHARD="$$shard"; \
	done

qualify-south-star1:
	$(MAKE) rebuild-south-star1 PYTHON="$(PYTHON)"
	$(MAKE) test-south-star1-fast PYTHON="$(PYTHON)"
	$(MAKE) test-south-star1-slow PYTHON="$(PYTHON)"

slow-south-star1: test-south-star1-slow
