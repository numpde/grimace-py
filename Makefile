.PHONY: rebuild-south-star1 test-south-star1-fast test-south-star1-slow-one test-south-star1-slow-shard test-south-star1-slow qualify-south-star1 slow-south-star1

PYTHON ?= python3
MATURIN ?= $(dir $(PYTHON))maturin
SLOW_SHARDS := zero-h-adjacent remote-a remote-b
SLOW_LAYERS := public-build public-recertification public-proofs offline-complete support-artifact support-reparse continuation stereo-audit
SLOW_LAYERS_ZERO_H := public-build public-recertification public-proofs offline-zero-h offline-adjacent support-artifact support-reparse continuation stereo-audit

rebuild-south-star1:
	$(MATURIN) develop --release --skip-install

test-south-star1-fast:
	PYTHONPATH=python:. $(PYTHON) -m unittest tests.run_south_star_semantics -q

test-south-star1-slow-one:
	@test -n "$(SLOW_SHARD)" || (echo "SLOW_SHARD is required" >&2; exit 2)
	@test -n "$(SLOW_LAYER)" || (echo "SLOW_LAYER is required" >&2; exit 2)
	@echo "SOUTH_STAR1_RUN_SLOW=1 SOUTH_STAR1_SLOW_SHARD=$(SLOW_SHARD) SOUTH_STAR1_SLOW_LAYER=$(SLOW_LAYER) PYTHONPATH=python:. $(PYTHON) -m tests.run_south_star1_slow"
	SOUTH_STAR1_RUN_SLOW=1 SOUTH_STAR1_SLOW_SHARD=$(SLOW_SHARD) SOUTH_STAR1_SLOW_LAYER=$(SLOW_LAYER) PYTHONPATH=python:. $(PYTHON) -m tests.run_south_star1_slow

test-south-star1-slow-shard:
	@test -n "$(SLOW_SHARD)" || (echo "SLOW_SHARD is required" >&2; exit 2)
	@for layer in $(if $(filter zero-h-adjacent,$(SLOW_SHARD)),$(SLOW_LAYERS_ZERO_H),$(SLOW_LAYERS)); do \
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
