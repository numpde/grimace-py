.PHONY: rebuild-south-star1 test-south-star1-fast test-south-star1-slow-one test-south-star1-slow-shard test-south-star1-slow test-south-star1-support-artifact-one test-south-star1-support-artifact qualify-south-star1 slow-south-star1

PYTHON ?= python3
MATURIN ?= $(dir $(PYTHON))maturin
SLOW_ASSET_ROOT := .south-star1-qualification/$(shell git rev-parse HEAD)

rebuild-south-star1:
	$(MATURIN) develop --release --skip-install

test-south-star1-fast:
	PYTHONPATH=python:. $(PYTHON) -m unittest tests.run_south_star_semantics -q

test-south-star1-slow-one:
	@test -n "$(SLOW_SHARD)" || (echo "SLOW_SHARD is required" >&2; exit 2)
	@test -n "$(SLOW_LAYER)" || (echo "SLOW_LAYER is required" >&2; exit 2)
	SOUTH_STAR1_RUN_SLOW=1 SOUTH_STAR1_SLOW_SHARD=$(SLOW_SHARD) SOUTH_STAR1_SLOW_LAYER=$(SLOW_LAYER) SOUTH_STAR1_SLOW_ASSET_ROOT=$(SLOW_ASSET_ROOT) PYTHONPATH=python:. $(PYTHON) -m tests.run_south_star1_slow

test-south-star1-slow-shard:
	@test -n "$(SLOW_SHARD)" || (echo "SLOW_SHARD is required" >&2; exit 2)
	SOUTH_STAR1_RUN_SLOW=1 SOUTH_STAR1_SLOW_SHARD=$(SLOW_SHARD) SOUTH_STAR1_SLOW_ASSET_ROOT=$(SLOW_ASSET_ROOT) PYTHONPATH=python:. $(PYTHON) -m tests.run_south_star1_slow --run-shard $(SLOW_SHARD)

test-south-star1-slow:
	SOUTH_STAR1_RUN_SLOW=1 SOUTH_STAR1_SLOW_ASSET_ROOT=$(SLOW_ASSET_ROOT) PYTHONPATH=python:. $(PYTHON) -m tests.run_south_star1_slow --run-all-product

test-south-star1-support-artifact-one:
	PYTHONPATH=python:. $(PYTHON) -m tests.run_writer_support_artifact_tests --domain $(SUPPORT_ARTIFACT_DOMAIN)

test-south-star1-support-artifact:
	PYTHONPATH=python:. $(PYTHON) -m tests.run_writer_support_artifact_tests --all

qualify-south-star1:
	$(MAKE) rebuild-south-star1 PYTHON="$(PYTHON)"
	$(MAKE) test-south-star1-fast PYTHON="$(PYTHON)"
	$(MAKE) test-south-star1-slow PYTHON="$(PYTHON)"

slow-south-star1: test-south-star1-slow
