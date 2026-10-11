.PHONY: rebuild-south-star1 test-south-star1-fast test-south-star1-slow-one test-south-star1-slow-shard test-south-star1-slow test-south-star1-support-artifact-one test-south-star1-support-artifact qualify-south-star1 slow-south-star1 test-bounded-runner

PYTHON ?= python3
MATURIN ?= $(dir $(PYTHON))maturin
BOUNDED_RUNNER ?= python3 scripts/run_bounded_tests.py
SLOW_ASSET_ROOT := .south-star1-qualification/$(shell git rev-parse HEAD)

rebuild-south-star1:
	$(BOUNDED_RUNNER) --wall-seconds 300 --rss-mib 4096 --cpu-seconds 300 --address-space-mib 8192 -- $(MATURIN) develop --release --skip-install

test-south-star1-fast:
	$(BOUNDED_RUNNER) --wall-seconds 120 --rss-mib 2048 --cpu-seconds 120 --address-space-mib 4096 -- env PYTHONPATH=python:. $(PYTHON) -m unittest tests.run_south_star_semantics -q

test-south-star1-slow-one:
	@test -n "$(SLOW_SHARD)" || (echo "SLOW_SHARD is required" >&2; exit 2)
	@test -n "$(SLOW_LAYER)" || (echo "SLOW_LAYER is required" >&2; exit 2)
	$(BOUNDED_RUNNER) --wall-seconds 300 --rss-mib 4096 --cpu-seconds 300 --address-space-mib 8192 -- env SOUTH_STAR1_RUN_SLOW=1 SOUTH_STAR1_SLOW_SHARD=$(SLOW_SHARD) SOUTH_STAR1_SLOW_LAYER=$(SLOW_LAYER) SOUTH_STAR1_SLOW_ASSET_ROOT=$(SLOW_ASSET_ROOT) PYTHONPATH=python:. $(PYTHON) -m tests.run_south_star1_slow

test-south-star1-slow-shard:
	@test -n "$(SLOW_SHARD)" || (echo "SLOW_SHARD is required" >&2; exit 2)
	$(BOUNDED_RUNNER) --wall-seconds 300 --rss-mib 4096 --cpu-seconds 300 --address-space-mib 8192 -- env SOUTH_STAR1_RUN_SLOW=1 SOUTH_STAR1_SLOW_SHARD=$(SLOW_SHARD) SOUTH_STAR1_SLOW_ASSET_ROOT=$(SLOW_ASSET_ROOT) PYTHONPATH=python:. $(PYTHON) -m tests.run_south_star1_slow --run-shard $(SLOW_SHARD)

test-south-star1-slow:
	$(BOUNDED_RUNNER) --wall-seconds 300 --rss-mib 4096 --cpu-seconds 300 --address-space-mib 8192 -- env SOUTH_STAR1_RUN_SLOW=1 SOUTH_STAR1_SLOW_ASSET_ROOT=$(SLOW_ASSET_ROOT) PYTHONPATH=python:. $(PYTHON) -m tests.run_south_star1_slow --run-all-product

test-south-star1-support-artifact-one:
	$(BOUNDED_RUNNER) --wall-seconds 120 --rss-mib 2048 --cpu-seconds 120 --address-space-mib 4096 -- env PYTHONPATH=python:. $(PYTHON) -m tests.run_writer_support_artifact_tests --domain $(SUPPORT_ARTIFACT_DOMAIN)

test-south-star1-support-artifact:
	$(BOUNDED_RUNNER) --wall-seconds 300 --rss-mib 4096 --cpu-seconds 300 --address-space-mib 8192 -- env PYTHONPATH=python:. $(PYTHON) -m tests.run_writer_support_artifact_tests --all

qualify-south-star1:
	$(MAKE) rebuild-south-star1 PYTHON="$(PYTHON)"
	$(MAKE) test-south-star1-fast PYTHON="$(PYTHON)"
	$(MAKE) test-south-star1-slow PYTHON="$(PYTHON)"

slow-south-star1: test-south-star1-slow

test-bounded-runner:
	/usr/bin/timeout --foreground --signal=TERM --kill-after=5s 60s bash -c 'ulimit -t 60; ulimit -v 1048576; exec $(PYTHON) -m unittest tests.bounded_runner_selftest -v'
