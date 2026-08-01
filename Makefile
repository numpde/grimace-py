.PHONY: rebuild-south-star1 test-south-star1-fast test-south-star1-slow qualify-south-star1 slow-south-star1

PYTHON ?= python3
MATURIN ?= $(dir $(PYTHON))maturin

rebuild-south-star1:
	$(MATURIN) develop --release --skip-install

test-south-star1-fast:
	PYTHONPATH=python:. $(PYTHON) -m unittest tests.run_south_star_semantics -q

test-south-star1-slow:
	SOUTH_STAR1_RUN_SLOW=1 PYTHONPATH=python:. $(PYTHON) -m unittest tests.run_south_star1_slow -q

qualify-south-star1:
	$(MAKE) rebuild-south-star1 PYTHON="$(PYTHON)"
	$(MAKE) test-south-star1-fast PYTHON="$(PYTHON)"
	$(MAKE) test-south-star1-slow PYTHON="$(PYTHON)"

slow-south-star1: test-south-star1-slow
