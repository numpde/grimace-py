.PHONY: slow-south-star1

PYTHON ?= python3

slow-south-star1:
	SOUTH_STAR1_RUN_SLOW=1 PYTHONPATH=python:. $(PYTHON) -m unittest tests.run_south_star_semantics -q
