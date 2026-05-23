SHELL := bash
.SHELLFLAGS := -eu -o pipefail -c
.DEFAULT_GOAL := help

COMPOSE_DIR ?= compose
DOCKER_COMPOSE ?= docker compose
ACTUAL_UID := $(shell id -u)

NON_ROOT_GUARD := if [[ "$(ACTUAL_UID)" == "0" ]]; then printf '%s\n' 'Refusing to run Docker lanes as root. Run make as a non-root user.' >&2; exit 2; fi

define compose_run
@$(NON_ROOT_GUARD); \
$(DOCKER_COMPOSE) -f $(COMPOSE_DIR)/$(1) run --build --rm $(2)
endef

.PHONY: help checks rust test parity exact-public-invariants ci

help:
	@printf '%s\n' \
	  'Supported lanes:' \
	  '  make checks  Run offline repository/source checks' \
	  '  make rust    Run Rust unit tests in the copied-context test image' \
	  '  make test    Run installed-package correctness in the test image' \
	  '  make parity  Run pinned RDKit parity in the test image' \
	  '  make exact-public-invariants  Run exact public invariant tests' \
	  '  make ci      Run checks, rust, test, parity, and exact invariants' \
	  '' \
	  'Docker-backed lanes refuse root execution and use strict Compose posture.'

checks:
	$(call compose_run,checks.yml,checks)

rust:
	$(call compose_run,test.yml,rust)

test:
	$(call compose_run,test.yml,test)

parity:
	$(call compose_run,test.yml,parity)

exact-public-invariants:
	$(call compose_run,test.yml,exact-public-invariants)

ci: checks rust test parity exact-public-invariants
