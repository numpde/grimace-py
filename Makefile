SHELL := bash
.SHELLFLAGS := -eu -o pipefail -c
.DEFAULT_GOAL := help

COMPOSE_DIR ?= compose
DOCKER_COMPOSE ?= docker compose
override ACTUAL_UID := $(shell id -u)
LOCAL_UID ?= $(shell id -u)
LOCAL_GID ?= $(shell id -g)
override REPO_ROOT := $(shell pwd -P)
override PERF_ARTIFACTS := docs/timings.tsv docs/timings.md notes/004_perf_history.jsonl

NON_ROOT_GUARD := if [[ "$(ACTUAL_UID)" == "0" || "$(LOCAL_UID)" == "0" || "$(LOCAL_GID)" == "0" ]]; then printf '%s\n' 'Refusing to run Docker lanes as root. Run make as a non-root user and do not set LOCAL_UID=0 or LOCAL_GID=0.' >&2; exit 2; fi
DIST_GUARD := if [[ -L dist ]]; then printf '%s\n' 'Refusing to use dist because it is a symlink.' >&2; exit 2; fi
PERF_ARTIFACTS_GUARD := repo_root="$(REPO_ROOT)"; for path in $(PERF_ARTIFACTS); do resolved="$$(realpath -e -- "$$path" 2>/dev/null || true)"; expected="$$repo_root/$$path"; if [[ ! -f "$$path" || "$$resolved" != "$$expected" ]]; then printf 'Refusing to bind perf artifact %s because it is missing, a symlink, or outside the repository.\n' "$$path" >&2; exit 2; fi; done
COMPOSE_ENV := LOCAL_UID=$(LOCAL_UID) LOCAL_GID=$(LOCAL_GID)

define compose_run
@$(NON_ROOT_GUARD); \
$(COMPOSE_ENV) $(DOCKER_COMPOSE) -f $(COMPOSE_DIR)/$(1) run --build --rm $(2)
endef

.PHONY: help checks rust test parity exact-public-invariants package perf ci

help:
	@printf '%s\n' \
	  'Supported lanes:' \
	  '  make checks  Run offline repository/source checks' \
	  '  make rust    Run Rust unit tests in the copied-context test image' \
	  '  make test    Run installed-package correctness in the test image' \
	  '  make parity  Run pinned RDKit parity in the test image' \
	  '  make exact-public-invariants  Run exact public invariant tests' \
	  '  make package  Build and validate wheel/sdist artifacts under dist/' \
	  '  make perf     Update opt-in timing docs and timing history' \
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

package:
	@$(NON_ROOT_GUARD)
	@$(DIST_GUARD)
	@mkdir -p dist
	@$(DIST_GUARD)
	@find dist -mindepth 1 -maxdepth 1 -exec rm -rf -- {} +
	$(call compose_run,package.yml,package)

perf:
	@$(NON_ROOT_GUARD); \
	$(PERF_ARTIFACTS_GUARD); \
	GRIMACE_PERF_GIT_COMMIT="$$(git rev-parse --short=12 HEAD)"; \
	GRIMACE_PERF_GIT_CHANGE="$$(git log -1 --format=%s HEAD)"; \
	if [[ -n "$$(git status --short)" ]]; then \
	  GRIMACE_PERF_GIT_DIRTY=1; \
	else \
	  GRIMACE_PERF_GIT_DIRTY=0; \
	fi; \
	export GRIMACE_PERF_GIT_COMMIT GRIMACE_PERF_GIT_CHANGE GRIMACE_PERF_GIT_DIRTY; \
	$(COMPOSE_ENV) $(DOCKER_COMPOSE) -f $(COMPOSE_DIR)/perf.yml run --build --rm perf

ci: checks rust test parity exact-public-invariants
