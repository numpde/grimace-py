SHELL := bash
.SHELLFLAGS := -eu -o pipefail -c
.DEFAULT_GOAL := help

COMPOSE_DIR ?= compose
DOCKER_COMPOSE ?= docker compose
override ACTUAL_UID := $(shell id -u)
override ACTUAL_GID := $(shell id -g)
override LOCAL_UID := $(ACTUAL_UID)
override LOCAL_GID := $(ACTUAL_GID)
override REPO_ROOT := $(shell pwd -P)
override PERF_ARTIFACT_FILES := docs/timings.tsv docs/timings.md notes/004_perf_history.jsonl
override PERF_ARTIFACT_DIRS := docs/timing-plots
override PERF_ARTIFACTS := $(PERF_ARTIFACT_FILES) $(PERF_ARTIFACT_DIRS)
override DOCS_SOURCE_DIR := docs
override DOCS_OUTPUT_DIR := build/docs-site
DOCS_PORT ?= 8000

NON_ROOT_GUARD := if [[ ! "$(ACTUAL_UID)" =~ ^[1-9][0-9]*$$ || ! "$(ACTUAL_GID)" =~ ^[1-9][0-9]*$$ ]]; then printf '%s\n' 'Refusing to run Docker lanes as root. Run make as a non-root user with positive numeric UID and GID.' >&2; exit 2; fi
DIST_GUARD := if [[ -L dist ]]; then printf '%s\n' 'Refusing to use dist because it is a symlink.' >&2; exit 2; fi
PERF_ARTIFACTS_GUARD := repo_root="$(REPO_ROOT)"; for path in $(PERF_ARTIFACT_FILES); do resolved="$$(realpath -e -- "$$path" 2>/dev/null || true)"; expected="$$repo_root/$$path"; if [[ ! -f "$$path" || "$$resolved" != "$$expected" ]]; then printf 'Refusing to bind perf artifact %s because it is missing, a symlink, or outside the repository.\n' "$$path" >&2; exit 2; fi; done; for path in $(PERF_ARTIFACT_DIRS); do resolved="$$(realpath -e -- "$$path" 2>/dev/null || true)"; expected="$$repo_root/$$path"; if [[ ! -d "$$path" || "$$resolved" != "$$expected" ]]; then printf 'Refusing to bind perf artifact directory %s because it is missing, a symlink, or outside the repository.\n' "$$path" >&2; exit 2; fi; done
DOCS_ARTIFACTS_GUARD := repo_root="$(REPO_ROOT)"; for path in $(DOCS_SOURCE_DIR) $(DOCS_OUTPUT_DIR); do resolved="$$(realpath -e -- "$$path" 2>/dev/null || true)"; expected="$$repo_root/$$path"; if [[ ! -d "$$path" || "$$resolved" != "$$expected" ]]; then printf 'Refusing to bind docs path %s because it is missing, a symlink, or outside the repository.\n' "$$path" >&2; exit 2; fi; done
COMPOSE_ENV := LOCAL_UID=$(LOCAL_UID) LOCAL_GID=$(LOCAL_GID) DOCS_PORT=$(DOCS_PORT)

define compose_run
@$(NON_ROOT_GUARD); \
$(COMPOSE_ENV) $(DOCKER_COMPOSE) -f $(COMPOSE_DIR)/$(1) run --build --rm $(2)
endef

.PHONY: help checks rust test parity exact-public-invariants package perf docs docs-serve ci

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
	  '  make docs     Build the documentation site under build/docs-site/' \
	  '  make docs-serve  Serve the documentation site at 127.0.0.1:$${DOCS_PORT:-8000}' \
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

docs:
	@$(NON_ROOT_GUARD)
	@mkdir -p $(DOCS_OUTPUT_DIR)
	@$(DOCS_ARTIFACTS_GUARD); \
	find $(DOCS_OUTPUT_DIR) -mindepth 1 -maxdepth 1 -exec rm -rf -- {} +; \
	$(COMPOSE_ENV) $(DOCKER_COMPOSE) -f $(COMPOSE_DIR)/docs.yml run --rm docs

docs-serve: docs
	@$(NON_ROOT_GUARD); \
	$(DOCS_ARTIFACTS_GUARD); \
	$(COMPOSE_ENV) $(DOCKER_COMPOSE) -f $(COMPOSE_DIR)/docs.yml run --rm --service-ports docs-serve

ci: checks rust test parity exact-public-invariants
