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
override TIMINGS_ENUM_ARTIFACT_FILES := docs/timings-enum.tsv docs/timings-enum.md notes/004_perf_history.jsonl
override TIMINGS_ENUM_ARTIFACT_DIRS := docs/timings-enum-plots
override TIMINGS_ENUM_ARTIFACTS := $(TIMINGS_ENUM_ARTIFACT_FILES) $(TIMINGS_ENUM_ARTIFACT_DIRS)
override DOCS_SOURCE_DIR := docs
override DOCS_OUTPUT_DIR := build/docs-site
override PREPARED_MOL_ZSTD_PACKAGE_DATA_DIR := python/grimace/data/prepared_mol_zstd
override TIMINGS_PREPARED_MOL_ZSTD_ARTIFACT_FILES := docs/timings-prepared-mol-zstd.tsv
override TIMINGS_PREPARED_MOL_ZSTD_ARTIFACT_DIRS := docs/timings-prepared-mol-zstd-plots
override TIMINGS_PREPARED_MOL_ZSTD_ARTIFACTS := $(TIMINGS_PREPARED_MOL_ZSTD_ARTIFACT_FILES) $(TIMINGS_PREPARED_MOL_ZSTD_ARTIFACT_DIRS)
DOCS_PORT ?= 8000
PREPARED_MOL_ZSTD_CREATED_DATE ?=
PREPARED_MOL_ZSTD_FORCE ?= 0

NON_ROOT_GUARD := if [[ ! "$(ACTUAL_UID)" =~ ^[1-9][0-9]*$$ || ! "$(ACTUAL_GID)" =~ ^[1-9][0-9]*$$ ]]; then printf '%s\n' 'Refusing to run Docker lanes as root. Run make as a non-root user with positive numeric UID and GID.' >&2; exit 2; fi
DOCS_PORT_GUARD := if [[ ! "$${DOCS_PORT}" =~ ^[1-9][0-9]{0,4}$$ || "$${DOCS_PORT}" -gt 65535 ]]; then printf '%s\n' 'Refusing to run docs lane with DOCS_PORT outside 1..65535.' >&2; exit 2; fi
DIST_GUARD := if [[ -L dist ]]; then printf '%s\n' 'Refusing to use dist because it is a symlink.' >&2; exit 2; fi
TIMINGS_ENUM_ARTIFACTS_GUARD := repo_root="$(REPO_ROOT)"; for path in $(TIMINGS_ENUM_ARTIFACT_FILES); do resolved="$$(realpath -e -- "$$path" 2>/dev/null || true)"; expected="$$repo_root/$$path"; if [[ ! -f "$$path" || "$$resolved" != "$$expected" ]]; then printf 'Refusing to bind enum timing artifact %s because it is missing, a symlink, or outside the repository.\n' "$$path" >&2; exit 2; fi; done; for path in $(TIMINGS_ENUM_ARTIFACT_DIRS); do resolved="$$(realpath -e -- "$$path" 2>/dev/null || true)"; expected="$$repo_root/$$path"; if [[ ! -d "$$path" || "$$resolved" != "$$expected" ]]; then printf 'Refusing to bind enum timing artifact directory %s because it is missing, a symlink, or outside the repository.\n' "$$path" >&2; exit 2; fi; done
TIMINGS_PREPARED_MOL_ZSTD_ARTIFACTS_GUARD := repo_root="$(REPO_ROOT)"; for path in $(TIMINGS_PREPARED_MOL_ZSTD_ARTIFACT_FILES); do resolved="$$(realpath -e -- "$$path" 2>/dev/null || true)"; expected="$$repo_root/$$path"; if [[ ! -f "$$path" || "$$resolved" != "$$expected" ]]; then printf 'Refusing to bind PreparedMol zstd timing artifact %s because it is missing, a symlink, or outside the repository.\n' "$$path" >&2; exit 2; fi; done; for path in $(TIMINGS_PREPARED_MOL_ZSTD_ARTIFACT_DIRS); do resolved="$$(realpath -e -- "$$path" 2>/dev/null || true)"; expected="$$repo_root/$$path"; if [[ ! -d "$$path" || "$$resolved" != "$$expected" ]]; then printf 'Refusing to bind PreparedMol zstd timing artifact directory %s because it is missing, a symlink, or outside the repository.\n' "$$path" >&2; exit 2; fi; done
DOCS_ARTIFACTS_GUARD := repo_root="$(REPO_ROOT)"; for path in $(DOCS_SOURCE_DIR) $(DOCS_OUTPUT_DIR); do resolved="$$(realpath -e -- "$$path" 2>/dev/null || true)"; expected="$$repo_root/$$path"; if [[ ! -d "$$path" || "$$resolved" != "$$expected" ]]; then printf 'Refusing to bind docs path %s because it is missing, a symlink, or outside the repository.\n' "$$path" >&2; exit 2; fi; done
TIMING_GIT_METADATA_ENV := GRIMACE_PERF_GIT_COMMIT="$$(git rev-parse --short=12 HEAD)"; GRIMACE_PERF_GIT_CHANGE="$$(git log -1 --format=%s HEAD)"; if [[ -n "$$(git status --short)" ]]; then GRIMACE_PERF_GIT_DIRTY=1; else GRIMACE_PERF_GIT_DIRTY=0; fi; export GRIMACE_PERF_GIT_COMMIT GRIMACE_PERF_GIT_CHANGE GRIMACE_PERF_GIT_DIRTY
COMPOSE_ENV := LOCAL_UID=$(LOCAL_UID) LOCAL_GID=$(LOCAL_GID)

define compose_run
@$(NON_ROOT_GUARD); \
$(COMPOSE_ENV) $(DOCKER_COMPOSE) -f $(COMPOSE_DIR)/$(1) run --build --rm $(2)
endef

.PHONY: help checks rust test parity exact-public-invariants package timings-enum prepared-mol-zstd-dictionary timings-prepared-mol-zstd docs docs-serve ci

docs docs-serve: export DOCS_PORT := $(value DOCS_PORT)
prepared-mol-zstd-dictionary: export PREPARED_MOL_ZSTD_CREATED_DATE := $(value PREPARED_MOL_ZSTD_CREATED_DATE)
prepared-mol-zstd-dictionary: export PREPARED_MOL_ZSTD_FORCE := $(value PREPARED_MOL_ZSTD_FORCE)

help:
	@printf '%s\n' \
	  'Supported lanes:' \
	  '  make checks  Run offline repository/source checks' \
	  '  make rust    Run Rust unit tests in the copied-context test image' \
	  '  make test    Run installed-package correctness in the test image' \
	  '  make parity  Run pinned RDKit parity in the test image' \
	  '  make exact-public-invariants  Run exact public invariant tests' \
	  '  make package  Build and validate wheel/sdist artifacts under dist/' \
	  '  make timings-enum  Measure enum/support timing tradeoffs' \
	  '  make prepared-mol-zstd-dictionary  Generate the PreparedMol zstd dictionary artifact' \
	  '  make timings-prepared-mol-zstd  Measure PreparedMol zstd timing tradeoffs' \
	  '  make docs     Build the documentation site under build/docs-site/' \
	  '  make docs-serve  Serve the documentation site on DOCS_PORT' \
	  '  make ci      Run checks, rust, test, parity, and exact invariants' \
	  '' \
	  'Variables:' \
	  '  DOCS_PORT=8000  Local docs URL and docs-serve host port; must be 1..65535' \
	  '  Example: make docs-serve DOCS_PORT=8010' \
	  '  PREPARED_MOL_ZSTD_CREATED_DATE=YYYYMMDD  Optional artifact date override' \
	  '  PREPARED_MOL_ZSTD_FORCE=1  Replace the computed artifact directory' \
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

timings-enum:
	@$(NON_ROOT_GUARD); \
	$(TIMINGS_ENUM_ARTIFACTS_GUARD); \
	$(TIMING_GIT_METADATA_ENV); \
	$(COMPOSE_ENV) $(DOCKER_COMPOSE) -f $(COMPOSE_DIR)/timings-enum.yml run --build --rm timings-enum

prepared-mol-zstd-dictionary:
	@$(NON_ROOT_GUARD); \
	output_dir="$(PREPARED_MOL_ZSTD_PACKAGE_DATA_DIR)"; \
	if [[ ! "$${PREPARED_MOL_ZSTD_FORCE}" =~ ^(0|1)$$ ]]; then \
	  printf '%s\n' 'PREPARED_MOL_ZSTD_FORCE must be 0 or 1.' >&2; \
	  exit 2; \
	fi; \
	if [[ -n "$${PREPARED_MOL_ZSTD_CREATED_DATE}" && ! "$${PREPARED_MOL_ZSTD_CREATED_DATE}" =~ ^[0-9]{8}$$ ]]; then \
	  printf '%s\n' 'PREPARED_MOL_ZSTD_CREATED_DATE must be YYYYMMDD when set.' >&2; \
	  exit 2; \
	fi; \
	mkdir -p -- "$$output_dir"; \
	resolved="$$(realpath -e -- "$$output_dir")"; \
	expected="$(REPO_ROOT)/$(PREPARED_MOL_ZSTD_PACKAGE_DATA_DIR)"; \
	if [[ ! -d "$$resolved" || "$$resolved" != "$$expected" || -L "$$output_dir" || -L "$$resolved" ]]; then \
	  printf '%s\n' 'Refusing to bind PreparedMol zstd package data directory because it is missing, a symlink, or outside the repository.' >&2; \
	  exit 2; \
	fi; \
	PREPARED_MOL_ZSTD_CREATED_DATE="$${PREPARED_MOL_ZSTD_CREATED_DATE}" \
	PREPARED_MOL_ZSTD_FORCE="$${PREPARED_MOL_ZSTD_FORCE}" \
	$(COMPOSE_ENV) $(DOCKER_COMPOSE) -f $(COMPOSE_DIR)/prepared-mol-zstd-dictionary.yml run --build --rm prepared-mol-zstd-dictionary

timings-prepared-mol-zstd:
	@$(NON_ROOT_GUARD); \
	$(TIMINGS_PREPARED_MOL_ZSTD_ARTIFACTS_GUARD); \
	$(TIMING_GIT_METADATA_ENV); \
	$(COMPOSE_ENV) $(DOCKER_COMPOSE) -f $(COMPOSE_DIR)/timings-prepared-mol-zstd.yml run --build --rm timings-prepared-mol-zstd

docs:
	@$(NON_ROOT_GUARD)
	@$(DOCS_PORT_GUARD)
	@mkdir -p $(DOCS_OUTPUT_DIR)
	@$(DOCS_ARTIFACTS_GUARD); \
	find $(DOCS_OUTPUT_DIR) -mindepth 1 -maxdepth 1 -exec rm -rf -- {} +; \
	$(COMPOSE_ENV) $(DOCKER_COMPOSE) -f $(COMPOSE_DIR)/docs.yml run --rm docs

docs-serve: docs
	@$(NON_ROOT_GUARD); \
	$(DOCS_PORT_GUARD); \
	$(DOCS_ARTIFACTS_GUARD); \
	$(COMPOSE_ENV) $(DOCKER_COMPOSE) -f $(COMPOSE_DIR)/docs.yml run --rm --publish "127.0.0.1:$${DOCS_PORT}:8000" docs-serve

ci: checks rust test parity exact-public-invariants
