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

.PHONY: help checks

help:
	@printf '%s\n' \
	  'Supported lanes:' \
	  '  make checks  Run offline repository/source checks' \
	  '' \
	  'Docker-backed lanes refuse root execution and use strict Compose posture.'

checks:
	$(call compose_run,checks.yml,checks)
