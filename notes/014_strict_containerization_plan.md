# Strict Containerization Plan

This note plans Docker-backed development, test, package, and release-support
lanes for `grimace-py`. The goal is a strict, auditable boundary similar in
spirit to `vote-mcp` and `dapp32`: routine checks should not depend on the host
Python environment, and every container lane should declare its filesystem,
network, user, and write posture.

## Goal

Make Docker-backed lanes the normal way to run local checks, package
validation, and release-support commands. The checklist below is the contract:
each item should be independently reviewable, runnable, and testable before the
next heavier lane is added.

## Target Layout

```text
.dockerignore
Makefile
compose/
  checks.yml
  test.yml
  package.yml
  perf.yml
containers/
  checks/Dockerfile
  test/Dockerfile
  package/Dockerfile
  perf/Dockerfile
tests/checks/
  test_container_posture.py
  test_release_notes.py
```

Dependency input files may be added later under `requirements/` if pinning
Python build/test tools outside Dockerfiles becomes useful. Do not add them
until the dependency surface is large enough to justify the extra object.

## Strategic Checklist

- [x] Freeze the current baseline.
  - Confirm `main` is clean and pushed.
  - Record current host-successful commands: `cargo test`, full Python suite,
    package build, and `twine check`.
  - Leave GitHub CI unchanged until local container lanes are proven.

  Baseline recorded on 2026-05-23 from commit `107529d` before containerization
  implementation:

  - `cargo test --lib`: 57 passed.
  - `maturin develop --release`: installed editable `grimace-py 0.1.11` in
    the active Python 3.12 environment.
  - `python -m unittest discover -s tests -t . -q`: 316 passed, 6 skipped.
  - `python -m unittest tests.run_pinned_rdkit_parity -q`: 11 passed.
  - `python -m unittest tests.run_exact_public_invariants -q`: 43 passed.
  - `maturin build --release --out /tmp/grimace-py-baseline-dist -i
    python3.12`: wheel built.
  - `maturin build --release --sdist --out /tmp/grimace-py-baseline-dist -i
    python3.12`: sdist built.
  - `python -m twine check /tmp/grimace-py-baseline-dist/*`: wheel and sdist
    passed.
  - Wheel installed-package correctness in a temporary venv: 70 passed.
  - Sdist installed-package correctness in a temporary venv: 70 passed.

  Local baseline environment:

  - Python `3.12.13` from `grimace-py-dev`.
  - Rust `1.83.0`.
  - RDKit `2026.3.1`.
  - Maturin `1.13.1`.
  - Twine `6.2.0`.

- [x] Define the strict container contract.
  - No host Python/conda dependency for normal checks.
  - No Docker socket mounts.
  - No root execution for Docker Make lanes.
  - No network during check/test runtime unless explicitly named.
  - Read-only repo mounts for source-inspection checks.
  - Copied build context for build, test, and package lanes.
  - Read-write repo mounts only for lanes whose purpose is to update checked-in
    artifacts.
  - Controlled write locations only for build artifacts and perf output.
  - Pinned base images by digest.
  - Pinned Rust `1.83.0`, RDKit `2026.3.1`, `maturin`, and `twine`.

- [x] Add `.dockerignore` first.
  - Exclude `.git`, `.codex`, `.agents`, `.idea`, `.vscode`, `.venv`.
  - Exclude `target`, `dist`, `build`, `*.egg-info`.
  - Exclude Python caches and coverage output.
  - Exclude compiled extension artifacts.
  - Exclude local secrets and env files.
  - Keep source, tests, docs, fixtures, `Cargo.lock`, `pyproject.toml`,
    `Cargo.toml`, and `rust-toolchain.toml`.

  Validated by exporting a scratch Docker build context to
  `/tmp/grimace-dockerignore-context`: required source/build inputs were
  present, and `.git`, `.venv`, `target`, `dist`, and nested `__pycache__`
  directories were absent.

- [x] Add the offline repository check lane.
  - Create `containers/checks/Dockerfile`.
  - Use a small pinned Python Alpine image.
  - Do not copy the repository into the image.
  - Mount the repository read-only at runtime.
  - Add non-root user `65532:65532`.
  - Keep checks dependency-free; use the Python standard library.

  Added `containers/checks/Dockerfile` from pinned
  `python:3.12.13-alpine3.22`. The image has no `COPY`, `ADD`, or `RUN`
  instructions. It was built locally and validated with `--network none`,
  `--read-only`, `--cap-drop ALL`, `no-new-privileges:true`, bounded pids and
  memory, and UID/GID `65532:65532`.

- [x] Add `compose/checks.yml`.
  - Mount the repository read-only.
  - Set `network_mode: "none"`.
  - Set `read_only: true`.
  - Drop all capabilities.
  - Set `security_opt: no-new-privileges:true`.
  - Add bounded `pids_limit` and `mem_limit`.
  - Use tmpfs for `/tmp`.
  - Do not mount the Docker socket.

  Added `compose/checks.yml` with a single `checks` service. Validated with
  `docker compose -f compose/checks.yml config` and
  `docker compose -f compose/checks.yml run --rm --build checks`.

- [x] Add the Makefile shell contract.
  - `SHELL := bash`.
  - `.SHELLFLAGS := -eu -o pipefail -c`.
  - `make help`.
  - `make checks`.
  - Root guard in the `dapp32` style.
  - `DOCKER_COMPOSE ?= docker compose`.

  Added the minimal Makefile contract for `help` and `checks`. Validated with
  `make help`, `make -n checks`, and `make checks`.

- [x] Add posture tests.
  - `tests/checks/test_container_posture.py`.
  - Assert no Compose file has top-level `name:`.
  - Assert `checks.yml` is offline, read-only, non-root, and capability-free.
  - Assert Makefile refuses root Docker lanes.
  - Assert `.dockerignore` excludes dangerous, local, and generated paths.
  - Assert every `v0.1.*` tag has `notes/releases/<tag>.md`.
  - Assert release workflow checks and uses `notes/releases/<tag>.md` from the
    tag ref.

  Added standard-library posture tests under `tests/checks/` and wired
  `compose/checks.yml` to run them. Validated directly with
  `python -m unittest discover -s tests/checks -t . -q` and through
  `make checks`.

- [x] Run and stabilize `make checks`.
  - It must be fast, offline, deterministic, and write-free.
  - Treat it as the guardrail before adding heavier lanes.

  Validated `make checks` from a clean working tree. It ran 7 offline posture
  tests, left no working-tree changes, and left no stopped `compose-checks`
  containers behind.

- [x] Add the test/build image.
  - Create `containers/test/Dockerfile`.
  - Use pinned glibc Python 3.12, not Alpine, because RDKit wheels are
    manylinux/glibc-oriented.
  - Install Rust `1.83.0`.
  - Install pinned `maturin` and `rdkit==2026.3.1`.
  - Copy the repository into the image as the build context.
  - Prefer building a wheel and installing it over `maturin develop`, so the
    test lane exercises package installation behavior.
  - Run as non-root.

  Added `containers/test/Dockerfile` using pinned Python slim and Rust slim
  base images, with no `apt-get` layer. The image copies the Python runtime
  into the Rust image, copies the repository into `/src`, builds a wheel with
  pinned `maturin`, installs it beside pinned `rdkit==2026.3.1`, and runs
  installed-package correctness as UID/GID `65532:65532`. Validated with
  `docker build --file containers/test/Dockerfile --tag grimace-py-test:local
  .`, strict `docker run` runtime flags, and `make checks`.

- [x] Add the correctness Compose lane.
  - Create `compose/test.yml`.
  - Provide services or commands for `rust`, `test`, `parity`, and
    `exact-public-invariants`.
  - Disable runtime network where feasible.
  - Avoid host `.venv`.
  - Do not mount the host repository read-write.
  - Do not depend on host-generated `target`, `dist`, `.venv`, or extension
    artifacts.

  Added `compose/test.yml` with copied-context services for Rust tests,
  installed-package correctness, pinned RDKit parity, and exact public
  invariants. Runtime services use no network, read-only root filesystems,
  non-root UID/GID, no capabilities, no host repository mount, and tmpfs build
  output. Validated all four services plus `make checks`.

- [x] Add Make targets for correctness.
  - `make rust`.
  - `make test`.
  - `make parity`.
  - `make exact-public-invariants`.
  - `make ci` initially expands to `checks rust test parity
    exact-public-invariants`.

  Added all correctness Make targets and `ci` dependency expansion. Validated
  with `make help`, posture tests, and `make ci`.

- [x] Add the package lane.
  - Create `containers/package/Dockerfile` only if the test image becomes too
    broad.
  - Copy the repository into the image as the build context.
  - Build wheel and sdist.
  - Run `twine check`.
  - Install built wheel/sdist and run installed-package correctness.
  - Write artifacts only to `dist/` or a named output path.

  Added a separate package image because package validation needs `twine` and
  writes artifacts. The package lane uses copied source context, writes only to
  host `dist/`, builds wheel and sdist, runs `twine check`, and validates both
  installed artifacts. Validated with `make package` and `make checks`.

- [x] Add the perf lane last.
  - `make perf`.
  - Keep it clearly opt-in.
  - Mount only the timing artifacts read-write because it updates
    `docs/timings.*` and `notes/004_perf_history.jsonl`.
  - Keep every other routine lane write-free with respect to the source tree.
  - Keep it out of default `make ci`.

  Added a separate perf image so the normal test image remains correctness-only.
  The Make target captures Git metadata before the benchmark writes timing
  artifacts, then the perf image builds and installs the package from copied
  context and runs the opt-in performance suite with only the timing outputs
  mounted read-write. Runtime network is disabled. Validated with
  `docker compose -f compose/perf.yml config`, `make checks`, and a targeted
  perf-container writable-artifact check.

- [x] Document minimally.
  - Add a short README section for containerized development.
  - Mention `make checks`, `make test`, `make package`, and `make perf`.
  - State that default lanes avoid host Python and host build artifacts.

  Added a short README section for Docker-backed local lanes and updated the
  timing-regeneration command to `make perf`. Added matching command pointers
  to `tests/README.md`. Validated with `make checks` and `git diff --check`.

- [x] Only then revise GitHub CI.
  - Switch CI to Make targets after local container lanes are stable.
  - Keep release workflow separate.
  - Do not make perf part of default CI.

  Replaced host-venv CI jobs with Docker-backed `make ci` and `make package`
  jobs. Kept perf out of default CI. Added offline posture checks for CI and
  release workflow token permissions and non-persistent checkout credentials.
  Added release artifact allowlist validation before GitHub release and PyPI
  publish. Pinned workflow action refs to commit SHAs and GitHub-hosted runner
  labels to `ubuntu-24.04`. Tag-triggered release builds now require `vX.Y.Z`
  tag shape before building artifacts.

- [x] Final validation before committing implementation.
  - `make checks`.
  - `make rust`.
  - `make test`.
  - `make parity`.
  - `make exact-public-invariants`.
  - `make package`.
  - `git diff --check`.
  - Confirm no generated/cache artifacts are tracked.

  Final validation passed on 2026-05-23. `make package` regenerated ignored
  artifacts under `dist/`; no generated or cache artifacts were untracked or
  staged for commit.

## Lane Notes

### Checks

`make checks` is the cheapest and strictest lane. It should inspect checked-in
text and rendered configuration only. It should not install dependencies, talk
to the network, write files, or depend on host state.

### Correctness

`make test`, `make parity`, and `make exact-public-invariants` should run from a
container-built wheel with pinned RDKit. This makes the lane closer to installed
package behavior than a source-tree import while still allowing tests to read
checked-in fixtures.

### Package

`make package` should produce and validate release-shaped artifacts, but it
should not publish. Publishing remains the tag-triggered GitHub workflow.
It refuses root execution, non-numeric or zero container UID/GID settings, and
refuses to clean or bind-mount `dist/` when `dist/` is a symlink.

### Perf

`make perf` is intentionally not strict in the same way as default checks: it is
write-enabled and long-running. Its strictness comes from being explicit,
opt-in, and limited to the timing output files it owns.
It refuses to bind perf output files that are missing, symlinks, or resolved
outside the physical repository path. The physical repository path and artifact
list are Makefile facts, not command-line overrides.

### Rust Dependency Lock

The Docker and release lanes use locked Rust builds (`cargo fetch --locked`,
`cargo test --offline`, and `maturin build --locked`), so `Cargo.lock` is part
of the source boundary. It must stay checked in and must not be excluded from
the Docker build context.

### Source Distribution Boundary

The sdist is a release artifact, not an archive of every checked-in diagnostic.
Package and release validation reject unsafe archive paths, links, local
credential filenames, build outputs, `tmp/`, and raw `notes/perf_reports/`
captures.

### Docker Build Context Secrets

The Docker build context excludes local editor state, generated artifacts, and
common credential files for Python, npm, Cargo, cloud CLIs, SSH, TLS keys, and
token-like filenames.

## Repository Boundary Decision

Use three explicit repository boundary modes:

- Read-only mount: static source checks, release-note checks, Docker/Compose
  posture checks.
- Copied build context: Rust tests, Python correctness tests, package builds,
  installed-package checks.
- Explicit read-write artifact mounts: opt-in lanes whose purpose is to update
  checked-in generated artifacts, currently timing docs and timing history.

The long-term default for `test` and `package` is copied build context. It is
stricter than a bind mount because `.dockerignore` controls the source snapshot,
host `.venv`, `target`, `dist`, local extension modules, and caches cannot leak
in, and the container cannot accidentally write into the working tree. This also
makes local and CI behavior converge around the same release-like package shape.

Use read-only runtime mounts only where the lane intentionally inspects the
operator's current working tree without building it. Use read-write mounts only
when the lane is explicitly an artifact update lane, and mount the narrowest
file or directory set that lane needs.
