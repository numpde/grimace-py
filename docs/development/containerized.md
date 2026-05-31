---
title: Containerized development
---

This page is for contributors. The Docker-backed `make` lanes use pinned images
and avoid the host Python environment for routine checks.

## Lane map

| Lane | Runs | Writes |
| --- | --- | --- |
| `make checks` | `checks` service in `compose/checks.yml`; read-only checkout | No project artifacts |
| `make rust` | `rust` service in `compose/test.yml`; copied-context test image | Container-local test output only |
| `make test` | `test` service in `compose/test.yml`; installed package | Container-local test output only |
| `make parity` | `parity` service in `compose/test.yml`; pinned RDKit parity | Container-local test output only |
| `make exact-public-invariants` | `exact-public-invariants` service in `compose/test.yml` | Container-local test output only |
| `make package` | `package` service in `compose/package.yml` | `dist/` |
| `make perf` | `perf` service in `compose/perf.yml` | Timing docs and timing history |
| `make prepared-mol-zstd-dictionary` | `prepared-mol-zstd-dictionary` service in `compose/prepared-mol-zstd-dictionary.yml` | `python/grimace/data/prepared_mol_zstd/` |
| `make docs` | `docs` service in `compose/docs.yml`; GitHub Pages Jekyll image | `build/docs-site/` |
| `make docs-serve` | `make docs`, then `docs-serve` service in `compose/docs.yml` | Rebuilds `build/docs-site/`; publishes a local HTTP port |
| `make ci` | `checks`, `rust`, `test`, `parity`, and `exact-public-invariants` | No package, performance, or docs artifacts |

## Details

`make ci` expands to:

```bash
make checks
make rust
make test
make parity
make exact-public-invariants
```

`make package` writes release artifacts under `dist/`. The Makefile refuses a
symlinked `dist/` and clears direct children before building.

`make perf` is opt-in and write-enabled for:

- `docs/timings.tsv`
- `docs/timings.md`
- `docs/timing-plots/`
- `notes/004_perf_history.jsonl`

`make prepared-mol-zstd-dictionary` builds an installed-package image with the
pinned generator dependencies, runs the generator contract and environment
preflight tests, then writes only to
`python/grimace/data/prepared_mol_zstd/`. The generator runs a post-flight
check on the written artifact before exiting successfully. Optional controls:

```bash
make prepared-mol-zstd-dictionary \
  PREPARED_MOL_ZSTD_CREATED_DATE=20260531 \
  PREPARED_MOL_ZSTD_FORCE=1
```

`make docs` reads `docs/` read-only and writes the local Pages build to
`build/docs-site/`. `build/` is ignored by git, so `git status build` reports
a clean tree after a docs build. The checked-in documentation source remains
under `docs/`.

`make docs-serve` first runs `make docs`, then serves `build/docs-site/`. The
Compose file does not publish a host port by itself; the Makefile publishes
`127.0.0.1:${DOCS_PORT}:8000` after validating `DOCS_PORT`. `DOCS_PORT`
defaults to `8000` and must be a decimal port from `1` to `65535` without
leading zeroes:

```bash
make docs-serve DOCS_PORT=8010
```

Routine check/test lanes do not use host `.venv`, `target`, `dist`, or
`build` artifacts.

The stricter containerization contract is tracked in
[the implementation note](https://github.com/numpde/grimace-py/blob/main/notes/014_strict_containerization_plan.md).
