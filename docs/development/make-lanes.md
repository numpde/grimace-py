---
title: Make lanes
---

This page is for contributors. The routine `make` lanes run in Docker
containers; they do not use the host Python environment.

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
| `make docs` | `docs` service in `compose/docs.yml`; GitHub Pages Jekyll image | `build/docs-site/` |
| `make docs-serve` | `make docs`, then `docs-serve` service in `compose/docs.yml` | Rebuilds `build/docs-site/`; publishes a local HTTP port |
| `make ci` | `checks`, `rust`, `test`, `parity`, and `exact-public-invariants` | No package, performance, or docs artifacts |

## CI lane

`make ci` expands to:

```bash
make checks
make rust
make test
make parity
make exact-public-invariants
```

## Artifact details

`make package` runs the `package` service from `compose/package.yml`. It writes
release artifacts under `dist/`. The Makefile refuses a symlinked `dist/` and
clears direct children before building.

`make perf` runs the `perf` service from `compose/perf.yml`. It is opt-in and
write-enabled for:

- `docs/timings.tsv`
- `docs/timings.md`
- `docs/timing-plots/`
- `notes/004_perf_history.jsonl`

## Documentation lanes

`make docs` runs the `docs` service from `compose/docs.yml`. It reads `docs/`
read-only and writes the local Pages build to `build/docs-site/`.

`build/` is ignored by git, so `git status build` reports a clean tree even
after a docs build. The checked-in documentation source remains under `docs/`.

`make docs-serve` first runs `make docs`, then runs the `docs-serve` service
from `compose/docs.yml`. The Compose file does not publish a host port by
itself; the Makefile publishes `127.0.0.1:${DOCS_PORT}:8000` after validating
`DOCS_PORT`.

`DOCS_PORT` defaults to `8000` and must be a decimal port from `1` to `65535`
without leading zeroes:

```bash
make docs-serve DOCS_PORT=8010
```
