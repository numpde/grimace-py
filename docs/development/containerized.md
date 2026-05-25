---
title: Containerized development
---

This page is for contributors. The Docker-backed Make lanes use pinned images
and avoid the host Python environment for routine checks.

For a lane-by-lane map of what runs where and what writes files, see
[Make lanes](make-lanes.md).

```bash
make checks
make ci
make test
make package
make docs
make docs-serve
```

- `make checks` runs offline source and container-posture checks against a
  read-only checkout.
- `make ci` expands to checks, Rust tests, installed-package correctness,
  pinned RDKit parity, and exact public invariants.
- `make test` runs installed-package correctness from a container-built wheel.
- `make package` builds and validates wheel/sdist artifacts under `dist/`.
- `make perf` is opt-in and write-enabled; it refreshes `docs/timings.*`,
  `docs/timing-plots/`, and `notes/004_perf_history.jsonl`.
- `make docs` builds the Pages site under `build/docs-site/` with the
  GitHub Pages Jekyll image.
- `make docs-serve` rebuilds that site and serves it at
  `http://127.0.0.1:${DOCS_PORT}/` from a container. `DOCS_PORT` defaults to
  `8000`.

Routine check/test lanes do not use host `.venv`, `target`, or `dist`
artifacts.

The stricter containerization contract is tracked in
[the implementation note](https://github.com/numpde/grimace-py/blob/main/notes/014_strict_containerization_plan.md).
