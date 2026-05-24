---
title: Containerized development
---

This page is for contributors. The Docker-backed Make lanes use pinned images
and avoid the host Python environment for routine checks.

```bash
make checks
make ci
make test
make package
```

- `make checks` runs offline source and container-posture checks against a
  read-only checkout.
- `make ci` expands to checks, Rust tests, installed-package correctness,
  pinned RDKit parity, and exact public invariants.
- `make test` runs installed-package correctness from a container-built wheel.
- `make package` builds and validates wheel/sdist artifacts under `dist/`.
- `make perf` is opt-in and write-enabled; it refreshes `docs/timings.*` and
  `notes/004_perf_history.jsonl`.

Routine check/test lanes do not use host `.venv`, `target`, or `dist`
artifacts.

The stricter containerization contract is tracked in
[the implementation note](https://github.com/numpde/grimace-py/blob/main/notes/014_strict_containerization_plan.md).
