---
title: Timings
---

The checked-in timing pages are opt-in benchmark snapshots from one development
machine. They are useful for comparing code paths and storage choices in this
repository, not for predicting every workload.

| Benchmark | Measures | Artifacts |
|---|---|---|
| [Enum/support timings](timings-enum.md) | Exact support enumeration, decoder traversal, and RDKit random-writer sampling on curated molecules. | `docs/timings-enum.tsv`, `docs/timings-enum-plots/` |
| [PreparedMol zstd timings](timings-prepared-mol-zstd.md) | PreparedMol compression and decompression across zstd levels, with and without the shipped dictionary. | `docs/timings-prepared-mol-zstd.tsv`, `docs/timings-prepared-mol-zstd-plots/` |

Both lanes run in Docker-backed, network-disabled runtime containers. See
[Containerized development](development/containerized.md) for the exact `make`
commands and writable outputs.
