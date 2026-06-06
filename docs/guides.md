---
title: Guides
---

## Public workflows

- [Prepared molecules](guides/prepared-mol.html): prepare an RDKit molecule
  once, serialize it, and reuse it without RDKit at runtime.
- [Deviation diagnostics](guides/deviation.html): find the first token or
  character where a candidate leaves the supported language.
- [Token inventories](guides/token-inventory.html): build required SMILES-token
  vocabulary coverage for a molecule or dataset.

## Contributor workflows

- [Containerized development](development/containerized.html): run checks,
  tests, docs, packaging, timings, and artifact generation through
  Docker-backed `make` lanes.
- [Testing fixtures](testing-fixtures.html): read fixture-family counts,
  provenance counts, promotion rules, and fixture maintenance rules.
- [RDKit serializer coverage](rdkit-serializer-coverage.html): trace reviewed
  RDKit source-tree serializer tests to executable Grimace fixture evidence.

For signatures, see [API](api/python.html). For supported flags and roots, see
[Runtime](runtime.html).
