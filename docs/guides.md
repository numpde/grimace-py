---
title: Guides
---

## Public workflows

| Task | Guide |
|---|---|
| Prepare an RDKit molecule once, serialize it, and reuse it without RDKit at runtime. | [Prepared molecules](guides/prepared-mol.md) |
| Find the first token or character where a candidate leaves the supported language. | [Deviation diagnostics](guides/deviation.md) |
| Build required SMILES-token vocabulary coverage for a molecule or dataset. | [Token inventories](guides/token-inventory.md) |

## Contributor workflows

| Task | Guide |
|---|---|
| Run checks, tests, docs, packaging, timings, and artifact generation through Docker-backed `make` lanes. | [Containerized development](development/containerized.md) |
| Read fixture-family counts, provenance counts, promotion rules, and fixture maintenance rules. | [Testing fixtures](testing-fixtures.md) |
| Trace reviewed upstream RDKit serializer tests to executable Grimace fixture evidence. | [RDKit serializer coverage](rdkit-serializer-coverage.md) |

For signatures, see [API](api/python.md). For supported flags and roots, see
[Runtime](runtime.md).
