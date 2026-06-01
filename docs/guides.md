---
title: Guides
---

## Public workflows

| Task | Guide |
|---|---|
| Prepare an RDKit molecule once, serialize it, and reuse it without RDKit at runtime. | [Prepared molecules](guides/prepared-mol.html) |
| Find the first token or character where a candidate leaves the supported language. | [Deviation diagnostics](guides/deviation.html) |
| Build required SMILES-token vocabulary coverage for a molecule or dataset. | [Token inventories](guides/token-inventory.html) |

## Contributor workflows

| Task | Guide |
|---|---|
| Run checks, tests, docs, packaging, timings, and artifact generation through Docker-backed `make` lanes. | [Containerized development](development/containerized.html) |
| Read fixture-family counts, provenance counts, promotion rules, and fixture maintenance rules. | [Testing fixtures](testing-fixtures.html) |
| Trace reviewed upstream RDKit serializer tests to executable Grimace fixture evidence. | [RDKit serializer coverage](rdkit-serializer-coverage.html) |

For signatures, see [API](api/python.html). For supported flags and roots, see
[Runtime](runtime.html).
