---
title: RDKit serializer coverage
---

This page explains which RDKit SMILES-writer tests Grimace has reviewed and how
each relevant RDKit behavior is covered in Grimace's checked-in fixtures.
Here, "upstream" means RDKit's own source tree: the tests and source blocks
maintained by RDKit, not tests invented inside Grimace.

Use it to answer two questions:

- Did we inspect the RDKit source-tree serializer case?
- Which Grimace fixture proves the matching behavior, or records the known gap?

The ledger is the traceability map. The tests enforce the claims by loading the
linked fixtures.

Ledger:

- `tests/fixtures/rdkit_upstream_serializer_coverage/2026.03.1.json`

Audited RDKit source snapshot:

- `tests/fixtures/rdkit_upstream_serializer_sources/2026.03.1/`

## Current coverage

Snapshot for RDKit `2026.03.1`, generated from:

```bash
python scripts/report_rdkit_serializer_coverage.py
```

| Status | Entries | Meaning |
|---|---:|---|
| `covered` | 54 | Relevant upstream claim has executable Grimace evidence. |
| `known-gap` | 6 | Relevant upstream claim has executable failing diagnostics. |
| `out-of-scope` | 209 | Reviewed upstream block does not map to the current Grimace public surface. |
| `needs-fixture` | 0 | No unfinished relevant entries are left without fixture mapping. |
| `unreviewed` | 0 | No regenerated entries are waiting for triage. |

Covered entries currently link to 76 Grimace fixture references.

By upstream file:

| Upstream file | Entries |
|---|---:|
| `Code/GraphMol/SmilesParse/catch_tests.cpp` | 151 |
| `Code/GraphMol/Wrap/rough_test.py` | 68 |
| `Code/GraphMol/SmilesParse/cxsmiles_test.cpp` | 31 |
| `Code/JavaWrappers/gmwrapper/src-test/org/RDKit/SmilesDetailsTests.java` | 19 |

By parser kind:

| Kind | Entries |
|---|---:|
| `cpp_section` | 120 |
| `python_test` | 68 |
| `cpp_test_case` | 62 |
| `java_test` | 19 |

Most matched serializer terms:

| Term | Entries |
|---|---:|
| `MolToSmiles` | 130 |
| `CXSmiles` | 128 |
| `MolToCXSmiles` | 111 |
| `SmilesWriteParams` | 69 |
| `rootedAtAtom` | 12 |
| `allBondsExplicit` | 11 |
| `isomericSmiles` | 10 |
| `allHsExplicit` | 7 |
| `doRandom` | 6 |
| `kekuleSmiles` | 6 |
| `MolToRandomSmilesVect` | 5 |
| `ignoreAtomMapNumbers` | 4 |

For fixture-family and provenance counts, see
[Testing fixtures](testing-fixtures.html).

## How to read an entry

Each ledger entry has three parts:

1. Parser-owned fields: upstream file, line range, language, kind, matched
   serializer terms, and snippet hash.
2. Reviewed fields: status, claim label, notes, and `grimace_links`.
3. Linked fixtures: concrete fixture files and case IDs that enforce covered
   or known-gap claims.

Use `grimace_links` for executable evidence. Avoid prose-only coverage claims
when a claim can be represented by a fixture case.

## Status meanings

`covered` means a relevant upstream serializer claim has corresponding Grimace
evidence: exact support equality, token-inventory equality, deterministic
writer-output membership, or bounded decoder-path membership when full support
materialization is too large.

`known-gap` means the upstream claim is relevant and has executable pinned
fixture coverage, but at least one parity assertion intentionally fails against
the current implementation.

`out-of-scope` means the upstream test does not map to Grimace's current public
surface. Common examples are CXSMILES extension serialization, wrapper API
smoke tests, canonical-ranking behavior outside the supported
`canonical=False, doRandom=True` regime, and internal RDKit helper APIs that
Grimace does not expose.

`needs-fixture` and `unreviewed` are unfinished triage states. The checked-in
ledger should keep both at zero.

## Known gaps

The six `known-gap` entries are concentrated in RDKit's #4582 and manual
bond-stereo regressions:

- GitHub #4582 bulk random double-bond/ring-closure outputs for CHEMBL409450.
- GitHub #4582 continued / #3967 part 2 directional ring-closure output.
- Manual multi-double-bond stereo outputs from `testBondSetStereoDifficultCase`.
- Manual stereo-atom mutation outputs from `testBondSetStereoAtoms`.

These point at RDKit-equivalent traversal-order state for coupled directional
stereo tokens.

## Maintenance workflow

When updating RDKit serializer coverage:

1. Refresh or add the local RDKit source snapshot and manifest.
2. Regenerate parser-owned coverage fields:

```bash
python scripts/extract_rdkit_serializer_cases.py --write
```

3. Review every new `unreviewed` entry.
4. Add or link executable fixtures for every in-scope claim.
5. Run the reports and contract tests.

Useful commands:

```bash
python scripts/report_rdkit_serializer_coverage.py
python scripts/report_correctness_coverage.py
python scripts/report_rdkit_serializer_coverage.py --fail-untriaged
```
