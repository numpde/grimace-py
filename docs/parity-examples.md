---
title: Parity examples
---

These examples show how to read Grimace's RDKit writer-parity claims. For the
policy, see [Correctness contracts](correctness-contracts.html). For fixture
families and counts, see [Testing fixtures](testing-fixtures.html).

## Exact support parity

Fixture case `cco_root1_nonstereo` pins a small exact support:

| Field | Value |
|---|---|
| Molecule | `CCO` |
| RDKit version | `2026.03.1` |
| Writer surface | `canonical=False`, `doRandom=True`, `isomericSmiles=False`, `rootedAtAtom=1` |
| Exact support | `C(C)O`, `C(O)C` |

The corresponding test checks the complete rooted support, not one sampled
output. It also checks that the token inventory matches the same support.

Fixture:
[`rdkit_exact_small_support/2026.03.1.json`](https://github.com/numpde/grimace-py/blob/main/tests/fixtures/rdkit_exact_small_support/2026.03.1.json)

## Writer membership parity

Fixture case `writer_flags_01_propane_all_bonds_explicit` pins one RDKit writer
output:

| Field | Value |
|---|---|
| Molecule | `CCC` |
| RDKit version | `2026.03.1` |
| RDKit writer call | `canonical=True`, `doRandom=False`, `isomericSmiles=False`, `allBondsExplicit=True` |
| Grimace support surface | `canonical=False`, `doRandom=True`, `isomericSmiles=False`, `allBondsExplicit=True` |
| RDKit output | `C-C-C` |

The matching test checks that this exact RDKit output is in Grimace's supported
writer language for the same non-ranking writer flags. This is the right
evidence shape when the important claim is membership of a concrete RDKit
writer string, not storing every possible support string in the documentation.

Fixture:
[`rdkit_writer_membership/2026.03.1/20_writer_flags.json`](https://github.com/numpde/grimace-py/blob/main/tests/fixtures/rdkit_writer_membership/2026.03.1/20_writer_flags.json)

## Semantic equivalence is not writer parity

The `cco_root1_nonstereo` case also shows the boundary. Under the same RDKit
`2026.03.1` fixture, the string `CCO` parses to ethanol, but it is not in the
pinned rooted writer support for `rootedAtAtom=1`:

```text
C(C)O
C(O)C
```

So `CCO` is chemically fine for the molecule, but it is not a member of that
specific rooted writer language. Grimace's parity tests use string-level writer
membership, not only parsed-molecule equivalence.

## Known parity gap

Fixture case `github3967_part2_directional_ring_closure_canonical` pins a
current failing parity case:

| Field | Value |
|---|---|
| Input | `C1=CC/C=C2C3=C/CC=CC=CC\3C\2C=C1` |
| RDKit version | `2026.03.1` |
| RDKit writer call | `canonical=True`, `doRandom=False`, `isomericSmiles=True` |
| RDKit output | `C1=CC/C=C2\C3=C\CC=CC=CC3C2C=C1` |

That RDKit output is valid evidence, but it is not yet in Grimace's passing
writer-parity corpus. It stays in the known-gap fixture until the coupled
directional-stereo behavior is implemented and the case can move into a
passing fixture family.

Fixture:
[`rdkit_known_stereo_gaps/2026.03.1.json`](https://github.com/numpde/grimace-py/blob/main/tests/fixtures/rdkit_known_stereo_gaps/2026.03.1.json)
