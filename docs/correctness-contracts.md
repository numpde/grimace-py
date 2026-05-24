---
title: Correctness contracts
---

Grimace intentionally separates chemical validity from RDKit writer parity.
That distinction matters when a string is chemically reasonable but not a
string RDKit would emit for the supported writer regime.

## Principled layer

The principled layer is the chemistry and language-semantics layer:

- every emitted string must be syntactically valid SMILES for the supported
  writer flags
- every emitted string must parse back to the same molecular graph and stereo
  assignment intended by the input
- equivalent SMILES strings that differ only in legal placement of directional
  bond markers may still be equally valid at this layer
- decoder prefixes and token inventories must describe a real language of
  valid continuations, not just terminal strings that happen to parse

This layer is where parsed molecules, stereo assignments, and language-level
well-formedness belong.

Semantic equivalence is not enough for the current public `MolToSmilesEnum`
contract, because `grimace` also exposes token-level decoding. A terminal
string that parses correctly does not prove that all intermediate prefixes,
token choices, or branch structure match a documented language.

## RDKit writer-parity layer

The RDKit writer-parity layer is narrower and deliberately implementation
specific:

- match the string support of RDKit's supported writer regime
- preserve RDKit-compatible public flag behavior where it is meaningful
- mirror RDKit traversal, rooting, fragment-order, and slash/backslash
  placement conventions when the public contract says "RDKit writer support"
- key exact expectations by RDKit version

This layer is narrower because it mirrors RDKit's spelling choices. That is
necessary when a fixture or API claim says a string is in covered RDKit writer
support for the `canonical=False, doRandom=True` writer convention.

## Current public contract

The current public runtime scope is documented in
[Runtime](runtime.md), and active limitations are documented in
[Limitations](current-limitations.md).

Therefore, RDKit string support remains the oracle for current public parity
tests. Parsed-object equivalence may be added as a separate evidence layer, but
it should not silently replace string equality in RDKit-parity tests.

## Classification policy

When a case differs from RDKit, classify it explicitly:

- `exact-rdkit-match`: the string is in RDKit's pinned writer support
- `semantic-equivalent`: the string parses to the same molecular graph and
  stereo assignment, but is not known to be in RDKit's pinned writer support
- `rdkit-only`: RDKit emits the string and `grimace` does not
- `semantic-error`: the string does not parse to the intended molecule/stereo
  assignment
- `rdkit-quirk`: RDKit behavior is unusual but intentionally mirrored for
  parity
- `known-rdkit-gap`: `grimace` does not yet mirror a pinned RDKit writer case

Tests and fixtures should use these labels instead of implying that every
RDKit mismatch is chemically wrong or that every semantically valid alternative
is part of RDKit writer support.

## Implementation rule

Keep the code paths and tests separable:

- principled semantic constraints should be named as semantic constraints
- RDKit-specific compatibility rules should be named as RDKit writer rules
- fixtures derived from RDKit outputs must be version-keyed
- suspected RDKit serializer quirks belong in known-quirk fixtures, not hidden
  inside generic semantic tests
- if a future API exposes semantic support broader than RDKit's writer support,
  it should be a distinct documented mode, not a fallback inside the RDKit
  parity path
