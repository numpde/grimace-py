---
title: Correctness contracts
---

Grimace separates chemical validity from RDKit writer parity. A string can
parse to the right molecule and still not be a string RDKit would emit for the
supported writer regime.

| Layer | Question | Evidence |
|---|---|---|
| Chemical/language semantics | Is the string valid SMILES for the intended molecule, stereo assignment, and prefix language? | Parsed molecule checks, stereo checks, decoder-language checks. |
| RDKit writer parity | Would RDKit emit this exact string under the supported writer flags? | RDKit-versioned fixtures and string-level parity tests. |

Semantic equivalence alone is not enough for `MolToSmilesEnum(...)` because
Grimace also exposes token-level decoding. A terminal string that parses
correctly does not prove that intermediate prefixes, token choices, or branch
structure match the supported writer language.

## Current public contract

The current public runtime scope is documented in
[Runtime](runtime.html), and active limitations are documented in
[Limitations](current-limitations.html).

RDKit string support remains the oracle for public parity tests. Parsed-object
equivalence may be added as a separate evidence layer, but it must not replace
string equality in RDKit-parity tests.

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

Keep code paths and tests separable:

- principled semantic constraints should be named as semantic constraints
- RDKit-specific compatibility rules should be named as RDKit writer rules
- fixtures derived from RDKit outputs must be version-keyed
- suspected RDKit serializer quirks belong in known-quirk fixtures, not hidden
  inside generic semantic tests
- if a future API exposes semantic support broader than RDKit's writer support,
  it should be a distinct documented mode, not a fallback inside the RDKit
  parity path
