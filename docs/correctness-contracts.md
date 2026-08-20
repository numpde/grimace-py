# Correctness Contracts

`grimace` needs two correctness vocabularies. They are related, but they are
not the same contract.

## Principled Layer

The principled layer is the chemistry and language-semantics layer:

- every emitted string must be syntactically valid SMILES for the supported
  writer flags
- every emitted string must parse back to the same molecular graph and stereo
  assignment intended by the input
- equivalent SMILES strings that differ only in legal placement of directional
  bond markers may still be equally valid at this layer
- decoder prefixes and token inventories must describe a real language of
  valid continuations, not just terminal strings that happen to parse

This layer is the right place to ask whether `grimace` is building better
software than a single serializer implementation. It is also the right place
to compare parsed molecules, stereo assignments, and language-level
well-formedness.

Semantic equivalence is not enough for the current public `MolToSmilesEnum`
contract, because `grimace` also exposes token-level decoding. A terminal
string that parses correctly does not prove that all intermediate prefixes,
token choices, or branch structure match a documented language.

## RDKit Writer-Parity Layer

The RDKit writer-parity layer is narrower and deliberately implementation
specific:

- match the string support of RDKit's supported writer regime
- preserve RDKit-compatible public flag behavior where it is meaningful
- mirror RDKit traversal, rooting, fragment-order, and slash/backslash
  placement conventions when the public contract says "RDKit writer support"
- key exact expectations by RDKit version

This layer contains bolted-on RDKit-matching behavior. That is not a criticism:
it is necessary when the public promise is exact support for RDKit's
`canonical=False, doRandom=True` writer convention. But those rules should stay
visibly separated from general SMILES validity or chemical equivalence.

## Current Public Contract

The current public runtime contract is RDKit writer parity for the supported
subset:

- `canonical=False`
- `doRandom=True`
- supported writer flags listed in the Python API docs
- exact expectations validated against the pinned current stable RDKit writer
  convention

Therefore, RDKit string support remains the oracle for current public parity
tests. Parsed-object equivalence may be added as a separate evidence layer, but
it should not silently replace string equality in RDKit-parity tests.

## Classification Policy

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

## Implementation Rule

Keep the code paths and tests separable:

- principled semantic constraints should be named as semantic constraints
- RDKit-specific compatibility rules should be named as RDKit writer rules
- fixtures derived from RDKit outputs must be version-keyed
- suspected RDKit serializer quirks belong in known-quirk fixtures, not hidden
  inside generic semantic tests
- if a future API exposes semantic support broader than RDKit's writer support,
  it should be a distinct documented mode, not a fallback inside the RDKit
  parity path
