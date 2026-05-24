---
title: Concepts
---

Grimace is easiest to use if you keep four ideas separate: support, root,
decoder token, and writer parity.

## Support

For one molecule and one set of writer flags, the support is the set of SMILES
strings that Grimace can emit.

```python
support = tuple(
    grimace.MolToSmilesEnum(
        mol,
        rootedAtAtom=-1,
        isomericSmiles=False,
        canonical=False,
        doRandom=True,
    )
)
```

This is different from RDKit's `MolToSmiles(..., doRandom=True)`. RDKit returns
one sampled string per call. `MolToSmilesEnum(...)` enumerates the full support
for the same writer regime.

## Root

A root is the atom where a rooted SMILES traversal starts.

- `rootedAtAtom=-1` means all valid roots.
- `rootedAtAtom=0` means start at atom `0`.
- for disconnected molecules, the original RDKit fragment order is preserved.

Most users should start with `rootedAtAtom=-1`. Use one explicit root when you
need to compare or constrain a particular traversal start.

## Decoder token

A Grimace token is one string emitted by one decoder transition. It is not an
integer token id, and it is not necessarily one character.

Examples include:

- atom or bond fragments such as `C`, `c`, `Cl`, `[C@H]`, `=`, `/`, and `\\`
- syntax fragments such as `(`, `)`, `1`, and `%10`

`MolToSmilesDecoder(...)` is branch-preserving: two choices can have the same
token text if they represent different underlying writer branches.
`MolToSmilesDeterminizedDecoder(...)` merges same-text choices.

## Writer parity

The current public contract is RDKit writer parity for the supported runtime
regime. A SMILES string can be chemically valid and parse to the same molecule
while still being outside RDKit's writer support for that regime.

Use [Correctness contracts](correctness-contracts.md) for the detailed
boundary between chemical semantics and RDKit writer parity.

## What to use when

- Need every supported finished string: use `MolToSmilesEnum(...)`.
- Need legal next tokens while building a string: use
  `MolToSmilesDecoder(...)` or `MolToSmilesDeterminizedDecoder(...)`.
- Need to explain why a candidate is not supported: use
  [deviation diagnostics](guides/deviation.md).
- Need vocabulary coverage for a dataset: use
  [token inventories](guides/token-inventory.md).
- Need to reuse the same molecule repeatedly or store it without RDKit: use
  [prepared molecules](guides/prepared-mol.md).
