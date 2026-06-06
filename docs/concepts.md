---
title: Concepts
---

Keep these concepts separate:

| Concept | Meaning | Where to look |
|---|---|---|
| Support | The complete set of SMILES strings Grimace can emit for one molecule and writer options. | `MolToSmilesEnum(...)` |
| Root | The atom where a rooted traversal starts, or all roots when `rootedAtAtom=-1`. | `rootedAtAtom` |
| Decoder token | One string emitted by one decoder transition; not necessarily one character. | `MolToSmilesDecoder(...)`, `MolToSmilesDeterminizedDecoder(...)` |
| Sample | One seeded walk through Grimace's supported decoder language, with the visible choices recorded at each prefix. | `MolToSmilesSample(...)` |
| Writer parity | String-level agreement with RDKit's supported writer behavior, not just chemical equivalence. | [Correctness contracts](correctness-contracts.html), [RDKit serializer coverage](rdkit-serializer-coverage.html) |

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
of Grimace's supported writer language for the same writer options.

## Root

A root is the atom where a rooted SMILES traversal starts.

- `rootedAtAtom=-1` means all valid roots.
- `rootedAtAtom=0` means start at atom `0`.
- For disconnected molecules, the original RDKit fragment order is preserved.

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

The public choice object reports `branch_count`, the number of branch-preserving
choices represented by the visible token at the current prefix.

## Sample

`MolToSmilesSample(...)` draws one complete supported token path from the
decoder language and returns both the finished string and the per-prefix
visible token choices seen along the way.

This is a Grimace sampler, not RDKit random-writer sequence reproduction. The
sample is controlled by a required Grimace seed and by an explicit
`decoder_view`/`sampling_mode` pair.

## Writer parity

Writer parity is a string-level claim, not just a chemical-equivalence claim.
It asks whether a string belongs to the relevant writer language.

A SMILES string can be chemically valid and parse to the same molecule while
still being outside that writer language.

Use [Correctness contracts](correctness-contracts.html) for the detailed
boundary between chemical semantics and writer parity. Use
[Limitations](current-limitations.html) for the current supported scope.

## API choices

| Need | Use |
|---|---|
| Every supported finished string | `MolToSmilesEnum(...)` |
| Legal next tokens while building a string | `MolToSmilesDecoder(...)` or `MolToSmilesDeterminizedDecoder(...)` |
| One seeded legal string plus per-step token choices | `MolToSmilesSample(...)` |
| The first unsupported token or character in a candidate | [Deviation diagnostics](guides/deviation.html) |
| Dataset vocabulary coverage | [Token inventories](guides/token-inventory.html) |
| Repeated calls or storage without RDKit on read | [Prepared molecules](guides/prepared-mol.html) |
