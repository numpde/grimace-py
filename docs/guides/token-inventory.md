---
title: Token inventories
---

`MolToSmilesTokenInventory(...)` returns the exact sorted tuple of reachable
decoder tokens for one molecule under the public writer flags.

Use exact inventory when you need the tokens that can really occur during
runtime decoding for one molecule and one flag set.

```python
inventory = grimace.MolToSmilesTokenInventory(
    mol,
    rootedAtAtom=-1,
    isomericSmiles=False,
    canonical=False,
    doRandom=True,
)
```

Use the result as exact per-molecule coverage for Grimace decoder tokens.

For dataset vocabulary coverage, use the conservative inventory. It is designed
to be fast to union across many molecules:

```python
vocab_tokens = set()
for mol in mols:
    vocab_tokens.update(
        grimace.MolToSmilesTokenInventorySuperset(
            mol,
            rootedAtAtom=-1,
            isomericSmiles=True,
            canonical=False,
            doRandom=True,
        )
    )
```

For the same molecule and flags, the exact inventory is contained in the
superset inventory:

```python
set(grimace.MolToSmilesTokenInventory(mol, **kwargs)) <= set(
    grimace.MolToSmilesTokenInventorySuperset(mol, **kwargs)
)
```

Tokens are literal decoder fragments such as `C`, `Cl`, `[C@H]`, `=`, `(`,
`)`, `1`, and `%10`.

## Dataset workflow

For a molecular dataset, the practical tokenizer workflow is:

1. Parse each molecule with RDKit.
2. Choose the writer surface you want to support, usually
   `isomericSmiles=True` for stereo-aware data.
3. Union `MolToSmilesTokenInventorySuperset(...)` across the dataset.
4. Add ordinary tokenizer special tokens such as padding, unknown, beginning,
   or end tokens according to the tokenizer library you use.
5. Keep the Grimace inventory as a required vocabulary seed or coverage check.

The superset inventory answers a coverage question: which SMILES fragments must
be representable if your tokenizer should cover Grimace's supported writer
language for the dataset?

For more tokenizer terminology, see [Concepts](../concepts.md).
