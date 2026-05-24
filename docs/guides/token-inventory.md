# Token Inventories

`MolToSmilesTokenInventory(...)` returns the exact sorted tuple of reachable
decoder tokens for one molecule under the public writer flags.

```python
inventory = grimace.MolToSmilesTokenInventory(
    mol,
    rootedAtAtom=-1,
    isomericSmiles=False,
    canonical=False,
    doRandom=True,
)
```

This is an exact runtime inventory, not a frequency distribution and not a
general-purpose tokenizer vocabulary.

For fast dataset vocabulary coverage, use the static conservative inventory:

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
