---
title: Getting started
---

`grimace-py` installs as the PyPI distribution `grimace-py` and imports as
`grimace`:

```bash
python -m pip install grimace-py
```

```python
from rdkit import Chem
import grimace
```

All examples below use the currently supported runtime subset:

```python
FLAGS = dict(canonical=False, doRandom=True)
```

## Enumerate exact support

```python
mol = Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O")

all_smiles = tuple(
    grimace.MolToSmilesEnum(
        mol,
        rootedAtAtom=-1,
        isomericSmiles=False,
        **FLAGS,
    )
)

assert len(all_smiles) == 304
```

Use `rootedAtAtom=-1` for the all-roots support, or pass a nonnegative atom
index for one explicit root.

## Decode one token at a time

```python
decoder = grimace.MolToSmilesDecoder(
    mol,
    rootedAtAtom=0,
    isomericSmiles=False,
    **FLAGS,
)

for _ in range(7):
    prefix = decoder.prefix if decoder.prefix else '""'
    print(f"{prefix} -> {[choice.text for choice in decoder.next_choices]}")
    decoder = decoder.next_choices[0].next_state
```

Early output on aspirin:

```text
"" -> ['C']
C -> ['C']
CC -> ['(', '(']
CC( -> ['=']
CC(= -> ['O']
CC(=O -> [')']
CC(=O) -> ['O']
```

Duplicate token texts are meaningful in `MolToSmilesDecoder(...)`: they are
different branch choices with the same emitted text. Use
`MolToSmilesDeterminizedDecoder(...)` when you want at most one choice per
token text.

## What counts as a token?

A Grimace token is one string emitted by one decoder transition. Tokens are
defined by the walker, not by splitting a finished SMILES into characters and
not by integer token IDs.

Examples include `C`, `c`, `Cl`, `[C@H]`, `=`, `/`, `\\`, `(`, `)`, `1`, and
`%10`.

## Next pages

- [Runtime requirements](runtime.md)
- [Python API](api/python.md)
- [Prepared molecules](guides/prepared-mol.md)
- [Deviation diagnostics](guides/deviation.md)
- [Token inventories](guides/token-inventory.md)
