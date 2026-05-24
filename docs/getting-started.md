---
title: Intro
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

All examples below use the currently supported runtime subset. Keep these
flags together while you are getting started:

```python
FLAGS = dict(canonical=False, doRandom=True)
```

## Enumerate exact support

Use `MolToSmilesEnum(...)` when you want the finished SMILES strings Grimace can
emit for a molecule.

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
index for one explicit root. See [Runtime](runtime.md) for the full rooting
rules.

## Decode one token at a time

Use `MolToSmilesDeterminizedDecoder(...)` when you are constructing a SMILES
string and need one legal next choice per token text for the current prefix.

```python
decoder = grimace.MolToSmilesDeterminizedDecoder(
    mol,
    rootedAtAtom=-1,
    isomericSmiles=False,
    **FLAGS,
)

target_tokens = [
    "c", "1", "(", "c", "c", "c", "c", "c", "1", "O", "C", "(",
    "=", "O", ")", "C", ")", "C", "(", "O", ")", "=", "O",
]
# This fixed target path is only for the example. Grimace tokens can be
# multi-character, so keep it as a token list rather than slicing a string.

for token in target_tokens:
    prefix = decoder.prefix if decoder.prefix else '""'
    print(f"{prefix} -> {[choice.text for choice in decoder.next_choices]}")
    decoder = next(
        choice.next_state
        for choice in decoder.next_choices
        if choice.text == token
    )
```

Output on aspirin along that target path:

```text
"" -> ['C', 'O', 'c']
c -> ['1']
c1 -> ['(', 'c']
c1( -> ['C', 'O', 'c']
c1(c -> ['(', 'c']
c1(cc -> ['c']
c1(ccc -> ['c']
c1(cccc -> ['c']
c1(ccccc -> ['1']
c1(ccccc1 -> ['C', 'O']
c1(ccccc1O -> ['C']
c1(ccccc1OC -> ['(']
c1(ccccc1OC( -> ['=', 'C']
c1(ccccc1OC(= -> ['O']
c1(ccccc1OC(=O -> [')']
c1(ccccc1OC(=O) -> ['C']
c1(ccccc1OC(=O)C -> [')']
c1(ccccc1OC(=O)C) -> ['C']
c1(ccccc1OC(=O)C)C -> ['(']
c1(ccccc1OC(=O)C)C( -> ['=', 'O']
c1(ccccc1OC(=O)C)C(O -> [')']
c1(ccccc1OC(=O)C)C(O) -> ['=']
c1(ccccc1OC(=O)C)C(O)= -> ['O']
c1(ccccc1OC(=O)C)C(O)=O -> []
```

The determinized decoder merges same-text continuations. Use
`MolToSmilesDecoder(...)` when you need branch-preserving choices instead.

## Next pages

- [Concepts](concepts.md)
- [Runtime](runtime.md)
- [API](api/python.md)
- [Prepared molecules](guides/prepared-mol.md)
- [Deviation diagnostics](guides/deviation.md)
- [Token inventories](guides/token-inventory.md)
