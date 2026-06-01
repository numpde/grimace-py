---
title: Prepared molecules
---

`PrepareMol(...)` pays the RDKit preparation cost once and returns an opaque
`PreparedMol`.

Use it when the same molecule will be decoded, enumerated, inventoried, or
checked more than once. It is also the storage path when you want to load a
prepared molecule later without touching RDKit again.

```python
from rdkit import Chem
import grimace

mol = Chem.MolFromSmiles("CCO")
prepared = grimace.PrepareMol(mol, isomericSmiles=False)

payload = prepared.to_bytes()
restored = grimace.PreparedMol.from_bytes(payload)
```

For compact storage, write a zstd-compressed payload:

```python
payload = prepared.to_bytes(compression="zstd")
restored = grimace.PreparedMol.from_bytes(payload)
```

The compressed payload records the shipped dictionary id in the zstd frame.
`PreparedMol.from_bytes(...)` uses that id to select the right built-in
dictionary.

`PreparedMol` is accepted anywhere the public runtime accepts a molecule:

- `MolToSmilesEnum(...)`
- `MolToSmilesDecoder(...)`
- `MolToSmilesDeterminizedDecoder(...)`
- `MolToSmilesDeviation(...)`
- `MolToSmilesTokenInventory(...)`
- `MolToSmilesTokenInventorySuperset(...)`

Writer-surface flags passed to `PrepareMol(...)` are baked into the prepared
object. Runtime calls with conflicting writer flags raise `ValueError`.
`rootedAtAtom`, `canonical`, and `doRandom` remain runtime options.

```python
FLAGS = dict(canonical=False, doRandom=True)

all_smiles = tuple(
    grimace.MolToSmilesEnum(
        restored,
        rootedAtAtom=-1,
        isomericSmiles=False,
        **FLAGS,
    )
)
```

`PreparedMol.to_bytes()` returns a versioned binary payload owned by the Rust
core. Treat the bytes as opaque.

For the supported writer flags, see [Runtime](../runtime.md).
