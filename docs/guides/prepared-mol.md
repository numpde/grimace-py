# Prepared Molecules

`PrepareMol(...)` pays the RDKit preparation cost once and returns an opaque
`PreparedMol`.

```python
from rdkit import Chem
import grimace

mol = Chem.MolFromSmiles("CCO")
prepared = grimace.PrepareMol(mol, isomericSmiles=False)

payload = prepared.to_bytes()
restored = grimace.PreparedMol.from_bytes(payload)
```

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
