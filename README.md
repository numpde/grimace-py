# grimace-py

Rust-first SMILES enumeration and next-token decoding, with a small Python API.

## Public API

The only supported public Python package is `smiles_next_token`.

Current entrypoints:

- `MolToSmilesEnum(...)`
- `MolToSmilesDecoder(...)`

Both use the compiled Rust extension. There is no public runtime fallback.

## Quickstart

```python
from rdkit import Chem
import smiles_next_token

mol = Chem.MolFromSmiles("F/C=C\\Cl")

outputs = list(
    smiles_next_token.MolToSmilesEnum(
        mol,
        rootedAtAtom=0,
        canonical=False,
        doRandom=True,
    )
)

decoder = smiles_next_token.MolToSmilesDecoder(
    mol,
    rootedAtAtom=0,
    canonical=False,
    doRandom=True,
)
tokens = decoder.nextTokens()
```

## Current limits

The public API mirrors RDKit `MolToSmiles` flag names, but only a strict subset
is implemented today:

- `rootedAtAtom >= 0`
- `canonical=False`
- `doRandom=True`
- singly-connected molecules only

Supported writer flags:

- `isomericSmiles`
- `kekuleSmiles`
- `allBondsExplicit`
- `allHsExplicit`
- `ignoreAtomMapNumbers`

Unsupported combinations fail fast with `NotImplementedError`.

## Docs

- [Docs index](docs/README.md)
- [Python API](docs/api/python.md)
- [Concepts](docs/concepts.md)
- [Correctness](docs/correctness.md)
