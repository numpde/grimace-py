# grimace-py

Rust-first SMILES enumeration and next-token decoding, with a small Python API.

> [!WARNING]
> This library is work in progress. Expect API changes, incomplete feature
> coverage, and rough edges between releases.

## Public API

The only supported public Python package is `grimace`.

Current entrypoints:

- `MolToSmilesEnum(...)`
- `MolToSmilesDecoder(...)`

Both use the compiled Rust extension. There is no public runtime fallback.

## Quickstart

```python
from rdkit import Chem
import grimace

mol = Chem.MolFromSmiles("F/C=C\\Cl")

outputs = list(
    grimace.MolToSmilesEnum(
        mol,
        rootedAtAtom=0,
        canonical=False,
        doRandom=True,
    )
)

decoder = grimace.MolToSmilesDecoder(
    mol,
    rootedAtAtom=0,
    canonical=False,
    doRandom=True,
)
tokens = decoder.nextTokens()
```

## Install

```bash
python -m pip install grimace
```

For local development:

```bash
python -m pip install maturin
maturin develop --release
```

## License

`grimace` is source-available under PolyForm Noncommercial 1.0.0.
Commercial use requires a separate commercial license from the author.
The software is provided as is, without warranty or liability, to the extent
allowed by law. See [LICENSE](LICENSE) for the full terms.

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
