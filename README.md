# GRIMACE

SMILES enumeration with exact next-token decoding.

`grimace-py` is a Rust-first RDKit add-on for exact rooted SMILES support
enumeration and online next-token decoding. It provides:

- exact support enumeration for a molecule under RDKit-style writer flags
- exact token inventories implied by that support
- legal next-token choices from a current SMILES prefix
- prepared molecule byte round trips for reuse outside RDKit

The public import name is `grimace`. Install the PyPI distribution named
`grimace-py`:

```bash
python -m pip install grimace-py
```

```python
import grimace
```

Plain `pip install grimace` installs an unrelated older package.

`grimace-py` is distributed under `PolyForm-Noncommercial-1.0.0`. Commercial
use is not permitted under the current license.

## Quick Example

The current public runtime targets RDKit writer parity for
`canonical=False, doRandom=True`.

```python
from rdkit import Chem
import grimace

mol = Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O")

all_smiles = tuple(
    grimace.MolToSmilesEnum(
        mol,
        rootedAtAtom=-1,
        isomericSmiles=False,
        canonical=False,
        doRandom=True,
    )
)

assert len(all_smiles) == 304
```

## Public API

- `MolToSmilesEnum(...)`
- `MolToSmilesDecoder(...)`
- `MolToSmilesDeterminizedDecoder(...)`
- `MolToSmilesDeviation(...)`
- `MolToSmilesTokenInventory(...)`
- `MolToSmilesTokenInventorySuperset(...)`
- `PrepareMol(...)`
- `PreparedMol`

Start with the [documentation index](docs/index.md), then see:

- [Getting started](docs/getting-started.md)
- [Runtime requirements](docs/runtime.md)
- [Python API](docs/api/python.md)
- [Prepared molecules](docs/guides/prepared-mol.md)
- [Deviation diagnostics](docs/guides/deviation.md)
- [Token inventories](docs/guides/token-inventory.md)

## Install Matrix

Package metadata declares Python `>=3.11` and `rdkit>=2026.3`.

The currently exercised release matrix publishes Linux `x86_64` wheels for
CPython `3.12` and `3.13`, plus a source distribution. Other Python versions
and non-Linux platforms are expected source-build paths today.

For a host source build, you need Rust `>=1.83` and `maturin`:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install maturin
maturin develop --release
```

## Development

Routine local checks are Docker-backed:

```bash
make checks
make ci
make package
```

See [containerized development](docs/development/containerized.md) for the
lane contract.

## License

`grimace-py` is source-available under
[PolyForm Noncommercial 1.0.0](LICENSE). Third-party components remain under
their own licenses; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
Commercial use requires a separate commercial license from the author.
