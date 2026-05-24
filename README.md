# grimace-py

`grimace-py` is a Rust-first RDKit add-on for modeled exact rooted SMILES
support enumeration, online next-token decoding, and reusable prepared
molecules. It provides:

- exact support enumeration for Grimace's modeled RDKit-style writer language
- exact token inventories implied by that support
- legal next-token choices from a current SMILES prefix
- prepared molecule bytes for reuse outside RDKit

GRIMACE stands for "graph representation integrating multiple alternate
chemical equivalents", motivated by research on NMR spectroscopy with
language transformers ([link](https://numpde.github.io/shared/msc/)).

The public import name is `grimace`. Install the PyPI distribution named
`grimace-py`:

```bash
python -m pip install grimace-py
```

```python
import grimace
```

Plain `pip install grimace` installs an unrelated older package.

Documentation: [numpde.github.io/grimace-py](https://numpde.github.io/grimace-py/).
Repository: [github.com/numpde/grimace-py](https://github.com/numpde/grimace-py).

`grimace-py` is distributed under `PolyForm-Noncommercial-1.0.0`. Commercial
use is not permitted under the current license.

## Quick example

The current public runtime models the documented RDKit-style writer regime for
`canonical=False, doRandom=True`. Full RDKit serializer parity is not complete;
known stereo gaps are tracked in
[RDKit serializer coverage](docs/rdkit-serializer-coverage.md).

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

## What to use

- Enumerate every supported string with `MolToSmilesEnum(...)`.
- Step through legal next tokens with `MolToSmilesDecoder(...)` or
  `MolToSmilesDeterminizedDecoder(...)`.
- Diagnose rejected candidates with `MolToSmilesDeviation(...)`.
- Build dataset token coverage with `MolToSmilesTokenInventorySuperset(...)`.
- Reuse prepared molecules with `PrepareMol(...)` and `PreparedMol`.

Start with the [documentation site](https://numpde.github.io/grimace-py/), or
read the checked-in [documentation index](docs/index.md):

- [Getting started](docs/getting-started.md)
- [Concepts](docs/concepts.md)
- [Runtime requirements](docs/runtime.md)
- [Python API](docs/api/python.md)
- [Prepared molecules](docs/guides/prepared-mol.md)
- [Deviation diagnostics](docs/guides/deviation.md)
- [Token inventories](docs/guides/token-inventory.md)

## Install matrix

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
