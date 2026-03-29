# grimace-py

`grimace` is a Rust-first cheminformatics library for exact SMILES enumeration
and next-token decoding from RDKit molecules. It computes the full rooted SMILES
support of a molecule under an RDKit-style writing regime, and exposes the
corresponding next-token choices during decoding. This is useful when you want
to train, test, or debug molecular generation models against all valid rooted
SMILES continuations of the same molecule, rather than against a single
serialization or a few random samples.

The package is motivated by research on NMR spectroscopy with language
transformers: <https://numpde.github.io/shared/msc/>.

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

For the current GitHub release, install the wheel that matches your Python
version:

```bash
python3.12 -m pip install \
  https://github.com/numpde/grimace-py/releases/download/v0.1.0/grimace-0.1.0-cp312-cp312-manylinux_2_34_x86_64.whl

python3.13 -m pip install \
  https://github.com/numpde/grimace-py/releases/download/v0.1.0/grimace-0.1.0-cp313-cp313-manylinux_2_34_x86_64.whl
```

For local development:

```bash
python -m venv .venv
. .venv/bin/activate
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
