# GRIMACE

SMILES enumeration with next-token distribution.

`grimace` is a Rust-first cheminformatics library for exact rooted SMILES
enumeration and online next-token decoding from RDKit molecules. It computes
the full rooted SMILES support of a molecule under an RDKit-style writing
regime, and it can also step through the same support one token at a time:
at each prefix it exposes the legal next tokens, then advances when you choose
one.

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

mol = Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O")

outputs = list(
    grimace.MolToSmilesEnum(
        mol,
        rootedAtAtom=0,
        isomericSmiles=False,
        canonical=False,
        doRandom=True,
    )
)
# len(outputs) == 12

decoder = grimace.MolToSmilesDecoder(
    mol,
    rootedAtAtom=0,
    isomericSmiles=False,
    canonical=False,
    doRandom=True,
)
while decoder.prefix() != "CC(=O)Oc1c":
    print(f"{decoder.prefix()} -> {list(decoder.nextTokens())}")
    decoder.advance(decoder.nextTokens()[0])

print(f"{decoder.prefix()} -> {list(decoder.nextTokens())}")
```

Expected output:

```python
#  -> ['C']
# C -> ['C']
# CC -> ['(']
# CC( -> ['=']
# CC(= -> ['O']
# CC(=O -> [')']
# CC(=O) -> ['O']
# CC(=O)O -> ['c']
# CC(=O)Oc -> ['1']
# CC(=O)Oc1 -> ['c']
# CC(=O)Oc1c -> ['(', 'c']
```

The decoder is online. It does not precompute one fixed trajectory. At each
step it exposes the legal next tokens for the current emitted prefix.

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
