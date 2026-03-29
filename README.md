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

all_smiles = {
    smiles
    for root_atom in range(mol.GetNumAtoms())
    for smiles in grimace.MolToSmilesEnum(
        mol,
        rootedAtAtom=root_atom,
        isomericSmiles=False,
        canonical=False,
        doRandom=True,
    )
}
```

Then `len(all_smiles) == 304`.


```python
decoder = grimace.MolToSmilesDecoder(
    mol,
    rootedAtAtom=0,
    isomericSmiles=False,
    canonical=False,
    doRandom=True,
)

while not decoder.is_terminal:
    print(f"{decoder.prefix} -> {list(decoder.next_tokens)}")
    decoder.advance(decoder.next_tokens[0])
```

Expected output:

```text
-> ['C']
C -> ['C']
CC -> ['(']
CC( -> ['=', 'O']
CC(= -> ['O']
CC(=O -> [')']
...
CC(=O)Oc1 -> ['c']
CC(=O)Oc1c -> ['(', 'c']
CC(=O)Oc1c( -> ['C', 'c']
CC(=O)Oc1c(C -> ['(']
CC(=O)Oc1c(C( -> ['=', 'O']
CC(=O)Oc1c(C(= -> ['O']
CC(=O)Oc1c(C(=O -> [')']
...
CC(=O)Oc1c(C(=O)O)cccc -> ['1']
```

The decoder is online. It does not precompute one fixed trajectory. At each
step it exposes the legal next tokens for the current emitted prefix.

## pip install ...

Install with `pip install <wheel>` with one of those (download or use the link directly):

| System | 3.12 | 3.13 |
| --- | --- | --- |
| Linux x86_64 | [wheel](https://github.com/numpde/grimace-py/releases/download/v0.1.0/grimace-0.1.0-cp312-cp312-manylinux_2_28_x86_64.whl) | [wheel](https://github.com/numpde/grimace-py/releases/download/v0.1.0/grimace-0.1.0-cp313-cp313-manylinux_2_28_x86_64.whl) |

For local development:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install maturin
maturin develop --release
```

## License

`grimace` is source-available under [PolyForm Noncommercial 1.0.0](LICENSE).
Commercial use requires a separate commercial license from the author.
The software is provided as is, without warranty or liability, to the extent
allowed by law.

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
