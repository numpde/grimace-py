# GRIMACE

SMILES enumeration with next-token distribution.

`grimace` is a Rust-first cheminformatics library for exact rooted SMILES
enumeration and online next-token decoding from RDKit molecules. It computes
the full rooted SMILES support of a molecule under an RDKit-style writing
regime, and it can also step through that support one token at a time: at each
prefix it exposes the legal next tokens, then advances when you choose one.
By "support" we mean the full set of reachable rooted SMILES strings for the
given molecule and writer flags.

The reason this library exists is that RDKit does not provide a deterministic
SMILES enumeration routine, and it does not expose the legal continuations of a
SMILES prefix as an online decoding API.

GRIMACE stands for "graph representation integrating multiple alternate
chemical equivalents", motivated by research on NMR spectroscopy
with language transformers ([link](https://numpde.github.io/shared/msc/)).

> [!WARNING]
> This library is work in progress. Expect API changes, incomplete feature
> coverage, and rough edges between releases.

## Public API

The only supported public Python package is `grimace`.

Current entrypoints:

- `MolToSmilesEnum(...)`
- `MolToSmilesDecoder(...)`
- `MolToSmilesTokenInventory(...)`

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

Here a "token" means one string emitted by one decoder transition. Tokens are
defined by the walker itself, not by splitting a finished SMILES into
characters and not by integer token IDs. They come from two places:

- the prepared graph's RDKit-style atom and bond tokens, such as `C`, `c`,
  `Cl`, `[C@H]`, `=`, `/`, or `\\`
- SMILES syntax literals inserted by the walker, such as `(`, `)`, `1`, or
  `%10`

So a token is exactly one appendable SMILES fragment for the current state. It
may be one character or several.

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

## Timings

Example timings from the opt-in performance benchmark, measured in release mode
on one development machine. Treat them as indicative, not as a portability or
stability guarantee.

- `Support`: the size of the exact rooted SMILES support across all root atoms.
- `Grimace enum (all roots)`: direct exact enumeration with `MolToSmilesEnum(...)`,
  unioned across all roots.
- `Decoder enum (all roots)`: exact enumeration by exhaustive traversal of
  `MolToSmilesDecoder(...)`, unioned across all roots.
- `RDKit to 1/2 support`: repeated RDKit `MolToSmiles(..., canonical=False,
  doRandom=True)` draws across all roots until half of the exact support has
  been seen.
- `RDKit to full support`: the same sampling process until the full exact
  support has been seen.
- All timing columns are shown as `time mean ± std`.
- The two RDKit columns also show `(draw mean ± std)` over repeated seeded
  trials.

| Canonical SMILES | Atoms | Support | Grimace enum (all roots) | Decoder enum (all roots) | RDKit to 1/2 support | RDKit to full support |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| `CC(=O)Oc1ccccc1C(=O)O` | 13 | 304 | **15.3** ± 1.0 ms | **34.1** ± 2.4 ms | **4.7** ± 0.4 ms (230.0 ± 18.8 draws) | **56.2** ± 15.2 ms (3086.7 ± 921.8 draws) |
| `C1CC2(CCO1)CO2` | 8 | 36 | **3.1** ± 0.2 ms | **5.3** ± 0.2 ms | **0.3** ± 0.0 ms (23.0 ± 1.8 draws) | **1.9** ± 0.4 ms (155.6 ± 35.8 draws) |
| `CN1CCC[C@H]1c1cccnc1` | 12 | 136 | **14.3** ± 0.3 ms | **22.2** ± 0.5 ms | **1.8** ± 0.2 ms (97.4 ± 8.7 draws) | **18.0** ± 3.1 ms (987.9 ± 169.9 draws) |
| `CNC(=O)O/N=C(\C)SC` | 10 | 72 | **17.5** ± 0.1 ms | **19.8** ± 0.1 ms | **0.6** ± 0.0 ms (44.1 ± 2.5 draws) | **6.0** ± 1.5 ms (483.0 ± 122.3 draws) |
| `N[C@@H](Cc1ccc(O)c(O)c1)C(=O)O` | 14 | 688 | **50.4** ± 2.2 ms | **96.8** ± 4.2 ms | **10.1** ± 0.4 ms (514.3 ± 12.9 draws) | **150.6** ± 45.9 ms (7946.7 ± 2448.6 draws) |
| `COc1ccc2cc([C@H](C)C(=O)O)ccc2c1` | 17 | 1504 | **111.6** ± 1.4 ms | **219.9** ± 1.5 ms | **26.5** ± 0.7 ms (1143.0 ± 34.0 draws) | **570.1** ± 115.7 ms (24406.3 ± 4916.2 draws) |

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
