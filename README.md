# GRIMACE

SMILES enumeration with next-token distribution.

`grimace` is a Rust-first cheminformatics library for exact rooted SMILES
enumeration and online next-token decoding from RDKit molecules. It computes
the full rooted SMILES support of a molecule under an RDKit-style writing
regime, and it can also step through that support one token at a time: at each
prefix it exposes the legal next choices, then advances when you choose one.
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
    print(f"{decoder.prefix} -> {[choice.text for choice in decoder.next_choices]}")
    decoder = decoder.next_choices[0].next_state
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
CC(=O)Oc1c -> ['(', '(']
CC(=O)Oc1c( -> ['C']
CC(=O)Oc1c(C -> ['(']
CC(=O)Oc1c(C( -> ['=', 'O']
CC(=O)Oc1c(C(= -> ['O']
CC(=O)Oc1c(C(=O -> [')']
...
CC(=O)Oc1c(C(=O)O)cccc -> ['1']
```

The decoder is online. It does not precompute one fixed trajectory. At each
step it exposes the legal next choices for the current emitted prefix.

The decoder is branch-preserving, not a determinized frontier decoder. Here
"branch-preserving" means `next_choices` may contain multiple choices with the
same `choice.text` when they correspond to different underlying continuations.
Those are semantically different choices because they carry different successor
states. By contrast, a "determinized frontier" decoder would merge all
same-text continuations into one combined successor state for that token.

GRIMACE intentionally preserves branch identity in the public decoder. Apparent
duplicates are therefore meaningful. This avoids hiding distinct continuations
behind one merged token choice and avoids pushing the cost of that implicit
determinization into the runtime.

As a consequence, two decoder states may share the same `prefix` but expose
different `next_choices`, depending on which earlier branch led there.

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
| Linux x86_64 | [wheel](https://github.com/numpde/grimace-py/releases/download/v0.1.2/grimace-0.1.2-cp312-cp312-manylinux_2_28_x86_64.whl) | [wheel](https://github.com/numpde/grimace-py/releases/download/v0.1.2/grimace-0.1.2-cp313-cp313-manylinux_2_28_x86_64.whl) |

For local development:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install maturin
maturin develop --release
```

## Timings

The opt-in timing benchmark generates a small markdown document with the
current table and a short explanation of the columns:

- [docs/timings.md](docs/timings.md)

Regenerate it with:

```bash
RUN_PERF_TESTS=1 PYTHONPATH=python:. python3 -m unittest tests.perf.test_readme_timings -q
```

## License

`grimace` is source-available under [PolyForm Noncommercial 1.0.0](LICENSE).
Third-party components remain under their own licenses; see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
Commercial use requires a separate commercial license from the author.
The software is provided as is, without warranty or liability, to the extent
allowed by law.

## Current limits

The public API mirrors RDKit `MolToSmiles` flag names, but only a strict subset
is implemented today:

- `rootedAtAtom == -1` or `rootedAtAtom >= 0`
- `canonical=False`
- `doRandom=True`

Supported writer flags:

- `isomericSmiles`
- `kekuleSmiles`
- `allBondsExplicit`
- `allHsExplicit`
- `ignoreAtomMapNumbers`

Unsupported combinations fail fast with `NotImplementedError`.

Disconnected molecules are supported by the public APIs. `MolToSmilesEnum(...)`
and `MolToSmilesDecoder(...)` compose fragment-wise behavior directly;
`MolToSmilesTokenInventory(...)` returns the union of fragment inventories plus
the `"."` separator token.

## Docs

- [Docs index](docs/README.md)
- [Python API](docs/api/python.md)
