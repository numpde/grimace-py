# GRIMACE

SMILES enumeration with exact next-token decoding.

`grimace` is a Rust-first cheminformatics library for exact rooted SMILES
enumeration, exact token inventory extraction, and online next-token decoding
from RDKit molecules. It computes the full rooted SMILES support of a molecule
under an RDKit-style writing regime, and it can also step through that support
one token at a time: at each prefix it exposes the legal next choices, then
advances when you choose one. By "support" we mean the full set of reachable
rooted SMILES strings for the given molecule and writer flags.

The reason this library exists is that RDKit does not provide a deterministic
SMILES enumeration routine, and it does not expose the legal continuations of a
SMILES prefix as an online decoding API.

`grimace` targets the current stable RDKit writer convention, currently
`RDKit 2026.03.1`. Older slash/backslash serialization conventions are out of
scope.

It requires Python `>=3.11` and `rdkit>=2026.3`.

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
- `MolToSmilesDeterminizedDecoder(...)`
- `MolToSmilesTokenInventory(...)`

Supporting public result type:

- `MolToSmilesChoice`

The public API uses the compiled Rust extension end to end.

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
    prefix = decoder.prefix if decoder.prefix else '""'
    print(f"{prefix} -> {[choice.text for choice in decoder.next_choices]}")
    decoder = decoder.next_choices[0].next_state
```

Expected output from that exact snippet:

```text
"" -> ['C']
C -> ['C']
CC -> ['(', '(']
CC( -> ['=']
CC(= -> ['O']
CC(=O -> [')']
CC(=O) -> ['O']
CC(=O)O -> ['c']
CC(=O)Oc -> ['1', '1']
CC(=O)Oc1 -> ['c']
CC(=O)Oc1c -> ['c']
CC(=O)Oc1cc -> ['c']
CC(=O)Oc1ccc -> ['c']
CC(=O)Oc1cccc -> ['c']
CC(=O)Oc1ccccc -> ['1']
CC(=O)Oc1ccccc1 -> ['C']
CC(=O)Oc1ccccc1C -> ['(', '(']
CC(=O)Oc1ccccc1C( -> ['=']
CC(=O)Oc1ccccc1C(= -> ['O']
CC(=O)Oc1ccccc1C(=O -> [')']
CC(=O)Oc1ccccc1C(=O) -> ['O']
```

This transcript follows `next_choices[0]` at each step. The decoder is online.
It does not precompute one fixed trajectory. At each step it exposes the legal
next choices for the current emitted prefix.

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

`MolToSmilesDeterminizedDecoder(...)` is the merged alternative. It exposes at
most one choice per token text by merging same-text continuations into one
combined successor state. Use it when you want prefix-level next-token choices
without preserving branch identity.

For example, the unrooted determinized decoder can trace one exact route to
`c1(ccccc1OC(=O)C)C(O)=O` for aspirin:

```python
route = [
    "c", "1", "(", "c", "c", "c", "c", "c", "1",
    "O", "C", "(", "=", "O", ")", "C", ")",
    "C", "(", "O", ")", "=", "O",
]

decoder = grimace.MolToSmilesDeterminizedDecoder(
    mol,
    rootedAtAtom=-1,
    isomericSmiles=False,
    canonical=False,
    doRandom=True,
)

for token in route:
    choices = {choice.text: choice.next_state for choice in decoder.next_choices}
    decoder = choices[token]

assert decoder.is_terminal
assert decoder.prefix == "".join(route)
```

The merged early decisions on that route are:

- `""`: choose `"c"` from `["C", "O", "c"]`
- `"c1"`: choose `"("` from `["(", "c"]`
- `"c1("`: choose `"c"` from `["O", "c", "C"]`
- `"c1(ccccc1"`: choose `"O"` from `["C", "O"]`

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

`grimace` currently publishes Linux `x86_64` wheels for CPython `3.12` and
`3.13`. Other supported Python versions and platforms currently require a
source build.

Install with `pip install <wheel>` using one of these release assets:

| System | 3.12 | 3.13 |
| --- | --- | --- |
| Linux x86_64 | [wheel](https://github.com/numpde/grimace-py/releases/download/v0.1.3/grimace-0.1.3-cp312-cp312-manylinux_2_28_x86_64.whl) | [wheel](https://github.com/numpde/grimace-py/releases/download/v0.1.3/grimace-0.1.3-cp313-cp313-manylinux_2_28_x86_64.whl) |

The built package depends on `rdkit>=2026.3`.

For local development:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install maturin
maturin develop --release
```

## Timings

The opt-in timing benchmark now generates two artifacts:

- [docs/timings.tsv](docs/timings.tsv): raw measured summary data
- [docs/timings.md](docs/timings.md): rendered table and column descriptions

The table currently reports both decoder variants:

- branch-preserving exhaustive traversal via `MolToSmilesDecoder(...)`
- determinized exhaustive traversal via `MolToSmilesDeterminizedDecoder(...)`

Current takeaway from the generated table:

- `MolToSmilesEnum(...)` is still the fastest exact route in every listed case
- `MolToSmilesDeterminizedDecoder(...)` can reduce exhaustive decoder cost on
  some molecules, but it is still slower than direct exact enumeration here

Regenerate it with:

```bash
RUN_PERF_TESTS=1 PYTHONPATH=python:. .venv/bin/python -m unittest tests.perf.test_readme_timings -q
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

Disconnected molecules are supported by the public APIs.

- `MolToSmilesEnum(...)`, `MolToSmilesDecoder(...)`, and
  `MolToSmilesDeterminizedDecoder(...)` compose fragment-wise behavior
  directly.
- `MolToSmilesTokenInventory(...)` returns the union of fragment inventories
  plus the `"."` separator token.

## Docs

- [Docs index](docs/README.md)
- [Python API](docs/api/python.md)
