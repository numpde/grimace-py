# GRIMACE

SMILES enumeration with exact next-token decoding.

`grimace` is a Rust-first cheminformatics library for RDKit molecules. It
answers two questions that RDKit does not expose directly:

- What is the exact rooted SMILES support of this molecule under a given
  RDKit-style writer regime?
- Given the prefix emitted so far, what tokens are legal next continuations?

By "support" we mean the full set of reachable rooted SMILES strings for the
chosen writer flags. A "rooted SMILES" here is a SMILES string generated with a
fixed starting atom.

`grimace` can:

- enumerate that exact support
- expose the exact token inventory implied by that support
- decode online, one token at a time, from a current prefix

The reason this library exists is that RDKit does not provide either:

- an exact rooted SMILES enumeration routine
- an online next-token decoding API for SMILES prefixes

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

## Choose The API

The only supported public Python package is `grimace`.

Main entrypoints:

- `MolToSmilesEnum(...)`
  Returns the exact SMILES support as an iterator of finished strings.
- `MolToSmilesDecoder(...)`
  Returns an online branch-preserving decoder state.
- `MolToSmilesDeterminizedDecoder(...)`
  Returns an online decoder that merges same-text next choices.
- `MolToSmilesTokenInventory(...)`
  Returns the exact set of tokens that can appear in one decoder step.

Supporting public type:

- `MolToSmilesChoice`
  Each choice has `.text` for the emitted token and `.next_state` for the
  decoder state after taking that token.

The public API uses the compiled Rust extension end to end.

## Important Runtime Requirements Today

The public signatures mirror RDKit flag names and defaults, but the current
runtime intentionally supports only a strict subset.

Today, pass:

- `canonical=False`
- `doRandom=True`
- `rootedAtAtom=-1` or `rootedAtAtom >= 0`

Unsupported combinations fail fast with `NotImplementedError`.

The most important `rootedAtAtom` semantics are:

- `rootedAtAtom=<idx>` uses one explicit starting atom.
- `rootedAtAtom=-1` for `MolToSmilesEnum(...)` returns the exact support
  unioned across all root atoms.
- `rootedAtAtom=-1` for the decoder classes starts from one merged all-roots
  decoder state.
- `rootedAtAtom=None` for `MolToSmilesTokenInventory(...)` returns the token
  inventory unioned across all root atoms.

## Quickstart

All examples below use the current supported runtime subset:

```python
FLAGS = dict(
    canonical=False,
    doRandom=True,
)
```

### 1. Enumerate The Exact Support

If you want the exact support across all possible roots, `rootedAtAtom=-1` is
the simplest public entrypoint:

```python
from rdkit import Chem
import grimace

mol = Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O")

all_smiles = tuple(
    grimace.MolToSmilesEnum(
        mol,
        rootedAtAtom=-1,
        isomericSmiles=False,
        **FLAGS,
    )
)

assert len(all_smiles) == 304
```

If instead you want the exact support from one specific root atom, pass that
root explicitly:

```python
root_0_smiles = tuple(
    grimace.MolToSmilesEnum(
        mol,
        rootedAtAtom=0,
        isomericSmiles=False,
        **FLAGS,
    )
)
```

### 2. Decode Online, One Token At A Time

`MolToSmilesDecoder(...)` is branch-preserving. It exposes the exact legal next
choices for the current prefix, and each choice points to a successor state.

```python
decoder = grimace.MolToSmilesDecoder(
    mol,
    rootedAtAtom=0,
    isomericSmiles=False,
    **FLAGS,
)

for _ in range(7):
    prefix = decoder.prefix if decoder.prefix else '""'
    print(f"{prefix} -> {[choice.text for choice in decoder.next_choices]}")
    decoder = decoder.next_choices[0].next_state
```

Early output on aspirin looks like:

```text
"" -> ['C']
C -> ['C']
CC -> ['(', '(']
CC( -> ['=']
CC(= -> ['O']
CC(=O -> [')']
CC(=O) -> ['O']
```

Notice the duplicate `"("` at `CC`. Those are different branches with the same
emitted token text. That is deliberate: `MolToSmilesDecoder(...)` preserves
branch identity instead of merging it away.

### 3. Merge Same-Text Choices When You Only Care About Token Text

`MolToSmilesDeterminizedDecoder(...)` exposes at most one choice per token text
by merging same-text continuations into one combined state.

For example, the merged all-roots decoder can trace one exact route to
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
    **FLAGS,
)

for token in route:
    choices = {choice.text: choice.next_state for choice in decoder.next_choices}
    decoder = choices[token]

assert decoder.is_terminal
assert decoder.prefix == "".join(route)
```

The first few merged decisions on that route are:

- `""`: choose `"c"` from `["C", "O", "c"]`
- `"c1"`: choose `"("` from `["(", "c"]`
- `"c1("`: choose `"c"` from `["O", "c", "C"]`
- `"c1(ccccc1"`: choose `"O"` from `["C", "O"]`

### 4. Ask For The Exact Token Inventory

`MolToSmilesTokenInventory(...)` answers a different question: not "what full
strings are possible?" but "what one-step tokens can ever appear?"

```python
inventory = grimace.MolToSmilesTokenInventory(
    mol,
    rootedAtAtom=None,
    isomericSmiles=False,
    **FLAGS,
)

assert "C" in inventory
assert "(" in inventory
assert "c" in inventory
```

The result is a sorted tuple of distinct tokens.

### What Counts As A Token?

A token is one string emitted by one decoder transition. Tokens are defined by
the walker itself, not by splitting a finished SMILES into characters and not
by integer token IDs. They come from two places:

- the prepared graph's RDKit-style atom and bond tokens, such as `C`, `c`,
  `Cl`, `[C@H]`, `=`, `/`, or `\\`
- SMILES syntax literals inserted by the walker, such as `(`, `)`, `1`, or
  `%10`

So a token is exactly one appendable SMILES fragment for the current state. It
may be one character or several.

## Installation

`grimace` currently publishes Linux `x86_64` wheels for CPython `3.12` and
`3.13`. Other supported Python versions and platforms currently require a
source build.

Install with `pip install <wheel>` using one of these release assets:

| System | 3.12 | 3.13 |
| --- | --- | --- |
| Linux x86_64 | [wheel](https://github.com/numpde/grimace-py/releases/download/v0.1.4/grimace-0.1.4-cp312-cp312-manylinux_2_28_x86_64.whl) | [wheel](https://github.com/numpde/grimace-py/releases/download/v0.1.4/grimace-0.1.4-cp313-cp313-manylinux_2_28_x86_64.whl) |

The built package depends on `rdkit>=2026.3`.

For local development:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install maturin
maturin develop --release
```

## Timings

The opt-in timing benchmark generates two artifacts:

- [docs/timings.tsv](docs/timings.tsv): raw measured summary data
- [docs/timings.md](docs/timings.md): rendered table and column descriptions

The table reports both decoder variants:

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

## Current Limits

The public API mirrors RDKit `MolToSmiles` flag names, but only a strict subset
is implemented today.

Required runtime values today:

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

## License

`grimace` is source-available under [PolyForm Noncommercial 1.0.0](LICENSE).
Third-party components remain under their own licenses; see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
Commercial use requires a separate commercial license from the author.
The software is provided as is, without warranty or liability, to the extent
allowed by law.
