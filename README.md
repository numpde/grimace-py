# GRIMACE

SMILES enumeration with exact next-token decoding.

`grimace` is a Rust-first RDKit add-on focused on exact rooted SMILES support
and exact next-token decoding. In its current public form, it answers two
questions that RDKit does not expose directly:

- What is the exact rooted SMILES support of this molecule under a given
  RDKit-style writer regime?
- Given the prefix emitted so far, what tokens are legal next continuations?

By "support" we mean the full set of reachable rooted SMILES strings for the
chosen writer flags. A "rooted SMILES" here is a SMILES string generated with a
fixed starting atom for a connected molecule, or with one rooted fragment/local
root inside the preserved fragment order for a disconnected molecule.

`grimace` can:

- enumerate that exact support
- expose the exact token inventory implied by that support
- decode online, one token at a time, from a current prefix

Today, that public runtime is intentionally narrow: exact support and decoding
for RDKit's `canonical=False, doRandom=True` writer regime under the current
stable writer convention.

There are two separate correctness ideas in this project:

- principled SMILES/chemistry semantics: emitted strings should be valid and
  parse back to the intended graph and stereo assignment
- RDKit writer parity: emitted strings should match RDKit's actual writer
  support for the supported regime

The current public API is an RDKit writer-parity API. Some SMILES strings may
be semantically valid and parse to the same molecule while still not belonging
to RDKit's writer support. Those cases must be documented and tested as a
separate semantic layer, not silently mixed into RDKit-parity claims.

The reason this library exists is that RDKit does not provide either:

- an exact rooted SMILES enumeration routine
- an online next-token decoding API for SMILES prefixes

`grimace` targets the current stable RDKit writer convention, currently
`RDKit 2026.03.1`. Older slash/backslash serialization conventions are out of
scope. The dependency floor is `rdkit>=2026.3`, but exact output parity is
only validated against that current stable writer convention; newer RDKit
releases may still require fixture or expectation updates.

The package metadata declares Python `>=3.11` and `rdkit>=2026.3`. The
currently exercised CI and release matrix is narrower and documented below.

GRIMACE stands for "graph representation integrating multiple alternate
chemical equivalents", motivated by research on NMR spectroscopy
with language transformers ([link](https://numpde.github.io/shared/msc/)).

> [!WARNING]
> `grimace` is still evolving. The supported public API is usable for the
> documented runtime subset, but feature coverage is still limited and some
> public details may continue to change between releases.

> [!IMPORTANT]
> `grimace` is distributed under `PolyForm-Noncommercial-1.0.0`. Commercial
> use is not permitted under the current license.

## Choose the API

The only supported public Python import name is `grimace`.

> [!CAUTION]
> Install the PyPI distribution named `grimace-py`; import the package as
> `grimace`. Plain `pip install grimace` installs an unrelated older package,
> not this library.
>
> ```bash
> python -m pip install grimace-py
> ```

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

## Important runtime requirements today

The public signatures mirror RDKit flag names and defaults, but the current
runtime intentionally supports only a strict subset.

> [!CAUTION]
> The signatures preserve RDKit-like defaults for surface compatibility, but
> those defaults are not currently supported. A naive
> `grimace.MolToSmilesEnum(mol)` call raises `NotImplementedError`; pass
> `canonical=False` and `doRandom=True` explicitly.

Today, pass:

- `canonical=False`
- `doRandom=True`
- omit `rootedAtAtom` or pass `rootedAtAtom=-1` for all-roots behavior
- pass `rootedAtAtom >= 0` for one explicit root
- other negative integer `rootedAtAtom` values are also accepted for RDKit
  compatibility and behave like `-1`, but `-1` is the preferred public
  spelling
- `rootedAtAtom=None` is not supported; omit the argument or use `-1` instead

Unsupported flag combinations fail fast with `NotImplementedError`. Other
invalid public inputs can still raise more specific exceptions such as
`IndexError` or `ValueError`.

The most important `rootedAtAtom` semantics are:

- `rootedAtAtom=<idx>` uses one explicit starting atom for connected
  molecules.
- `rootedAtAtom=-1` for `MolToSmilesEnum(...)` returns the exact support
  unioned across all root atoms.
- `rootedAtAtom=-1` for the decoder classes starts from one merged all-roots
  decoder state.
- `rootedAtAtom=-1` for `MolToSmilesTokenInventory(...)` returns the token
  inventory unioned across all root atoms.
- omitting `rootedAtAtom` means the same thing as passing `-1`.
- other negative integer `rootedAtAtom` values also behave like `-1`, to stay
  close to RDKit's public binding behavior.
- `rootedAtAtom=None` is rejected across the public API, matching RDKit's
  binding behavior.
- for disconnected molecules, fragment order is preserved; a nonnegative
  `rootedAtAtom` selects the rooted fragment and its local root atom within
  that fixed fragment order, but non-rooted fragments can still vary
  internally.

## Quickstart

All examples below use the current supported runtime subset:

```python
FLAGS = dict(
    canonical=False,
    doRandom=True,
)
```

### 1. Enumerate the exact support

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

### 2. Decode online, one token at a time

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

### 3. Merge same-text choices when you only care about token text

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

### 4. Ask for the exact token inventory

`MolToSmilesTokenInventory(...)` answers a different question: not "what full
strings are possible?" but "what one-step tokens can ever appear?"

```python
inventory = grimace.MolToSmilesTokenInventory(
    mol,
    rootedAtAtom=-1,
    isomericSmiles=False,
    **FLAGS,
)

assert "C" in inventory
assert "(" in inventory
assert "c" in inventory
```

The result is a sorted tuple of distinct tokens.

### What counts as a token?

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

Install the PyPI distribution named `grimace-py`:

```bash
python -m pip install grimace-py
```

Then import it as `grimace`:

```python
import grimace
```

Plain `pip install grimace` installs an unrelated older project with the same
name, not this library.

PyPI and GitHub release assets currently publish Linux `x86_64` wheels for
CPython `3.12` and `3.13`. Other environments may require a source build and
are not covered by the release wheels.

Current continuously exercised matrix:

- Linux source-tree tests on CPython `3.12`
- Linux wheel build and smoke tests on CPython `3.12` and `3.13`
- source distribution build plus `twine check` metadata validation

The published sdist is not currently installed and smoke-tested in CI as an
artifact. Treat it as a supported source-build path, but with weaker
continuous evidence than the Linux wheel path.

Other Python versions and non-Linux platforms are expected source-build paths,
not part of the current release asset or CI matrix.
Python `3.11` is in that source-build category today: declared, but not part of
the current CI matrix.

GitHub release wheels are also available:

| System | 3.12 | 3.13 |
| --- | --- | --- |
| Linux x86_64 | [wheel](https://github.com/numpde/grimace-py/releases/download/v0.1.9/grimace_py-0.1.9-cp312-cp312-manylinux_2_28_x86_64.whl) | [wheel](https://github.com/numpde/grimace-py/releases/download/v0.1.9/grimace_py-0.1.9-cp313-cp313-manylinux_2_28_x86_64.whl) |

The built package depends on `rdkit>=2026.3`.

For local development or a source build, you need:

- a Rust toolchain with `rustc >= 1.83`
- `maturin`

Then:

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

- the table does not time the direct public
  `MolToSmilesEnum(..., rootedAtAtom=-1)` path
- the published `Grimace enum` column is the more conservative exact baseline:
  explicit union over per-root `MolToSmilesEnum(..., rootedAtAtom=root_idx)`
  calls
- some merged decoder rows are numerically lower than that per-root union
  column, so this table does not prove a universal exact-method ranking
- `MolToSmilesDeterminizedDecoder(...)` can reduce exhaustive decoder cost on
  some molecules
- the table is still a small curated benchmark: 9 molecules, 2 writer modes,
  7 timing repeats, and one development machine
- this is not a workload study and not an exact-versus-exact comparison
- the `Grimace enum` row times explicit union over per-root
  `MolToSmilesEnum(..., rootedAtAtom=root_idx)` calls, not the direct public
  `MolToSmilesEnum(..., rootedAtAtom=-1)` path
- the RDKit columns are not exact enumeration; they are random sampling until
  RDKit happens to reach `1/2` or full support
- because of that, RDKit can be cheaper when you only want a few random
  strings, especially on small cases
- but on the larger molecules in this table, Grimace exact methods were
  usually much faster than this RDKit sampling-to-coverage baseline when
  guaranteed full support was the goal

Regenerate it with:

```bash
RUN_PERF_TESTS=1 PYTHONPATH=python:. .venv/bin/python -m unittest tests.perf.test_readme_timings -q
```

## Current limits

The public API keeps RDKit `MolToSmiles` flag names, but it does not aim for
full RDKit writer-surface parity yet.

For terminology and test policy, see
[Correctness contracts](docs/correctness-contracts.md). In short: RDKit
writer-matching behavior is intentionally separate from the principled
SMILES/chemistry semantics layer. RDKit-specific traversal and directional-bond
placement rules belong to the writer-parity layer, not the generic semantic
layer.

Known stereo writer-parity work in progress:

- `minimal_nonstereo_double_hazard`
  (`C/N=C1C=C/C(=N/C)[N-]/1`) currently has the same direction-erased
  skeleton support as RDKit, but some slash/backslash markers are placed in
  different writer positions.
- `reduced_porphyrin_traversal_coupling` currently exercises a larger
  traversal-coupling gap where RDKit's writer policy narrows the traversal
  assignment support beyond the semantic carrier choices.
- A separate RDKit-known quirk,
  `dative_carbonyl_stereo_annotation_drops_on_smiles_roundtrip`, is tracked as
  observed RDKit behavior rather than a generic SMILES semantics rule.

This work is active on the `stereo-constraint-model` branch and is not yet in
the released mainline runtime. The goal is exact RDKit writer support for the
documented runtime subset, without mixing RDKit-specific spelling quirks into
the generic semantic layer.

Current public runtime contract:

- `canonical=False`
- `doRandom=True`
- omit `rootedAtAtom` or pass `rootedAtAtom=-1` for all-roots behavior
- pass `rootedAtAtom >= 0` for one explicit root
- other negative integer `rootedAtAtom` values are also accepted for RDKit
  compatibility, but `-1` is the preferred public spelling
- `rootedAtAtom=None` is not supported; omit the argument or use `-1`

Supported writer flags today:

- `isomericSmiles`
- `kekuleSmiles`
- `allBondsExplicit`
- `allHsExplicit`
- `ignoreAtomMapNumbers`

Anything outside that runtime subset fails fast. Unsupported flag combinations
raise `NotImplementedError`. Other invalid public inputs can still raise more
specific exceptions such as `IndexError` or `ValueError`.

Disconnected molecules are supported by the public APIs.

- `MolToSmilesEnum(...)`, `MolToSmilesDecoder(...)`, and
  `MolToSmilesDeterminizedDecoder(...)` compose fragment-wise behavior
  directly.
- `MolToSmilesTokenInventory(...)` returns the union of fragment inventories
  plus the `"."` separator token.

## Docs

- [Docs index](docs/README.md)
- [Python API](docs/api/python.md)
- [Correctness contracts](docs/correctness-contracts.md)
- [Testing fixtures](docs/testing-fixtures.md)

## License

`grimace` is source-available under [PolyForm Noncommercial 1.0.0](LICENSE).
Third-party components remain under their own licenses; see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
Commercial use requires a separate commercial license from the author.
The software is provided as is, without warranty or liability, to the extent
allowed by law.
