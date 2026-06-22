# Grimace

Grimace is a Rust-first Python package for exact SMILES generation and online
next-token decoding.

The vision is simple: make molecular string generation inspectable. A model,
search procedure, or human-in-the-loop tool should be able to ask not only
"what SMILES strings can this molecule produce?", but also "what can come next,
right now, and why?"

Grimace treats SMILES serialization as a finite language problem. It builds the
reachable writer state space, exposes exact continuations, and keeps enough
structure to diagnose where a candidate string leaves the supported language.

## What It Is For

- exact rooted SMILES support enumeration
- online legal-next-token decoding
- branch-preserving or determinized decoder states
- token inventory construction for molecular language models
- diagnostics for rejected SMILES candidates

The public runtime is currently an RDKit writer-parity interface. Grimace keeps
that layer separate from principled chemistry semantics: a string can be
chemically valid while still not belonging to the writer language being modeled.

## Install

Install the PyPI distribution named `grimace-py`:

```bash
python -m pip install grimace-py
```

Import it as `grimace`:

```python
import grimace
```

Plain `pip install grimace` installs an unrelated package.

Grimace depends on RDKit and a compiled Rust extension. The package metadata
declares Python `>=3.11` and `rdkit>=2026.3`.

## Quick Taste

```python
from rdkit import Chem
import grimace

mol = Chem.MolFromSmiles("CCO")
flags = dict(canonical=False, doRandom=True)

support = tuple(
    grimace.MolToSmilesEnum(
        mol,
        rootedAtAtom=-1,
        isomericSmiles=False,
        **flags,
    )
)

decoder = grimace.MolToSmilesDeterminizedDecoder(
    mol,
    rootedAtAtom=-1,
    isomericSmiles=False,
    **flags,
)

while not decoder.is_terminal:
    choice = decoder.next_choices[0]
    print(decoder.prefix, "->", [item.text for item in decoder.next_choices])
    decoder = choice.next_state
```

Core public entrypoints:

- `MolToSmilesEnum(...)`
- `MolToSmilesDecoder(...)`
- `MolToSmilesDeterminizedDecoder(...)`
- `MolToSmilesDeviation(...)`
- `MolToSmilesTokenInventory(...)`
- `MolToSmilesTokenInventorySuperset(...)`

## Direction

Grimace is being built toward proof-carrying molecular serialization:

- exact writer languages instead of sampled writer behavior
- resumable decoder states instead of opaque string generation
- explicit graph, ring, and stereo obligations instead of post-hoc filtering
- diagnostics that point to the operation that made a candidate impossible

The name stands for "graph representation integrating multiple alternate
chemical equivalents", motivated by research on NMR spectroscopy with language
transformers ([link](https://numpde.github.io/shared/msc/)).

## More

- [Python API](docs/api/python.md)
- [Correctness contracts](docs/correctness-contracts.md)
- [Testing fixtures](docs/testing-fixtures.md)

## License

`grimace` is source-available under [PolyForm Noncommercial 1.0.0](LICENSE).
Commercial use requires a separate commercial license from the author.
Third-party components remain under their own licenses; see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
