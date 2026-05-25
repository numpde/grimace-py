---
layout: page
title: grimace-py
---

`grimace-py` is a Rust-first RDKit add-on for exact rooted SMILES support
enumeration and online next-token decoding.

Use it when you need Grimace's supported random-writer language for a molecule,
or when you need exact legal next tokens while constructing a SMILES string.

GRIMACE stands for "graph representation integrating multiple alternate
chemical equivalents", motivated by research on NMR spectroscopy with
language transformers ([link](https://numpde.github.io/shared/msc/)).

Install the distribution named `grimace-py`, then import `grimace`:

```bash
python -m pip install grimace-py
```

```python
import grimace
```

Plain `pip install grimace` installs an unrelated older package.

Repository: [github.com/numpde/grimace-py](https://github.com/numpde/grimace-py).

See [Limitations](current-limitations.md) for supported runtime scope
and known gaps.

## Start here

- [Intro](getting-started.md)
- [Concepts](concepts.md)
- [Runtime](runtime.md)
- [Limitations](current-limitations.md)
- [Guides](guides.md)
- [Timings](timings.md)
- [API](api/python.md)

## Guides

[Guides](guides.md) cover prepared molecules, deviation diagnostics, and token
inventories.

## Correctness and evidence

- [Correctness contracts](correctness-contracts.md)
- [Testing fixtures](testing-fixtures.md)
- [RDKit serializer coverage](rdkit-serializer-coverage.md)

## Development

- [Containerized development](development/containerized.md)
- [Rust-first architecture](architecture/rust-first.md)

## License

`grimace-py` is source-available under the
[PolyForm Noncommercial 1.0.0](https://github.com/numpde/grimace-py/blob/main/LICENSE).
Commercial use is not permitted under the current license.
