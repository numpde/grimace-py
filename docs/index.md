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

## Where to start

| Task | Read |
|---|---|
| Install and run the first enumeration/decoder examples | [Intro](getting-started.md) |
| Understand support, roots, decoder tokens, and writer parity | [Concepts](concepts.md) |
| Choose supported flags and root behavior | [Runtime](runtime.md) |
| Check current gaps and supported release assets | [Limitations](current-limitations.md) |
| Use prepared molecules, deviation diagnostics, or token inventories | [Guides](guides.md) |
| Compare measured benchmark snapshots | [Timings](timings.md) |
| Look up signatures and return objects | [API](api/python.md) |
| Inspect checked-in RDKit evidence and counts | [Testing fixtures](testing-fixtures.md) |
| Trace upstream RDKit serializer coverage | [RDKit serializer coverage](rdkit-serializer-coverage.md) |
| Work on the codebase in containers | [Containerized development](development/containerized.md) |
| Change internals | [Rust-first architecture](architecture/rust-first.md) |

## License

`grimace-py` is source-available under the
[PolyForm Noncommercial 1.0.0](https://github.com/numpde/grimace-py/blob/main/LICENSE).
Commercial use is not permitted under the current license.
