---
layout: page
title: grimace-py
---

`grimace-py` is a Rust-first RDKit add-on for exact rooted SMILES support
enumeration and online next-token decoding.

Install the distribution named `grimace-py`, then import `grimace`:

```bash
python -m pip install grimace-py
```

```python
import grimace
```

Plain `pip install grimace` installs an unrelated older package.

The current public runtime targets RDKit writer parity for
`canonical=False, doRandom=True` under the documented writer flags.

## Start here

- [Getting started](getting-started.md)
- [Runtime requirements](runtime.md)
- [Python API](api/python.md)

## Guides

- [Prepared molecules](guides/prepared-mol.md)
- [Deviation diagnostics](guides/deviation.md)
- [Token inventories](guides/token-inventory.md)

## Correctness and evidence

- [Correctness contracts](correctness-contracts.md)
- [Testing fixtures](testing-fixtures.md)
- [RDKit serializer coverage](rdkit-serializer-coverage.md)
- [Timings](timings.md)

## Development

- [Containerized development](development/containerized.md)
- [Rust-first architecture](architecture/rust-first.md)

## License

`grimace-py` is source-available under the
[PolyForm Noncommercial 1.0.0](https://github.com/numpde/grimace-py/blob/main/LICENSE).
Commercial use is not permitted under the current license.
