---
layout: page
title: grimace-py
---

`grimace-py` is a Rust-first RDKit add-on for exact rooted SMILES support
enumeration, online next-token decoding, and seeded sampling.

Use it when you need Grimace's supported random-writer language for a molecule,
exact legal next tokens while constructing a SMILES string, or one seeded legal
path with its per-step choices.

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

<table>
  <tbody>
    <tr>
      <td>Install and run the first enumeration/decoder examples</td>
      <td><a href="getting-started.html">Intro</a></td>
    </tr>
    <tr>
      <td>Understand support, roots, decoder tokens, and writer parity</td>
      <td><a href="concepts.html">Concepts</a></td>
    </tr>
    <tr>
      <td>Choose supported flags and root behavior</td>
      <td><a href="runtime.html">Runtime</a></td>
    </tr>
    <tr>
      <td>Check current gaps and supported release assets</td>
      <td><a href="current-limitations.html">Limitations</a></td>
    </tr>
    <tr>
      <td>Use prepared molecules, deviation diagnostics, or token inventories</td>
      <td><a href="guides.html">Guides</a></td>
    </tr>
    <tr>
      <td>Compare measured benchmark snapshots</td>
      <td><a href="timings.html">Timings</a></td>
    </tr>
    <tr>
      <td>Look up signatures and return objects</td>
      <td><a href="api/python.html">API</a></td>
    </tr>
    <tr>
      <td>Inspect checked-in RDKit evidence and counts</td>
      <td><a href="testing-fixtures.html">Testing fixtures</a></td>
    </tr>
    <tr>
      <td>Trace upstream RDKit serializer coverage</td>
      <td><a href="rdkit-serializer-coverage.html">RDKit serializer coverage</a></td>
    </tr>
    <tr>
      <td>Work on the codebase in containers</td>
      <td><a href="development/containerized.html">Containerized development</a></td>
    </tr>
    <tr>
      <td>Change internals</td>
      <td><a href="architecture/rust-first.html">Rust-first architecture</a></td>
    </tr>
  </tbody>
</table>

## License

`grimace-py` is source-available under the
[PolyForm Noncommercial 1.0.0](https://github.com/numpde/grimace-py/blob/main/LICENSE).
Commercial use is not permitted under the current license.
