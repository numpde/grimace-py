# Third-Party Notices

This project is published under `PolyForm-Noncommercial-1.0.0`. Third-party
software used by `grimace` remains under its own license terms.

This file is the minimal notice set for the current shipped surface:

- the external Python runtime dependency on RDKit
- the copied/adapted RDKit-derived test cases in `tests/rdkit_serialization/`
- the Rust crates linked into the compiled extension module

## RDKit

`grimace` depends on RDKit as an external Python dependency. RDKit is not
authored by this project and remains under its own permissive license:

- project: <https://github.com/rdkit/rdkit>
- license: `BSD-3-Clause`

Some tests under `tests/rdkit_serialization/` are derived from local RDKit
serialization tests. Their provenance is called out in the test files
themselves, and the underlying RDKit material remains under the RDKit license.

## Rust crates linked into the extension

The compiled `_core` extension is built from `grimace-kernel` plus the Rust
dependency graph resolved by Cargo. The list below comes from
`cargo metadata --format-version 1 --locked` for the current release.

| Crate | Version | License |
| --- | ---: | --- |
| `heck` | `0.5.0` | `MIT OR Apache-2.0` |
| `libc` | `0.2.183` | `MIT OR Apache-2.0` |
| `once_cell` | `1.21.4` | `MIT OR Apache-2.0` |
| `portable-atomic` | `1.13.1` | `Apache-2.0 OR MIT` |
| `proc-macro2` | `1.0.106` | `MIT OR Apache-2.0` |
| `pyo3-build-config` | `0.28.2` | `MIT OR Apache-2.0` |
| `pyo3-ffi` | `0.28.2` | `MIT OR Apache-2.0` |
| `pyo3-macros-backend` | `0.28.2` | `MIT OR Apache-2.0` |
| `pyo3-macros` | `0.28.2` | `MIT OR Apache-2.0` |
| `pyo3` | `0.28.2` | `MIT OR Apache-2.0` |
| `quote` | `1.0.45` | `MIT OR Apache-2.0` |
| `syn` | `2.0.117` | `MIT OR Apache-2.0` |
| `target-lexicon` | `0.13.5` | `Apache-2.0 WITH LLVM-exception` |
| `unicode-ident` | `1.0.24` | `(MIT OR Apache-2.0) AND Unicode-3.0` |

Upstream repositories:

- PyO3 family (`pyo3`, `pyo3-ffi`, `pyo3-macros`, `pyo3-macros-backend`, `pyo3-build-config`):
  <https://github.com/PyO3/pyo3>
- `proc-macro2`, `quote`, `syn`, `unicode-ident`:
  <https://github.com/dtolnay>
- `libc`: <https://github.com/rust-lang/libc>
- `once_cell`: <https://github.com/matklad/once_cell>
- `portable-atomic`: <https://github.com/taiki-e/portable-atomic>
- `target-lexicon`: <https://github.com/bytecodealliance/target-lexicon>
- `heck`: <https://github.com/withoutboats/heck>

## Build tooling

This repository also uses `maturin` as a build tool. It is not part of the
runtime package, but for completeness:

- project: <https://github.com/PyO3/maturin>
- license: `MIT OR Apache-2.0`
