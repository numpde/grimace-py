Prepared runtime boundary
=========================

Intent
------

The public Python API remains RDKit-first, but runtime execution should be
prepared-first:

```text
public RDKit Mol input -> PrepareMol -> PreparedMol fragments -> runtime/core
```

RDKit is allowed at preparation. It should not be needed after a public call has
normalized its input to `PreparedMol`.

Current invariant
-----------------

Public runtime operations normalize RDKit inputs through `PrepareMol` before
enumeration, decoding, inventories, or deviation diagnostics proceed.

Covered public operations:

- `MolToSmilesEnum`
- `MolToSmilesDecoder`
- `MolToSmilesDeterminizedDecoder`
- `MolToSmilesTokenInventory`
- `MolToSmilesTokenInventorySuperset`
- `MolToSmilesDeviation`

For `PreparedMol` inputs, runtime operations validate writer flags and consume
stored prepared fragments. They do not call `PrepareMol`.

Disconnected runtime planning now accepts only `PreparedMol` fragments. Public
RDKit input reaches that path only after `PrepareMol`.

Boundary tests
--------------

The behavioral tests check three things:

1. If `PrepareMol` is patched to raise, every public operation with RDKit input
   raises that exact error. This proves RDKit input cannot bypass preparation.
2. If `PrepareMol` is patched to raise, every public operation with a
   byte-round-tripped `PreparedMol` still works. This proves prepared input does
   not reprepare.
3. If RDKit fragmentation and reference graph preparation are poisoned
   immediately after `PrepareMol` returns, every public operation still
   completes. This proves there is no late RDKit/reference preparation after
   public normalization.

The source boundary test is narrower and cheaper: `_runtime.py` and
`_deviation.py` must not contain direct `rdkit` imports. RDKit-facing helpers
live in `_prepared_mol.py`, where preparation belongs.

Why imports, not every call site
--------------------------------

The import rule is deliberately simple. It is a direct-source check, not a
transitive import-graph claim, and it does not try to prove semantic correctness
by scanning for every possible RDKit-shaped string. The behavioral poison tests
prove the runtime contract; the import test prevents obvious boundary drift.

Current caveat
--------------

`_runtime.py` still contains lower-level compatibility paths for connected
prepared graph objects and reference prepared graph conversion. Those are not
the normal public RDKit path. The important boundary is that public RDKit input
normalizes through `PreparedMol`, and `_runtime.py` no longer imports RDKit
directly.

Next cleanup
------------

Once legacy/internal prepared graph entrypoints are no longer needed, runtime
helpers can be narrowed further to accept only `PreparedMol` plus connected core
prepared fragments. That would remove remaining mixed-shape branches from
`_runtime.py`.
