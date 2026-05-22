Rust-backed PreparedMol status
==============================

Public shape
------------

The public cache/reuse surface is:

```python
prepared = grimace.PrepareMol(mol)
payload = prepared.to_bytes()
restored = grimace.PreparedMol.from_bytes(payload)
```

`PrepareMol` accepts an RDKit molecule and writer-surface flags. `PreparedMol`
is an opaque public wrapper. Its public contract is runtime acceptance and byte
round trips.

Implemented boundary
--------------------

- Python `PrepareMol` is the RDKit boundary.
- Python `PreparedMol` wraps a Rust `_core.PreparedMol`.
- Runtime operations normalize RDKit input through `PrepareMol`.
- Runtime operations consume `PreparedMol` without late RDKit preparation.
- Writer-surface flags are baked into `PreparedMol`.
- Runtime calls reject writer-flag mismatches.
- `to_bytes()` and `from_bytes()` use a versioned Rust-owned binary payload.

The current single-object byte payload starts with the `GPM\0` magic prefix.
That prefix identifies the payload family before Rust decodes the versioned
body. The field layout remains internal.

Accepted runtime inputs
-----------------------

`PreparedMol` is accepted by:

- `MolToSmilesEnum`
- `MolToSmilesDecoder`
- `MolToSmilesDeterminizedDecoder`
- `MolToSmilesTokenInventory`
- `MolToSmilesTokenInventorySuperset`
- `MolToSmilesDeviation`

Tests
-----

The test suite covers:

- connected and disconnected prepared molecules
- stereo and writer-flag surfaces
- byte round trips
- malformed payload rejection
- writer-flag mismatch rejection
- runtime equivalence between RDKit input and byte-round-tripped `PreparedMol`
- public runtime failure when RDKit input cannot be prepared
- successful prepared runtime after RDKit preparation helpers are poisoned
- direct runtime-source checks against RDKit imports outside the preparation
  boundary

Still separate
--------------

Bulk dataset storage is not part of this object. A future bulk format can pack
many prepared molecules and add indexing/compression, but it should build on
this single-object contract rather than redefine it.
