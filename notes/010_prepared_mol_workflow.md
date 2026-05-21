PreparedMol workflow
====================

Purpose
-------

`PreparedMol` is the durable form of a molecule after Grimace has extracted the
writer surface it needs. Preparation may use RDKit. Reading and using a
`PreparedMol` should not.

The primary use case is large molecular datasets where the expensive RDKit
preparation step is paid once, then cached for fast enumeration, decoding,
token-inventory estimation, and diagnostics.

Semantics
---------

A `PreparedMol` is specific to a SMILES writer surface. These options are baked
into the prepared object:

- `isomericSmiles`
- `kekuleSmiles`
- `allBondsExplicit`
- `allHsExplicit`
- `ignoreAtomMapNumbers`

These options remain runtime choices:

- `rootedAtAtom`
- `canonical`
- `doRandom`

Disconnected molecules are part of the model. A prepared molecule stores ordered
prepared fragments, not just one connected graph. Each fragment keeps enough
identity to relate fragment-local graph data back to the original atom indices.

Design direction
----------------

The long-term representation should be Rust-native and cheap to load. Python
dict materialization is useful for tests and migration, but it should not define
the fast storage format.

The working shape is:

```text
PreparedMol
  schema_version
  writer_flags
  fragments[]

PreparedMolFragment
  atom_indices[]
  prepared_graph
```

`prepared_graph` should eventually be decoded directly into the Rust runtime
shape. A compact JSON or dict round-trip may be useful as a compatibility
bridge, but not as the performance target.

Workflow
--------

1. Add new-file-only design and tests while parallel runtime work is active.
   This can define the API contract without touching existing runtime files.

2. Prototype storage formats on real dataset samples. Compare bytes per
   molecule, read speed, allocation cost, and bulk compression. Include random
   and sequential samples.

3. Implement the Rust prepared representation and validation. The Rust layer
   should own the loaded shape, schema evolution, and fast decode path.

4. Add a small Python facade:

   ```python
   prepared = grimace.PrepareMol(mol, isomericSmiles=True)
   prepared.write(path)
   prepared = grimace.PreparedMol.read(path)
   ```

5. Wire `PreparedMol` into the public runtime APIs once conflicts are unlikely:
   `MolToSmilesEnum`, `MolToSmilesTokenInventoryEstimate`,
   `MolToSmilesTokenSuperset`, decoder construction, and deviation diagnostics.

6. Add a bulk store after the single-object semantics are stable. For 100M-scale
   datasets, block-compressed batches with an index are likely more important
   than a perfect standalone object envelope.

Test requirements
-----------------

Tests should cover:

- connected and disconnected molecules
- stereo and nonstereo surfaces
- explicit bonds, explicit hydrogens, kekule output, and atom-map handling
- fragment order and original atom-index mapping
- serialization round trips
- malformed payload rejection
- version/schema mismatch behavior
- equivalence with current RDKit-backed public APIs
- no RDKit dependency after `PrepareMol`

Release posture
---------------

`PreparedMol` should not become public API until it can be read back and used
through the normal Grimace runtime without RDKit. Before that point, it can be
developed as an internal or experimental shape, with tests defining the intended
contract.
