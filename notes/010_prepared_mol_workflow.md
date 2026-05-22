PreparedMol inevitables
=======================

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

Inevitable shape
----------------

To support disconnected molecules without RDKit after preparation, the prepared
shape must contain:

```text
PreparedMol
  writer_flags
  fragments[]

fragment
  atom_indices[]
  prepared_graph
```

The exact Python/Rust classes are implementation details. The data obligations
are not: writer flags, ordered fragments, original atom indices, prepared graph
data, and versioned bytes. In Python, `PreparedMol` should expose this as an
opaque object, not as public structural fields.

Inevitable work
---------------

1. Define the prepared shape.
   The object needs schema/version metadata, writer flags, ordered fragments,
   original atom-index mappings, and one prepared graph per fragment.

2. Implement preparation.
   `PrepareMol(mol, writer flags...)` may use RDKit. It must split fragments in
   RDKit order, prepare each connected fragment under the selected writer
   surface, preserve original atom indices, and store no RDKit molecule.

3. Validate loaded objects.
   Serialized objects need early rejection for unsupported byte-format
   versions, malformed flags, malformed fragments, atom-index/graph mismatches,
   and graph writer flags that disagree with the outer prepared molecule.

4. Implement serialization.
   A single-object `to_bytes()` / `from_bytes()` path is needed for caches,
   fixtures, tests, and format evolution. The first format can be simple, but it
   must be versioned.

5. Expose the minimal API.
   Once public, the surface should be small: `PreparedMol`, `PrepareMol`, and
   serialization methods on `PreparedMol`. Structural fields are not public
   API.

6. Prove no RDKit after preparation.
   Tests should check that a prepared object stores only prepared graph data and
   primitive/container metadata, not `Chem.Mol`.

7. Wire runtime consumption.
   `PreparedMol` must eventually be accepted by:
   `MolToSmilesEnum`, `MolToSmilesTokenInventory`,
   `MolToSmilesTokenInventorySuperset`, decoder construction, and deviation
   diagnostics.
   Without this, it is only a stored artifact.

8. Optimize storage after semantics settle.
   Fast reads are inevitable. The exact encoding is not. Rust-native decoding
   and block-compressed bulk storage can follow once the object contract is
   stable.

Choices, not inevitables
------------------------

- The private module name.
- Whether the public constructor is `PrepareMol(...)` or
  `PreparedMol.from_mol(...)`.
- Whether the first implementation stores Python dict-compatible graph payloads
  or decodes directly into Rust.

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

`PreparedMol` becomes public API only when it can be read back and used through
the normal Grimace runtime without RDKit. That gate is now satisfied by the
Rust-backed prepared object and boundary tests.
