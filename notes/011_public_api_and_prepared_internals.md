Public API and prepared internals
=================================

Position
--------

The Python public API should remain RDKit-first. The internal runtime should be
prepared-first.

Most users already have RDKit molecules, and Grimace should fit that workflow:

```python
grimace.MolToSmilesEnum(mol, ...)
grimace.MolToSmilesDecoder(mol, ...)
grimace.MolToSmilesTokenInventory(mol, ...)
```

`PreparedMol` is an optimization and persistence object, not a replacement
molecule model.

Internal model
--------------

The internal direction is still:

```text
RDKit Mol -> PrepareMol -> prepared fragments -> Rust runtime
```

RDKit is allowed at the preparation boundary. The runtime should operate on
prepared data and should not need RDKit after preparation.

Flag ownership
--------------

Writer-surface flags belong to preparation:

- `isomericSmiles`
- `kekuleSmiles`
- `allBondsExplicit`
- `allHsExplicit`
- `ignoreAtomMapNumbers`

Runtime traversal flags belong to enumeration/decoding:

- `rootedAtAtom`
- `canonical`
- `doRandom`

For RDKit inputs, the one-shot Python APIs can accept both groups and prepare
internally. For `PreparedMol` inputs, writer-surface flags are already baked in;
runtime APIs should reject conflicting writer flags rather than silently
reinterpret the prepared object.

PreparedMol role
----------------

`PreparedMol` exists for workloads that want to pay the RDKit preparation cost
once and reuse or serialize the result:

```python
prepared = grimace.PrepareMol(mol, isomericSmiles=True)
blob = prepared.to_bytes()
prepared = grimace.PreparedMol.from_bytes(blob)
```

Docs should not present this as the normal path for casual use. The normal path
is still passing an RDKit molecule directly.

Long-term rule
--------------

- Python public surface: RDKit-first.
- Advanced/cache surface: `PreparedMol`.
- Rust/internal runtime: prepared-data-first.
- RDKit dependency: preparation boundary only.

This keeps Grimace ergonomic for Python users while keeping the runtime
architecture independent of RDKit.
