# RDKit BondDir surface normalization

## Decision

Grimace should normalize an RDKit molecule to the selected RDKit writer surface,
not to a generic "all RDKit metadata removed" shape.

For the nonstereo surface, that means:

- call `Chem.RemoveStereochemistry()`;
- then clear any remaining `BondDir` values;
- do this only after choosing `CONNECTED_NONSTEREO_SURFACE`.

Do not clear directions before surface selection. When Grimace selects the
stereo-capable surface, direction metadata is part of the surface and must be
preserved and validated explicitly.

## RDKit boundary

RDKit `BondDir` is not purely stereochemistry. It is also writer, parser,
depiction, molfile/CXSMILES, and restoration metadata.

In RDKit source, `RemoveStereochemistry()` clears atom stereo, double-bond
stereo, stereo groups, and directions on single/double bonds. It does not mean
"clear all bond directions"; aromatic `BondDir` metadata can remain.

RDKit has C++ APIs such as `clearSingleBondDirFlags()`,
`clearAllBondDirFlags()`, and `clearDirFlags()`. In the current Python runtime
these are not exposed, while `RemoveStereochemistry()`,
`SetBondStereoFromDirections()`, and `SetDoubleBondNeighborDirections()` are.

## Grimace invariant

Prepared graphs should carry only metadata needed for the selected Grimace
writer surface.

When the selected surface is nonstereo, hidden residual `BondDir` metadata is
not part of the writer-visible language. Keeping it causes Grimace to reject
molecules that RDKit can serialize on that surface without `/` or `\` tokens.

For stereo-capable surfaces, direction metadata remains part of the prepared
surface. Those surfaces must continue to validate and consume it explicitly.

## Implementation stance

The current Python loop over bonds is deliberate:

```python
for bond in working_mol.GetBonds():
    bond.SetBondDir(Chem.BondDir.NONE)
```

It is small, public-API-compatible, and exactly scoped to Grimace's nonstereo
surface normalization. If RDKit later exposes a Python clear-all-direction API,
switch only after proving it is equivalent for this surface and does not widen
the normalization boundary.
