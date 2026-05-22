# South Star Public EnumS Surface Contract

Branch: `south-star`

Date: 2026-05-22

Task: `South Star 244: Specify public MolToSmilesEnumS surface`

## Purpose

Specify the first public `MolToSmilesEnumS` surface before exporting it.

This is a contract slice, not an export slice. `grimace.__init__` must continue
not to expose `MolToSmilesEnumS` until the promotion gates are satisfied.

## Proposed First Public Shape

The proposed first public API is:

`MolToSmilesEnumS(mol) -> tuple[str, ...]`

Contract:

- input: `rdkit.Chem.Mol` only;
- output: `tuple[str, ...]`;
- order: deterministic first-occurrence deduplication under the default South
  Star policy;
- primary correctness contract: support membership, not order;
- diagnostics: not exposed on the first public surface;
- policy surface: fixed default South Star policy;
- unsupported inputs: fail fast with `SouthStarUnsupportedFeatureError`;
- relation to `MolToSmilesEnum`: semantic support, not RDKit writer parity.

The proposed shape is now inspectable through:

`grimace._south_star.api.south_star_proposed_public_api_contract()`

It deliberately reports `exported_from_public_package=False`.

## Why This Shape

The surface should not mimic RDKit writer flags. `MolToSmilesEnumS` is not a
writer-parity function and should not inherit RDKit's `canonical`,
`doRandom`, `rootedAtAtom`, or writer-mode flag vocabulary by accident.

The return type should be concrete rather than lazy for the first public
surface. South Star support is computed as a finite semantic support set under
a named policy; returning a tuple makes that boundary clear and avoids implying
streaming support or RDKit-like sampling.

Diagnostics should remain private initially. They are useful engineering
guardrails, but exposing them would prematurely freeze internal names for
traversals, marker slots, components, equations, and product diagnostics.

## What This Does Not Do

This does not:

- export `MolToSmilesEnumS`;
- add a public implementation wrapper;
- expose diagnostics publicly;
- add policy arguments;
- claim RDKit writer-parity equality;
- change the existing public `MolToSmilesEnum` API.

## Next Required Slice

The remaining promotion-gate work is to make docs, release-note, and
performance-evidence review gates more checkable before deciding whether the
export itself is appropriate.

