# South Star Aromatic Directional Surface Probe

Branch: `south-star`

Date: 2026-05-22

Task: `South Star 239: Probe aromatic directional surfaces`

## Question

Can aromatic directional-marker surfaces be admitted as ordinary South Star
fixed-molecule semantics, or should they remain a fail-fast policy boundary?

This probe distinguishes:

- a direction flag attached directly to an aromatic bond;
- an ordinary directional alkene attached to an aromatic ring through a
  non-aromatic carrier edge.

## Probe

Script:

`PYTHONPATH=python:. python3 tmp/exploration/south_star_aromatic_directional/001_probe_aromatic_directional_surfaces.py`

Observed output:

```text
manual aromatic direction writer output: c1ccccc1
manual aromatic direction survives reparse: False
manual aromatic direction support categories: ['aromatic_directional_surface']
exocyclic aromatic alkene support categories: []
```

## Interpretation

An in-memory RDKit `BondDir` on an aromatic bond is not a stable SMILES
semantic fact in the current surface:

- RDKit can carry the flag in memory;
- RDKit's SMILES writer drops it for benzene;
- reparsing the written string loses the direction flag;
- there is no corresponding South Star constraint family today.

This is different from `c1ccccc1/C=C/Cl`. In that molecule, the directional
markers belong to the exocyclic alkene's non-aromatic carrier bonds. The
aromatic ring is graph/text context, not the directional stereo surface itself.

## Decision For Now

Keep `aromatic_directional_surface` as a fail-fast boundary.

This is not a claim that all future aromatic directional syntax is impossible.
It is a narrower statement: a directional flag directly on an aromatic bond is
not yet a principled fixed-molecule South Star obligation, and RDKit's writer
does not preserve the in-memory flag in the tested witness.

## Tests Added

`tests.south_star.test_aromatic_boundary` now pins both sides of the boundary:

- manually directed aromatic bonds are not SMILES round-trippable and remain
  blocked by `aromatic_directional_surface`;
- exocyclic directional alkenes attached to aromatic rings are not classified
  as aromatic directional overlays when their directional carrier bonds are
  non-aromatic.

## Follow-Up

Do not expand support gates for aromatic directional overlays until there is a
named constraint family and a fixture authority that defines what semantic
object is being enumerated.

