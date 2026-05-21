# South Star Readiness Reconciliation

Branch: `south-star`

Date: 2026-05-21

Task: `South Star 165: Reconcile readiness after combined-stereo expansion`

## Current Boundary

The current South Star surface is still private. It is a semantic enumerator
investigation surface, not RDKit writer parity and not a public API export.

The supported private fixture surface is now:

- exact first-domain directional-bond stereo cases;
- markerless acyclic trees;
- single- and two-atom atom-text cases;
- selected isotope, radical, charged, atom-map, explicit-H, combined modifier,
  double-bond, and triple-bond text cases;
- first markerless aromatic monocycle text case;
- simple and branched nonstereo monocycles;
- ring-stereo monocycles;
- ring-local tetrahedral monocycles;
- nonstereo and ring-stereo polycyclic skeleton witnesses;
- disconnected composition of already supported fragments;
- independent directional-stereo components;
- directional plus tetrahedral composition;
- disconnected mixed-stereo composition;
- exocyclic directional branches on monocycles;
- ring-local tetrahedral plus exocyclic directional branches on monocycles.

Every current expanded-support fixture case is unified-reference-backed. The
old temporary/regression authority names remain in the manifest vocabulary only
so tests can classify future or historical evidence deliberately; they do not
back any current fixture case.

## Reconciled Counts

As of the `South Star 164` checkpoint:

- expanded-support cases: `37`;
- exact first-domain cases: `5`;
- total semantic benchmark rows: `42`;
- public API blocker cases in the readiness matrix: `0`;
- temporary-witness-backed current cases: `0`;
- regression-witness-backed current cases: `0`.

The benchmark artifact is now pinned as evidence over the same 42 current
semantic fixture cases. It records policy set, command, environment metadata,
per-case output counts, and timings. It is not evidence for RDKit writer parity
or for package-level speed claims.

## Remaining Boundary

The main unresolved boundary is not the current fixture surface; it is breadth.
The next feature families should still enter through the same one-truth spine:
molecule facts, traversal and fragment events, typed renderer obligations,
semantic constraints, annotation policy, rendered output, and parse-back
evidence.

The next high-value surfaces are:

- broader atom/bond text combinations;
- an explicit aromatic semantic policy contract before aromatic output support;
- first aromatic witnesses only after that policy is named;
- documentation of the private EnumS contract before any export decision;
- an export-gate blocker inventory after the boundary/docs pass.
