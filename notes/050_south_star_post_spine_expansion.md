# South Star Post-Spine Expansion

Branch: `south-star`

Date: 2026-05-21

Task: `South Star 157: Map post-spine expansion families`

## Baseline

The current expanded-support checkpoint has crossed the first important
threshold:

- all current expanded-support fixture cases are unified-reference-backed;
- no current expanded-support case depends on temporary witness authority;
- no current expanded-support case is a public API blocker;
- the readiness matrix still describes a private South Star investigation
  surface, not a package-ready API.

This changes the next work from "clear known blockers" to "expand the
declared model without weakening the one-truth spine."

## Expansion Principle

New feature coverage should not be added as isolated case families. Each family
must attach to the same pipeline:

1. molecule facts;
2. traversal and fragment events;
3. typed renderer obligations;
4. semantic constraints;
5. annotation policy;
6. rendered output plus parse-back/conformance checks.

Fixture cases remain witnesses. They should not become the source of support.

## Expansion Families

### Domain-Gate Reconciliation

The proof-backed fixture surface is now broader than the old first-domain gate
language. Before export work, reconcile the private domain manifest, support
gates, readiness diagnostics, and unsupported-category names so they state the
same boundary.

This is not a license to support everything package-facing. It is a consistency
pass: the private South Star model should not call a feature both
unified-reference-backed and categorically unsupported without explaining the
layer boundary.

### Atom And Bond Text

The next low-risk expansion is atom/bond text because it mostly adds renderer
obligations over existing traversal events:

- isotopes;
- formal charge variants;
- radical variants;
- explicit hydrogens;
- non-default bracket atom spelling;
- additional non-aromatic bond orders that have clear grammar support.

The regret to avoid is adding text cases as per-atom fixtures rather than a
single atom-text obligation family.

### Combined Stereo

The next high-value correctness expansion is mixed stereo composition:

- multiple independent directional-bond components;
- directional bond stereo plus tetrahedral atom stereo;
- ring closure marker slots plus tetrahedral centers;
- polycyclic closure choices plus marker equations;
- disconnected composition of mixed-stereo fragments.

The point is not more examples. The point is to show that independent
components compose by Cartesian product, coupled components share equations,
and renderer obligations remain local to solved event assignments.

### Aromatic Policy

Aromatic support remains a separate policy-family problem. The existing
boundary note correctly keeps aromatic RDKit molecule facts fail-fast until the
model names:

- aromatic molecule-fact input contract;
- lowercase atom text policy;
- aromatic bond elision/rendering policy;
- aromatic/Kekule semantic equivalence expectations;
- directional overlays on aromatic bonds.

This should proceed as a policy contract before implementation.

### Complexity And Export

After gate reconciliation and at least one combined-stereo expansion, create an
EnumS benchmark artifact for the semantic enumerator. Only then should the
export gate decide whether `MolToSmilesEnumS` remains private, becomes an
experimental package surface, or waits for more feature families.

## Concrete Backlog Slices

1. Reconcile the promoted fixture domains with the private domain manifest and
   support-gate language.
2. Expand atom/bond text through shared renderer obligations.
3. Expand combined-stereo composition through the shared constraint spine.
4. Define the aromatic policy contract before aromatic implementation.
5. Build the first semantic-enumerator benchmark artifact.
6. Re-run the export sequencing decision after the above evidence exists.

