# Directional ring-pair semantic replay

## Concept split

The writer has one residual mutation for recording a directional physical bond:
restrict carrier variables, propagate, discharge the emitted-bond factor and any
completed site factors, project discharged variables, and append one bond
occurrence. Acyclic bond emission and ring-pair closure are two callers of that
mutation, not two implementations of it.

The shared private result belongs in `writer_stereo.py`. It exposes the exact
source/successor residual facts, propagation result, discharge/projected keys,
bond occurrence, and capabilities. Each caller remains responsible for its own
proof term and event vocabulary.

The artifact verifier does not consume that runtime helper. It reconstructs the
single-site ring relation from molecule facts, serialized policy identity, the
source open endpoint, and the paired event, then executes the residual-store
operations independently.

## Ownership boundaries

- `writer_stereo.py` owns live residual mutation and physical bond recording.
- `writer_residual_transition_terms.py` owns the v7 pair proof vocabulary.
- `writer_support_artifact_offline_verifier.py` owns producer-free replay and
  exact writer-state lifecycle anchoring.
- Ring-state transition code continues to own open/closed closure records and
  label allocation/release; the offline verifier checks those products rather
  than recreating them.

## Supported slice

The v7 pair term covers one specified directional site, one neighbor-atom
carrier model, a single non-bridge ring bond, empty endpoint bond markers, and
ordinary absent/forward/reverse direction marks. Shared carriers, non-single
ring bonds, closure-marker coupling, tetra ring-order work, and custom policy
surfaces remain explicit later boundaries.

## Serious alternatives and regrets

Calling `_on_bond_emitted` from the pair path hides which facts were used and
makes operation-specific proof construction awkward. Duplicating its residual
logic would let discharge and projection drift. The shared typed result is the
smallest seam that avoids both problems.

The term uses tuple-valued carrier models now even though v7 requires exactly
one. This preserves the proof vocabulary for a later shared-carrier extension
without claiming that shared replay is currently supported.

Reduced pair artifacts begin immediately before genuine `BondId(3)` pair
transitions. They are the ordinary-test seam; the full root-zero artifact stays
the single slow end-to-end integration probe.
