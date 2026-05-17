# RDKit Writer-Policy Separation Audit

Branch: `stereo-constraint-model`

## Purpose

The public runtime is an RDKit writer-parity API. The implementation still
needs a principled semantic core, but exact emitted strings must also follow
RDKit's `canonical=False, doRandom=True` writer behavior. This audit records
where the current branch already names that distinction and where generic names
still hide RDKit-specific policy.

## Current Separation That Looks Sound

- `StereoConstraintLayer::Semantic`, `RdkitLocalWriter`, and
  `RdkitTraversalWriter` provide the right high-level split. The enum now says
  explicitly which layer is molecule semantics and which layers are RDKit writer
  exclusions.
- `rdkit_component_token_flip_adjustment` and
  `RdkitTokenFlipAdjustmentObservations` are correctly named as RDKit writer
  policy, not generic stereo chemistry. The pinned witness note in
  `notes/018_rdkit_token_flip_adjustment_witnesses.md` gives this path a source
  reference and concrete fixture coverage.
- `rdkit_ring_closure_projected_marker_slots` is correctly named as RDKit
  projection behavior. It should stay separate from generic marker-slot parsing
  helpers.
- `tests/fixtures/rdkit_known_stereo_gaps/2026.03.1.json` now distinguishes
  red acceptance cases from same-family passing guards and classifies current
  failures as RDKit writer-policy gaps rather than unsupported chemistry.

## Still Blurred Or Too Generic

- `record_directional_marker_trace`, `record_marker_event_traces_for_edge`,
  `marker_event_facts_by_component`, and
  `shadow_marker_event_facts_by_component` are RDKit writer-policy observation
  paths, but their names read like generic marker semantics.
- `traversal_constraint_facts_by_component` produces facts used by RDKit
  traversal-writer layers. The name is accurate but does not advertise that
  these are writer-parity facts, not OpenSMILES-level constraints.
- `marker_obligation_domains_by_component` is a useful concept, but the current
  obligation shape is driven by RDKit visible-marker placement. Future names
  should say whether an obligation is semantic, RDKit-local, or RDKit-traversal.
- `deferred_candidate_survives_marker_rows` is now model-backed, but it still
  reads as an implementation detail rather than a boundary query over RDKit
  marker-placement rows.
- Python-side fragment/root behavior is documented as RDKit parity, but the
  lower-level helper names still look generic. That is acceptable while Python
  owns RDKit interop, but future Rust-side movement should keep "RDKit
  fragment/root policy" explicit.

## Naming Direction

Use generic names only for chemistry/SMILES semantics that should hold outside
RDKit. Use `rdkit_writer_*` or an explicit `Rdkit*` type when the rule is needed
only to match RDKit's serializer.

Suggested future renames:

- `marker_event_facts_by_component` -> `rdkit_writer_marker_event_facts_by_component`
- `shadow_marker_event_facts_by_component` -> `shadow_rdkit_writer_marker_event_facts_by_component`
- `traversal_constraint_facts_by_component` -> `rdkit_traversal_writer_facts_by_component`
- `deferred_candidate_survives_marker_rows` -> `rdkit_marker_rows_accept_deferred_token`
- `MarkerObligationDomain` -> split into a semantic obligation type only if one
  exists; otherwise use an RDKit-writer-specific name.

## Next Review Rule

Before adding a new stereo support-shaping rule, answer two questions in the
code or fixture metadata:

- Is this a semantic stereo/SMILES constraint, an RDKit local writer rule, or an
  RDKit traversal/emission writer rule?
- What pinned fixture, RDKit source reference, or exploration note explains why
  the rule exists?

If those answers are not clear, the rule should stay in exploration or a named
shadow diagnostic path rather than enter the production runtime.
