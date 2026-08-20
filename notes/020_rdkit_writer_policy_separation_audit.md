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
- `rdkit_token_flip_adjustment_observation_from_state` and
  `RdkitTokenFlipAdjustmentObservations` are correctly named as RDKit writer
  policy, not generic stereo chemistry. The pinned witness note in
  `notes/018_rdkit_token_flip_adjustment_witnesses.md` gives this path a source
  reference and concrete fixture coverage.
- `rdkit_traversal_writer_facts_by_component`,
  `rdkit_writer_selected_marker_event_facts_by_component`,
  `rdkit_writer_marker_event_facts_by_component`,
  `rdkit_writer_marker_obligation_domains_by_component`,
  `rdkit_writer_slot_coalesced_marker_event_facts_by_component`, and
  `rdkit_marker_rows_accept_deferred_token` now identify the marker-placement
  and deferred-token paths as RDKit writer-policy machinery rather than generic
  SMILES semantics.
- `rdkit_marker_row_survivor_component_state`,
  `rdkit_marker_placement_state_to_py`, and
  `rdkit_marker_placement_row_to_py` make the diagnostic marker-row path
  explicit as RDKit writer-policy state.
- `rdkit_ring_closure_projected_marker_slots` is correctly named as RDKit
  projection behavior. It should stay separate from generic marker-slot parsing
  helpers.
- `tests/fixtures/rdkit_known_stereo_gaps/2026.03.1.json` now distinguishes
  red acceptance cases from same-family passing guards and classifies current
  failures as RDKit writer-policy gaps rather than unsupported chemistry.

## Still Blurred Or Too Generic

- `record_directional_marker_trace` and `record_marker_event_traces_for_edge`
  are still generic names for RDKit writer-policy observation paths. They are
  lower-level trace recorders, so renaming them should be done only when the
  trace schema itself is next touched.
- The Python diagnostic payload still uses stable keys such as
  `traversal_facts`, `marker_event_facts`, and `marker_obligation_domains`.
  That is acceptable for consumers, but Rust helper names should continue to
  carry the RDKit writer-policy prefix.
- Python-side fragment/root behavior is documented as RDKit parity, but the
  lower-level helper names still look generic. That is acceptable while Python
  owns RDKit interop, but future Rust-side movement should keep "RDKit
  fragment/root policy" explicit.

## Naming Direction

Use generic names only for chemistry/SMILES semantics that should hold outside
RDKit. Use `rdkit_writer_*` or an explicit `Rdkit*` type when the rule is needed
only to match RDKit's serializer.

Completed rename direction:

- Selected-carrier marker events are now
  `rdkit_writer_selected_marker_event_facts_by_component`.
- All-candidate marker/no-marker events are now
  `rdkit_writer_marker_event_facts_by_component`.
- Traversal facts are now `rdkit_traversal_writer_facts_by_component`.
- Deferred-token marker-row acceptance is now
  `rdkit_marker_rows_accept_deferred_token`.
- Marker obligation domains use `RdkitWriterMarkerObligationDomain` and
  `rdkit_writer_marker_obligation_domains_*` helpers.
- Marker-row survivor diagnostics use `rdkit_marker_row_*` and
  `rdkit_marker_placement_*` helpers.

## Next Review Rule

Before adding a new stereo support-shaping rule, answer two questions in the
code or fixture metadata:

- Is this a semantic stereo/SMILES constraint, an RDKit local writer rule, or an
  RDKit traversal/emission writer rule?
- What pinned fixture, RDKit source reference, or exploration note explains why
  the rule exists?

If those answers are not clear, the rule should stay in exploration or a named
shadow diagnostic path rather than enter the production runtime.
