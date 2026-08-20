# Visible Marker Basis Taxonomy

Branch: `stereo-constraint-model`

## Purpose

This note defines the row/model vocabulary that should replace the current
cross-component topology guard in deferred directional-token support. It is a
design boundary only; it does not change runtime behavior.

The current runtime shortcut in `rust/src/rooted_stereo.rs` routes some
deferred tokens through the broad marker-row path when all of these are true:

- the prepared graph is acyclic;
- the runtime component has at least one side with multiple candidate carrier
  edges;
- one candidate carrier edge is shared with a different runtime component.

That rule is useful evidence, but it is not the invariant. The invariant we
need is: a visible slash/backslash marker has a legal interpretation only
relative to a basis, and the nontrivial bases need row-survivor witnesses.

This note resynthesizes the visible-marker audit in
`notes/019_visible_marker_carrier_selection_audit.md`, the shared-marker
alternatives in `notes/014_shared_marker_obligation_alternatives.md`, and the
decision to replace the cross-component deferred-token special case with an
explicit marker-basis model.

## Problem Shape

The old local assumption was:

```text
visible marker edge == selected stereo carrier edge
```

That is only valid in the selected-carrier case. RDKit can emit a visible
marker on a candidate, complement, or bridge edge while the selected carrier
edge for that component side is a different edge. A shared physical marker can
also be interpreted by more than one runtime component.

The failed replacements are informative:

- accepting all marker-row references is too broad and admits ordinary simple
  stereo spellings that RDKit does not emit;
- using the raw selected-carrier token first and an unconditional marker-row
  fallback is still too broad;
- selecting the carrier from the visible edge is wrong for bridge and
  complement-candidate witnesses;
- emulating RDKit's writer stack directly may be a useful oracle, but it does
  not give the principled row/state boundary we want.

The missing concept is not "use marker rows more often." The missing concept
is "which token basis is legal for this visible marker in this component row?"

## Terms

`emitted edge`
: The physical graph edge whose slash/backslash token is being emitted in the
current walker state.

`selected carrier edge`
: The carrier edge selected by a row/model assignment for a specific stereo
component side. This is the edge whose raw selected-carrier token defines the
ordinary token-flip basis for that side.

`raw token basis`
: The token basis obtained from the selected carrier edge, for example through
the current raw-token path for deferred directional edges. It is authoritative
only when the emitted edge and selected carrier edge coincide for the relevant
component side.

`emitted-edge basis`
: The token basis obtained from the actual emitted visible marker edge. This is
not automatically valid. It must be justified by a non-selected or shared
marker-basis row witness.

`marker event role`
: The writer-policy role of the marker event in the row model: selected-carrier
marker, non-selected visible marker, shared visible marker, no-marker event, or
deferred/discharged marker obligation.

`marker event slot`
: The direction-erased location where the marker event is observed. This should
be stable across slash/backslash choice and specific enough to distinguish
begin/end edge, branch, ring-closure, and later-slot writer behavior.

`row-survivor witness`
: Evidence that at least one marker-placement row remains alive after applying
the current carrier-choice, token-phase, marker, and no-marker facts. A
non-selected or shared marker interpretation is not accepted without such a
witness.

## Basis Classes

### SelectedCarrierBasis

The emitted edge is the selected carrier edge for the component side.

Facts:

- component and side identity;
- emitted edge;
- selected carrier edge;
- raw selected-carrier token basis;
- chosen marker token;
- marker event role and slot;
- row-survivor witness or equivalent selected-carrier consistency check.

Invariants:

- The raw selected-carrier basis is authoritative.
- The emitted-edge basis must not add support beyond the selected-carrier
  basis.
- Ordinary one-carrier and simple alkene cases should be explained here, not by
  the non-selected or shared basis.
- If the chosen token cannot be mapped from the raw token basis, the marker is
  rejected for this component side.

### NonSelectedVisibleEdgeBasis

The emitted edge is eligible as a visible marker edge for the component side,
but the row-selected carrier edge for that side is a different edge.

Facts:

- component and side identity;
- emitted edge and candidate neighbor represented by that edge;
- selected carrier edge and selected carrier neighbor from the row;
- raw selected-carrier token basis;
- emitted-edge token basis;
- chosen marker token;
- marker event role and slot;
- row-survivor witness after the positive marker event and any earlier
  no-marker facts.

Invariants:

- The visible edge does not select the carrier.
- The emitted-edge basis is legal only because the row witness says this marker
  placement is compatible with the selected carrier and token phase.
- The rule must not be inferred from graph topology alone.
- The accepted token flip must be derived from the row-compatible basis, not
  from an unconditional fallback.

### SharedVisibleEdgeBasis

One physical visible marker edge is interpreted by more than one runtime
component or side.

Facts:

- physical emitted edge and chosen marker token;
- per-component side identity;
- per-component selected carrier edge;
- per-component raw selected-carrier token basis;
- per-component emitted-edge basis;
- per-component marker event role and slot;
- per-component row-survivor witness;
- token compatibility across the components that observe the same visible
  marker.

Invariants:

- The same physical marker can be `SelectedCarrierBasis` for one component and
  `NonSelectedVisibleEdgeBasis` for another.
- There is no global token-flip shortcut for the shared edge.
- Acceptance is per component, then joined by compatibility of the single
  emitted token.
- A shared interpretation is accepted only if every observing component has a
  surviving row witness.

## Required Fact Shape

The eventual runtime/diagnostic object should carry these facts explicitly
instead of recovering them from scattered helper state:

- runtime component id and model component id;
- side id and endpoint atom;
- emitted begin atom, emitted end atom, and canonical emitted edge;
- selected carrier neighbor and canonical selected carrier edge;
- emitted-edge candidate neighbor for the side;
- raw selected-carrier token basis;
- emitted-edge token basis;
- chosen marker token;
- marker event role;
- direction-erased marker event slot;
- token flip implied by the selected-carrier basis, if any;
- token flip implied by the emitted-edge basis, if any;
- row ids before and after the marker event, or an equivalent nonempty survivor
  proof;
- basis class: selected carrier, non-selected visible edge, or shared visible
  edge.

## Runtime Invariants

- Do not treat a visible marker edge as the selected carrier edge except under
  `SelectedCarrierBasis`.
- Do not let an emitted-edge basis add support to ordinary selected-carrier
  cases.
- Do not accept a non-selected or shared marker interpretation without a
  row-survivor witness.
- Interpret a physical visible marker per component; join component-level
  interpretations only through the single emitted token.
- Keep parser-backed semantic validity separate from RDKit writer-policy
  placement. This taxonomy is an RDKit writer-policy boundary for the current
  public runtime.
- Keep diagnostics ahead of behavior changes. The next code slice should show
  the basis candidates and survivor witnesses before routing support through
  them.

## Relation To Existing Backlog

`Visible marker basis 02: expose diagnostics`
: Should add diagnostic payloads that list basis candidates, chosen basis,
implied token flips, and row-survivor evidence for deferred token decisions.

`Visible marker basis 03: pin witness classes`
: Should pin fixtures for selected-carrier, non-selected visible-edge, shared
visible-edge, and ordinary-simple-stereo non-regression cases.

`Visible marker basis 04: replace topology guard`
: Should replace the acyclic/multi-candidate/cross-component guard with a
basis-witness query.

`Visible marker basis 05: retire helpers`
: Should remove or demote the old topology and shadow-adapter helpers after
the basis-witness query is the source of truth.

`Replace runtime shared-carrier field forcing`
: Is adjacent but distinct. It concerns selected-neighbor resolution, not the
legal basis for interpreting a visible marker.

`Audit stereo completion identity vectors`
: Should consume the diagnostics from task 02 before judging whether completion
identity failures are semantic, writer-policy, or current-runtime artifacts.

## Rejected Final Rules

- Topology guard as final rule: too incidental and too hard to justify beyond
  the current witnesses.
- Marker rows everywhere: over-broad for ordinary simple stereo.
- Raw-token first plus unconditional marker-row fallback: still over-broad and
  duplicates support routes.
- Visible edge selects carrier: false for complement and bridge marker
  witnesses.
- Direct RDKit stack emulation: useful for investigation, but not the desired
  principled online row/state model.

## Implementation Target

The long-term runtime shape should have one internal query with a name close to
`visible_marker_basis_witnesses_for_deferred_token`. It should:

1. Build basis candidates for the emitted marker token and component side.
2. Classify each candidate as selected-carrier, non-selected visible-edge, or
   shared visible-edge.
3. Filter candidates through carrier, token-phase, positive-marker, and
   no-marker row facts.
4. Return only accepted token flips with their basis class and survivor
   witness.

The diagnostic answer should be explainable as:

```text
this token is accepted because this basis class has these facts and these rows
survive
```

not:

```text
this token is accepted because the component happened to be acyclic and shared
```
