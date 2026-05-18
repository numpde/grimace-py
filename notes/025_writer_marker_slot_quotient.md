# Writer Marker-Slot Quotient

Branch: `stereo-constraint-model`

Date: 2026-05-19

## Purpose

The `github3967_part2_directional_ring_closure_canonical` witness shows that
the next fix is not another local deferred-token patch.  The missing RDKit
writer spelling is parser-equivalent to the intended molecule, but it is
rejected by the current strict token-phase and marker-row model.

This note fixes the next target boundary: preserve strict semantic token-phase
state, and add a separately named RDKit writer marker-slot quotient if runtime
support needs to admit RDKit spellings that are equivalent at the emitted
marker-slot level but not represented by the current local row path.

## Evidence

Commits:

- `3b90c79`: exposed per-attempt token-flip row survival diagnostics.
- `7500a10`: exposed base token-phase state for those attempts.
- `5aee100`: pinned the github3967 token-phase quotient boundary in tests.
- `95792a2`: exposed emitted marker slots for deferred attempts.
- `b02b3d4`: measured base marker-row survival before candidate flip
  filtering.

For prefix `C1=CC/C=C2\C3=C`, the RDKit-needed terminal candidate is `\`.
The current diagnostic facts are:

- emitted marker slots are exactly the RDKit target slots:
  `(5, "/"), (9, "\\"), (13, "\\")`;
- those slots are in the pinned parser-equivalent minimal marker-slot set;
- strict semantic token-phase state is already forced to `stored`;
- the candidate requires `flipped`, leaving zero candidate token-phase rows;
- even without the candidate-flip filter, applying the candidate marker events
  to the base token-phase rows leaves zero marker rows.

So the failure is not only "candidate flip filter too strict".  It is also not
a row-local marker placement acceptance under the existing base state.  The
current row model represents one local spelling basis; RDKit's emitted spelling
uses a different marker-slot basis for the same parser-backed stereo
assignment.

## Decision

Do not broaden semantic token-phase assignments to make this pass.

Semantic token-phase state should continue to mean: which local token/phase
choices preserve the intended stereo assignment under the model's chosen
carrier and marker-placement representation.

RDKit writer parity needs an additional writer-policy layer that can say:

1. this emitted marker-slot set is parser-equivalent for the intended molecule
   and skeleton;
2. this emitted marker-slot set is the one RDKit's writer emits for the current
   traversal/rooting regime;
3. accepting it does not license arbitrary parser-equivalent spellings.

That layer is a quotient over marker-slot representations, not a mutation of
the semantic token-phase facts.

## Rejected Shortcuts

Broaden token-phase rows
: Rejected.  It would blur semantic state with writer spelling policy and make
  token-phase facts stop meaning one thing.

Bypass the candidate token-flip filter
: Rejected.  `b02b3d4` shows the base marker rows also reject the emitted RDKit
  marker slots.

Accept by completed-string parse equivalence
: Rejected for runtime.  The public API targets exact RDKit writer support and
  online decoding; post-hoc completed-string repair violates both boundaries.

Special-case github3967
: Rejected.  A compatibility case without a named writer-policy predicate would
  recreate the procedural patch problem this branch is trying to remove.

## Target Shape

The support boundary should have two clearly separated predicates:

- semantic row support: current carrier, token-phase, marker-row, and
  observation facts leave a nonempty semantic state;
- RDKit writer marker-slot quotient: current emitted marker slots are accepted
  because they are a pinned/witnessed RDKit writer spelling inside the
  parser-equivalent marker-slot class for the same semantic state.

The quotient must be online.  It may use facts available at the current
frontier:

- emitted marker slots so far;
- candidate marker slot and marker;
- direction-erased skeleton prefix context;
- component and side identity;
- outstanding no-marker observations;
- parser-equivalent marker-slot basis compiled or precomputed from the
  prepared molecule.

It must not use a completed-output repair phase.

## Next Slices

1. Add a diagnostic object that names quotient eligibility separately from
   semantic row survival.  For github3967 it should report:
   `semantic_row_accepts=false`, `marker_slot_quotient_candidate=true`, and
   `rdkit_writer_target_slots=true`.
2. Move parser-equivalent marker-slot computation out of the known-gap test
   body into a small test helper so more witnesses can reuse it without
   duplicating RDKit parser logic.
3. Add at least one more pinned witness, if available, where RDKit emits a
   parser-equivalent marker-slot basis that differs from the current row-local
   basis.  If no second witness exists, record that explicitly.
4. Only after the quotient diagnostics are fixture-backed, prototype a runtime
   support path that consults the quotient as a named RDKit writer-policy
   fallback after semantic row support rejects a candidate.
5. Before enabling that path broadly, require:
   exact public invariants, pinned RDKit parity, known-gap movement only for
   explicitly promoted cases, and no growth of non-target same-skeleton
   support.

## Runtime Integration Checkpoint

The first runtime prototype attempt exposes a sharper boundary: the quotient
cannot be implemented as "add the rejected marker token to
`deferred_token_support`".

Two runtime paths would still reject the candidate:

- `commit_deferred_token_choice` requires a semantic accepted token flip.  The
  github3967 RDKit target has no accepted flip under the strict semantic
  token-phase state, by design.
- `terminal_stereo_state_support_boundary_summary` validates completed states
  by recomputing semantic constraint state and marker-row survival from
  `marker_event_traces`.  For github3967, even the base token-phase rows leave
  zero marker rows after the RDKit target marker events.

So the runtime needs an explicit writer-policy representation, not a looser
semantic commit.  A quotient-accepted deferred token must carry enough state for
completion and memoization to distinguish:

- semantic row support accepted this token;
- RDKit writer marker-slot quotient accepted this token;
- neither accepted this token.

That representation must be part of the walker state and completion key if it
can affect future support, or it must be proven terminal-only before being kept
out of the key.  Treating it as a local support-list append would produce an
unrepresentable successor state.

The quotient fact source is also not solved by the current Rust runtime.  The
test fixtures prove two witnesses, but runtime support cannot consult test
fixtures.  The implementation needs a principled source for parser-equivalent
marker-slot classes or an explicitly named RDKit writer-policy table derived
from prepared molecule facts.  Until that source exists, accepting only the
fixture strings would be a special case, not a policy.

## Revised Runtime Slices

1. Add a writer-quotient acceptance fact to the walker state model, completion
   key, and terminal support summary as a named shadow/debug path first.  It
   should not change support yet.
2. Define the runtime source of marker-slot quotient facts.  Prefer a
   model-derived basis from prepared stereo components and marker slots.  If
   that is not expressive enough, document the exact missing RDKit writer fact
   before introducing a writer-policy table.
3. Add terminal-only quotient probing for github3967 that constructs the
   would-be successor state and reports why it is or is not complete under the
   explicit quotient fact.
4. Only after the state/key/summary representation exists, allow
   `deferred_token_support` and `commit_deferred_token_choice` to accept a
   quotient-backed token.  The semantic committed token flip must remain
   unchanged or absent; quotient acceptance must not masquerade as semantic
   token-phase acceptance.

## Runtime Fact Source

The quotient fact source should not be the pinned JSON fixture corpus.  Those
fixtures are evidence and regression tests, not runtime policy.  It also should
not be RDKit canonicalization or completed-string parse equivalence: that would
move the online writer into an oracle/repair phase.

The existing marker-placement row model is also not a sufficient source.  The
two pinned quotient witnesses already prove this:

- `github3967_part2_directional_ring_closure_canonical` has RDKit target
  slots `(5, "/"), (9, "\\"), (13, "\\")`, which are in the
  parser-equivalent minimal marker-slot class, but existing semantic/base rows
  still reject the emitted marker events.
- `github4582_chembl409450_random_vector_seed1_index0` has RDKit target
  slots `(8, "/"), (13, "/"), (35, "\\")`, also in the
  parser-equivalent minimal marker-slot class, while current same-skeleton
  support contains only different slot bases.

The runtime source should instead be a graph-level marker-equation predicate:

1. derive the relevant stereo double-bond equations from
   `PreparedSmilesGraphData` (`bond_stereo_kinds`, `bond_stereo_atoms`, stored
   bond endpoints, and the component model);
2. translate current `MarkerEventTrace` plus the candidate marker event into
   marker-slot observations on graph edges;
3. evaluate whether those observations induce the intended double-bond stereo
   assignment for the affected model component, independent of the currently
   selected marker-placement row basis;
4. separately require that the marker events are produced by the current RDKit
   writer traversal path, so parser-equivalent but non-writer spellings are not
   admitted.

This is still online.  At a deferred token frontier the walker has the prefix,
the emitted marker slots so far, the candidate marker slot, graph edge
provenance, component identity, and no-marker observations.  It does not need
the completed string, and it does not need to sample RDKit.

The missing runtime fact is therefore not a fixture lookup.  It is a named
component-local satisfiability fact, roughly:

`GraphMarkerSlotStereoSatisfies(component_idx, marker_events, candidate_event)`

That fact should feed `WriterMarkerSlotQuotientAcceptanceFact` only when the
graph-level equations accept the emitted marker basis and the current traversal
emitted that basis.  The existing semantic row support remains the primary
path; the quotient fact is an RDKit-writer spelling layer over marker-slot
representations.

## Next Implementation Slice

Prototype the graph-level marker-equation predicate as a diagnostic first.  It
should report, for each deferred marker token attempt:

- the stereo bonds in the affected model component;
- the marker-event edges and slots considered;
- whether the emitted marker basis satisfies the intended graph stereo;
- whether semantic marker rows reject the same basis;
- whether the candidate was emitted by the current writer traversal.

Acceptance for the diagnostic slice: both
`github3967_part2_directional_ring_closure_canonical` and
`github4582_chembl409450_random_vector_seed1_index0` should show
`graph_marker_equations_accept=true` while current semantic rows reject the
RDKit target basis.  If either witness fails, the diagnostic must name the
missing graph/writer fact precisely before any support change.

## Diagnostic Slice Result

The first graph-equation diagnostic should be interpreted as component-local,
not as full-output support proof.

For `github3967_part2_directional_ring_closure_canonical`, the exact terminal
target attempt now shows the intended split:

- semantic marker rows reject the candidate;
- graph marker equations accept both stereo bonds;
- the adjacent bonds use different parities (`stored` for bond `(3, 4)`,
  `flipped` for bond `(5, 6)`) while sharing the emitted marker at slot `9`.

For `github4582_chembl409450_random_vector_seed1_index0`, the diagnostic
currently proves the same shape at the component frontier, not for the whole
random-vector output in one row.  The component-1 frontier at root `13`, prefix
`c12c(NC(/C2=C2`, candidate `/`, has emitted target slots `(8, "/")` and
`(13, "/")`; semantic rows reject that basis while graph marker equations
accept bond `(8, 9)` with `flipped` parity.  Component-0 rows also demonstrate
graph-equation acceptance on their own frontier, but the current diagnostic
does not yet aggregate a complete writer path across model components and
roots.

The remaining missing fact for full CHEMBL coverage is therefore precise:
runtime diagnostics need a path-level writer-marker equation summary that
aggregates global emitted marker events across model components for one
candidate writer output.  Per-attempt component diagnostics are enough to
validate the graph equation source, but not enough to claim the complete
CHEMBL random-vector output has been represented as a single quotient path.
