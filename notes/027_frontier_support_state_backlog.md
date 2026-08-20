# Frontier Support-State Backlog

Branch: `stereo-constraint-model`

Date: 2026-05-19

Source: Notion table `Grimace-py x Codex`, rows with `Status = Backlog`
queried on 2026-05-19.

## Purpose

This note records the current straight-line backlog for making stereo support
flow through one explicit frontier support-state boundary.

The immediate target is still RDKit writer-string parity for
`canonical=False, doRandom=True`. The design should also leave a clean seam for
a separate principled exact-support layer. Do not blur those layers while
executing this backlog.

## Design Position

The current runtime has useful constraint machinery, but support decisions are
still split between model state, walker fields, marker-row filtering, and
procedural compatibility checks. The backlog below is ordered to converge on
one source of truth:

- walker states carry explicit facts observed so far;
- a frontier support-state query consumes those facts;
- the query returns survivor carrier, token-phase, and marker-row state;
- runtime support and commit decisions consume the query result;
- legacy reconstruction remains only as shadow diagnostics during migration;
- performance optimization comes last, after ownership is clean.

Rows remain an implementation choice, not the conceptual goal. They are useful
as an auditable finite representation of surviving assignments. If a later
fact/propagator representation is cleaner for a principled semantic layer, it
should be allowed to coexist beside the RDKit writer-policy projection.

## Backlog Sequence

### 01 Persist Typed Stereo Observations

Severity: High

Notion:
https://www.notion.so/01-Persist-typed-stereo-observations-365e13282611817aaaf2e196f24d0b1d

Add persistent per-state token-observation facts for phase, begin side,
selected begin token, first-emitted relation, and RDKit token-flip adjustment.
Make the current committed-vs-observed filter consume these carried facts, with
reconstructed observations retained only as a shadow assertion.

Expected result:

- inferred token constraints are carried as typed facts, not recomputed as the
  primary runtime path;
- reconstruction from walker fields becomes a debug/equivalence check;
- no inferred observation is promoted to a committed `StereoTokenFlipFact`.

### 02 Introduce FrontierSupportState Query

Severity: High

Notion:
https://www.notion.so/02-Introduce-FrontierSupportState-query-365e1328261181a9afe4d21518aa8374

Create one named support-state query from frontier facts. It should consume
carrier facts, committed token flips, inferred token observations, marker
events, deferred carrier blocks, and RDKit writer-policy facts; return survivor
carrier ids, token-phase ids, marker-row ids, forced selected neighbors, and
legal deferred-token flips.

Expected result:

- the query becomes the named boundary for stereo support decisions;
- raw walker fields stop being interpreted independently in multiple places;
- diagnostics expose survivor counts and forced/unresolved status.

### 03 Route Deferred Tokens Through Support State

Severity: High

Notion:
https://www.notion.so/03-Route-deferred-tokens-through-support-state-365e1328261181ed9d75cf037089434c

Refactor `deferred_token_support` and `commit_deferred_token_choice` to ask
`FrontierSupportState` which tokens and token flips survive from the current
frontier. Remove duplicate ad hoc constraint-state assembly from deferred-token
support paths once equivalence is proven.

Expected result:

- deferred-token support and commit use the same survivor state;
- token legality is explained by explicit facts and surviving assignments;
- old ad hoc assembly remains only until focused equivalence tests pass.

### 04 Demote Committed-Token Parity Filter

Severity: High

Notion:
https://www.notion.so/04-Demote-committed-token-parity-filter-365e132826118173b12df7ed9e8e4ffb

After `FrontierSupportState` rejects the same diene extras, move
`assert_committed_component_token_flips_match_boundary_observations` out of
production support and keep it as a debug/test equivalence assertion only.
Delete it once pinned RDKit parity and focused stereo witnesses prove
equivalence.

Expected result:

- the current production parity filter stops being support-shaping logic;
- diene-extra rejection is owned by frontier support-state survivors;
- parity filter cost leaves the hot path after equivalence is proven.

### 05 Replace Selected-Neighbor Repair

Severity: High

Notion:
https://www.notion.so/05-Replace-selected-neighbor-repair-365e132826118100a8cddff3dd272e74

Replace `resolved_selected_neighbors_from_fields` and shared-edge repair with
forced selected-neighbor results from `FrontierSupportState`. Keep
field-derived selected neighbors as shadow diagnostics during migration; do not
promote carrier-only state because it already lost valid diene support.

Expected result:

- selected carrier ownership moves from walker-field repair to survivor state;
- shared-carrier cases are resolved through joined support, not local repair;
- known carrier-only support loss remains blocked by design.

### 06 Model Observation Fields Directly

Severity: Medium

Notion:
https://www.notion.so/06-Model-observation-fields-directly-365e132826118173bc81e153e164c777

Extend constraint-model APIs so `StereoTokenObservationFact` filters
token-phase state by named observation fields instead of immediately collapsing
to an implied token flip. Keep current implied-flip conversion as a shadow
equivalence check while the richer rule layer is introduced.

Expected result:

- observation fields become model inputs, not just a path to final flips;
- RDKit writer adjustments remain named policy facts;
- token-phase filtering is more inspectable and less circular.

### 07 Add Support-State Guardrail Tests

Severity: Medium

Notion:
https://www.notion.so/07-Add-support-state-guardrail-tests-365e1328261181cfab55c7d15d6285bf

Add focused tests for the migration: oxime timing witness remains exact,
conjugated diene extras stay rejected, support-state and legacy filter agree
during shadow mode, exact public invariants pass, pinned RDKit parity passes,
and known-gap fixtures show no unintended support broadening.

Expected result:

- every migration slice has a small exactness witness;
- known regressions are pinned before deleting compatibility checks;
- performance and correctness evidence are kept separate.

### 08 Optimize Survivor-State Incrementally

Severity: Medium

Notion:
https://www.notion.so/08-Optimize-survivor-state-incrementally-365e1328261181b4af99fc8907503048

After the support-state boundary is the source of truth, optimize by carrying
compact survivor ids or bitsets in walker states and updating only touched
components per emitted token. Do not start this until semantic ownership is
clean; it is the performance phase, not the architecture phase.

Expected result:

- repeated full survivor reconstruction leaves the hot path;
- only touched components are updated after each emitted token;
- correctness invariants remain attached to the support-state boundary.

## Execution Guardrails

- Execute in order unless a slice exposes a strictly smaller prerequisite.
- Keep each task small enough to close as one Notion `In review` item with a
  short commit hash and concrete verification notes.
- Do not delete legacy logic until the replacement produces the same survivor
  decisions on the pinned witnesses.
- Keep RDKit writer-policy facts named as RDKit policy, not chemistry.
- Keep semantic/full-marker exploration separate from public RDKit parity
  runtime unless a test explicitly targets semantic equivalence.
- Treat performance wins as non-authoritative until exact public invariants and
  pinned RDKit parity pass.

## Critical Risks

- Reintroducing dual truth by carrying typed facts but still letting downstream
  code reinterpret raw walker fields.
- Hiding RDKit-specific policy in generic semantic names.
- Promoting carrier-only selected-neighbor state despite the known diene support
  loss.
- Deleting the committed-token filter before `FrontierSupportState` rejects the
  same extras.
- Optimizing survivor state before the ownership boundary is stable, which would
  make later semantic changes harder to audit.

## Done Criteria For This Backlog

This backlog is complete when runtime support decisions for stereo flow through
one frontier support-state query, legacy field repair and parity filters are
debug-only or gone, focused guardrails cover the known red/slow witnesses, and
incremental survivor-state updates restore acceptable performance without
changing support.
