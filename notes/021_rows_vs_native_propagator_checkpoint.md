# Rows vs Native Propagator Checkpoint

Branch: `stereo-constraint-model`

## Question

Should explicit assignment/marker rows become the production architecture, or
should they remain diagnostics/test oracle while a native fact propagator owns
runtime support?

## Decision Status

Do not make the final architecture decision yet. Keep explicit rows as the
current model-backed boundary for token phase, carrier, and marker-placement
reasoning, but treat that as a checkpoint architecture rather than a permanent
commitment.

The missing evidence is not philosophical. The current runtime still has three
large procedural surfaces:

- carrier commitment is partly embedded in edge emission;
- deferred token commit still mutates walker token-flip vectors;
- shared-carrier resolution still uses post-resolution repair rather than a
  joined survivor query.

Rows should not be judged as the final runtime representation until those
surfaces are either behind the boundary or explicitly proven incompatible with
it.

## Current Evidence For Rows

- Rows give inspectable witnesses for carrier choice, token phase, and
  visible-marker placement without hiding RDKit policy in local branches.
- The row model caught the important distinction between selected carrier edge
  and visible marker edge. That distinction is exactly what local procedural
  shortcuts blurred.
- The current fixture scale is still small enough for rows to be practical in
  diagnostics and targeted runtime filtering.
- The red acceptance tests are now classified by current result and gap class,
  so rows can be evaluated against concrete RDKit writer-policy failures rather
  than abstract preference.

## Current Evidence Against Declaring Rows Final

- Rows are currently materialized eagerly per model component. That may be fine
  for the pinned corpus, but it is not yet proven for large highly coupled
  stereo systems.
- Some state remains duplicated between row-derived facts and walker vectors.
  As long as that dual truth exists, a row-vs-propagator comparison is polluted
  by adapter complexity.
- Marker placement still needs negative-event handling and shared-obligation
  semantics before row filtering represents the whole online problem.
- A native propagator may represent the same constraints more compactly once
  the exact fact language is stable.

## Minimal-Regret Path

1. Keep rows as the executable specification and diagnostic oracle for the next
   slices.
2. Promote carrier commitment, deferred token commit, and shared-carrier
   resolution behind the named support boundary.
3. Measure row counts and survivor-filter work on pinned red cases and larger
   sampled cases after those promotions.
4. Only then decide one of:
   - rows stay in production because they remain bounded and inspectable;
   - rows stay as a test oracle while runtime uses a native propagator;
   - a hybrid keeps row ids for hard coupled components and direct propagation
     for simple components.

## Recommendation

For now, do not replace rows and do not make rows the declared final runtime
architecture. Use them as the clearest available specification boundary while
continuing the straight-line integration work. The decision becomes meaningful
only after the remaining procedural support-shaping paths stop bypassing the
boundary.

## Post-Boundary Revisit

Date: 2026-05-17

The main blockers named above have now moved far enough behind the support
boundary to make a first measured recommendation:

- carrier commitment is represented through explicit selected-neighbor facts
  and joined support-boundary checks;
- committed token state is represented as token-flip facts, with the walker
  vector retained as a shadow consistency check;
- deferred marker support now asks marker-placement rows for viable token
  flips before support/commit;
- the manual difficult cis/cis and cis/trans witnesses moved from red
  `support_missing` cases to support-present family guards.

This does not prove rows are the final architecture, but it removes the main
reason not to evaluate them.

## Current Measurements

Measured on RDKit `2026.03.1` pinned fixtures after the isolated
all-two-candidate observation fix.

- Exact small-support stereo cases: 9 cases, maximum 1 model component, 2
  sides, 2 marker-placement rows per component. Terminal survivor rows were
  always 1.
- Serializer regression stereo cases: 18 cases, maximum 1 model component, 4
  sides, 12 marker-placement rows per component. The largest fixture-backed
  terminal survivor count observed was 2 rows.
- Known stereo-gap corpus: 15 currently supported model-build cases, maximum 2
  model components, 6 sides, 72 marker-placement rows in the largest component,
  and 84 total marker rows across components.
- Manual difficult family: 11 atoms, 2 components, 4 sides, largest component
  has 72 marker rows; terminal survivor filtering had maximum 6 rows and median
  2.5 rows across output/component rows.
- Sampled ordinary connected bond-stereo cases from the default molecule set:
  22 supported cases up to 25 atoms; all had 1 component, 2 sides, and 2
  marker rows.

The current evidence is therefore not row explosion. The hard pinned cases are
still small enough that row ids are useful as an executable explanation of
support, not just a diagnostic artifact.

## Recommendation After Revisit

Keep explicit rows as the production support-boundary representation for the
current branch. Do not start a native propagator rewrite now.

The engineering reason is simple: the row model is currently the only
representation that makes carrier choice, token phase, marker placement,
negative marker information, and RDKit writer policy inspectable in one place.
It also caught the all-two-candidate vs anchored mixed-component distinction
that a broad local rule got wrong.

The no-regret target is a bounded-row production path with measurement gates:

1. Keep runtime filtering row-backed for coupled/stereo-writer-policy
   components.
2. Add row-count and survivor-count diagnostics to parity/known-gap fixtures
   before making further algorithmic rewrites.
3. Treat a native propagator as a fallback only if real fixtures or generated
   adversarial cases produce sustained row counts that are too large.
4. If that happens, prefer a hybrid: direct propagation for trivial
   one-candidate components and row-backed state for coupled components.

The immediate next task is not a rows-vs-native rewrite. It is to finish
removing stale shadow/procedural paths and keep adding measured red-gap targets
against the row-backed boundary.
