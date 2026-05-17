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
