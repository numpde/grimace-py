# South Star Starting Point

Branch: `south-star`

Date: 2026-05-19

Base: forked from `stereo-constraint-model` at `cca357e`
(`Define South Star semantic enumeration`).

## Purpose

This note is a neutral starting map for the `south-star` branch. It does not
pick the final architecture. Its job is to make the current mixed branch
inspectable before we remove, rename, or specialize anything.

The branch starts from `stereo-constraint-model`, which contains both:

- useful semantic/stereo constraint machinery; and
- substantial RDKit writer-parity machinery.

The first South Star work should make these surfaces visible. Only after that
separation is visible should we decide what to delete, quarantine, or reuse.

## Working Definition

South Star means mathematically clean semantic stereo enumeration with maximal
or explicit directional annotation, as described in
`notes/028_south_star_semantic_stereo_enumeration.md`.

For this starting point, the important distinction is:

- semantic correctness: emitted strings parse to the intended graph and stereo
  assignment;
- RDKit writer parity: emitted strings match RDKit's serializer support.

The `south-star` branch is for investigating the first without accidentally
depending on the second.

## Initial Assumptions

These are assumptions to test, not conclusions:

- The existing component decomposition is likely useful.
- The existing carrier and token-direction algebra is likely useful.
- The existing row/state machinery may be useful, but rows are not assumed to
  be the final representation.
- RDKit remains useful as an input/parser tool, even if RDKit serializer policy
  is not the target.
- Exact RDKit string fixtures remain useful comparison evidence, but should not
  define South Star correctness.

## Current Surface Inventory

### Likely Shared Infrastructure

These areas may belong in both North Star and South Star work, though names may
need cleanup:

- RDKit-to-prepared-graph input bridge;
- graph traversal and walker mechanics;
- stereo component discovery;
- side domains and carrier domains;
- slash/backslash token algebra;
- semantic constraint facts and assignment filtering;
- parser-backed graph/stereo round-trip checks;
- small exploration scripts that clarify stereo constraints.

### RDKit Writer-Policy Surface

These areas may be North Star-bound or comparison-only for South Star:

- `StereoConstraintLayer::RdkitLocalWriter`;
- `StereoConstraintLayer::RdkitTraversalWriter`;
- helpers named `rdkit_writer_*`, `rdkit_marker_*`, or
  `rdkit_traversal_writer_*`;
- `RdkitTokenFlipAdjustmentObservations`;
- RDKit marker event and no-marker event filtering;
- RDKit ring-closure marker projection;
- committed-token parity filters;
- pinned RDKit exact-string fixtures and runners;
- RDKit serializer-source coverage and regression-mining tooling;
- known RDKit quirks/gaps when the expectation is string membership.

None of these should be deleted just because they look North Star-bound. First
classify them as one of:

- required comparison evidence;
- useful implementation scaffolding;
- obsolete after a semantic replacement exists;
- unrelated to South Star and safe to remove later.

### Ambiguous Surface

These need review before classification:

- `StereoMarkerPlacementRow`: could be reusable as a generic assignment table,
  or it may be too RDKit-placement-shaped.
- `StereoConstraintState`: likely useful as an idea, but its current fields may
  mix semantic and writer-policy facts.
- selected-neighbor repair/support-boundary helpers: some parts may express
  semantic carrier constraints, while others are RDKit writer compatibility.
- `NoMarker` facts: may be semantic only if the South Star policy explicitly
  treats omission as an observation; otherwise likely writer-policy.

## Clean Starting Work

The first work is intended to create visibility, not delete code.

### 1. Baseline The Branch

Run and record:

- `cargo test --lib`;
- `PYTHONPATH=python:. python3 -m unittest tests.run_exact_public_invariants -q`;
- optional RDKit parity runners only as comparison evidence.

Purpose:

- know whether this branch starts green;
- avoid confusing pre-existing failures with South Star changes.

### 2. Create A Separate South Star Test Entry Point

Add a test package or runner whose name makes the target explicit, for example:

- `tests/south_star/`;
- `tests/run_south_star_semantics.py`.

The first runner should not include pinned RDKit string parity tests.

Purpose:

- prevent RDKit exact-string fixtures from defining South Star correctness;
- allow semantic tests to fail independently while the public runtime remains
  RDKit-parity-oriented.

### 3. Add Minimal Semantic Witnesses

Start with a small fixture set, not a broad corpus:

- one isolated double-bond case;
- one oxime-like case;
- one conjugated/shared-carrier case where RDKit and semantic support may
  diverge;
- one ring-closure/basis case if a small one is available.

Each witness should state:

- input molecule;
- intended graph/stereo assignment;
- candidate South Star strings;
- whether RDKit emits each string, as comparison metadata only;
- parser/semantic round-trip expectation.

Purpose:

- create a positive target before any excision;
- make South Star-only support explicit and reviewable.

### 4. Add A Diagnostic Query Before Runtime Mutation

Before changing public enumeration behavior, expose a diagnostic/internal query
that can answer:

- which semantic component assignments survive a prefix;
- which directional tokens are semantically allowed at a boundary;
- whether a complete candidate parses to the intended graph/stereo assignment;
- which current RDKit-policy filter would accept or reject the same candidate,
  if comparison data is available.

Purpose:

- compare semantic and RDKit paths without replacing either too early;
- identify which current helpers are genuinely semantic and which are writer
  policy.

## Questions To Decide Later

Do not force these decisions in the first slice:

- Should South Star use rows, bitsets, a native propagator, or a hybrid?
- What exactly does "maximal annotation" mean for multi-candidate sides?
- Are all eligible carrier edges marked, or only assignment-selected carrier
  edges?
- How should aromatic directional bonds be handled?
- Which non-double-bond stereo forms are in scope?
- Should South Star become a public API, an internal diagnostic, or a separate
  package mode?
- Should RDKit comparison fixtures stay in this branch long term?

## Guardrail For Later Excision

Do not delete North Star-bound code until all are true:

- the code has been classified as writer-policy rather than semantic support;
- a South Star semantic test covers the relevant behavior or proves it is not
  relevant;
- any useful comparison role has been moved to a clearly optional path;
- the South Star runner and core Rust tests pass.

Prefer quarantine over deletion when classification is uncertain.

## Candidate First Slice

A conservative first implementation slice could be:

1. Baseline tests and record results in a note or commit message.
2. Add a South Star semantic test runner.
3. Add two to four tiny semantic fixtures.
4. Add helper assertions for parser-backed graph/stereo equivalence.
5. Add only the minimum diagnostic/query surface needed by those tests.

This slice should avoid:

- deleting RDKit fixtures;
- removing RDKit writer layers;
- changing public `MolToSmilesEnum` behavior;
- renaming broad runtime structures;
- claiming a final representation.

## Review Checkpoint

After the first slice, review:

- Which failures are semantic gaps?
- Which failures are only RDKit writer-parity differences?
- Which current helpers are reusable without RDKit policy?
- Which helpers should be quarantined before deletion?
- Whether the South Star target needs a narrower first scope.

Only after this review should the branch consider excising North Star-bound
runtime code.
