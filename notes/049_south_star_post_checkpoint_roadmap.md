# South Star Post-Checkpoint Roadmap

Branch: `south-star`

Date: 2026-05-21

Task: `South Star 130: Deliberate post-checkpoint granular roadmap`

## Current Baseline

After the 129/129a/129b/129c/129d/129e sequence, the readiness matrix is no
longer blocked by stale shared-pipeline metadata.

Current matrix after the first-domain, markerless acyclic-tree,
disconnected-composition, nonstereo-monocycle, ring-stereo monocycle,
star-shaped tetrahedral atom-stereo, ring/tetrahedral, and nonstereo
polycyclic authority promotions, plus polycyclic ring-stereo promotion:

- unified-reference-backed cases: `35`
- shared-pipeline promotion candidates: `35`
- temporary-witness-backed cases: `0`
- graph-native regression-backed cases: `0`
- public API blocker cases: `0`

Remaining blocker count:

- `support_authority_is_not_unified_reference`: `0`

That is the right next frontier. The immediate post-checkpoint work should be
authority migration under the one-truth reference model, not broad API export.

## Roadmap Shape

The straight line is:

1. grow unified-reference authority from the smallest markerless cases outward;
2. fold temporary witness concepts into the shared fact/event/constraint/
   renderer spine;
3. keep RDKit writer parity as comparison metadata only;
4. add a separate semantic-enumerator benchmark artifact before any speed
   claim;
5. run the export gate only after the public blockers are materially reduced
   and the remaining unsupported surface is deliberately scoped.

## Next Backlog Rows

### 130a: Generalize Markerless Atom-Text Authority

Goal: extend the single-atom atom-text proof to connected markerless atom/bond
text cases.

Initial targets were:

- `explicit_bracket_hydrogen_h2`;
- `charged_atom_text_methylammonium`;
- neutral organic-subset acyclic markerless cases.

This should derive support from molecule facts, atom-text facts, bond-text
facts, traversal events, and renderer output. It should not use expected
fixture strings as generation input.

Status: complete for the current markerless acyclic-tree fixture slice:
ethanol, isopropanol, acetone, and acetonitrile are now
`unified_reference_markerless_acyclic_tree`.

### 130b: Promote First-Domain Directional Stereo Authority

Goal: migrate the acyclic directional-stereo first domain from temporary
witness authority to unified-reference authority.

This is the first real stereo proof. It must show that shared traversal events,
marker slots, component facts, parity equations, annotation policy, solver
assignments, and renderer output jointly define the full support set.

Status: complete for the current first-domain fixture. The authority is now
`unified_reference_first_domain_directional_bond_stereo`; the independent
first-domain witness remains only as cross-check evidence.

### 130c: Promote Disconnected Composition Authority

Goal: make disconnected composition unified-reference-backed once each
fragment's support authority is itself known.

The composition proof should be separate from per-fragment support: all fragment
orders, Cartesian product of fragment supports, dot rendering, and
first-occurrence deduplication.

Status: complete for the current two disconnected fixtures.
`disconnected_stereo_fragment_and_atom` is
`unified_reference_disconnected_composition` because its first-domain stereo
fragment and markerless atom fragment are unified-reference-backed.
`markerless_disconnected_ring_and_atom` is also
`unified_reference_disconnected_composition` because the ring fragment is now
nonstereo-monocycle unified-reference-backed.

### 130d: Promote Ring Traversal Authority

Goal: fold saturated monocycles, unsaturated nonstereo monocycles,
nonstereo-polycyclic skeletons, and the current ring-stereo monocycle slice
into the shared traversal/ring-label model.

This is larger than markerless acyclic work because closure-edge choice,
first-encounter labels, closure bond text, and possible marker slots must all
be first-class shared records.

Status: partial. Saturated and unsaturated nonstereo simple-monocycle cases are
now `unified_reference_nonstereo_monocycle_ring_traversal`. The current
ring-stereo monocycle fixture is also
`unified_reference_ring_stereo_monocycle_marker_obligations`. These proofs
derive support from molecule facts, shared connected-graph traversal plans,
ring-closure labels, closure bond text, marker slots where needed, parity
equations where needed, renderer events, and first-occurrence deduplication.
Ring/tetrahedral interactions are now promoted separately. The first
nonstereo-polycyclic skeleton is also promoted as
`unified_reference_nonstereo_polycyclic_closure_traversal`; polycyclic ring
stereo is now promoted as
`unified_reference_polycyclic_ring_stereo_marker_obligations`.

### 130e: Promote Tetrahedral Atom-Stereo Authority

Goal: move the star-shaped tetrahedral center cases from temporary witness
authority to shared atom-stereo obligations over traversal ligand order.

This should prove that `@` / `@@` output is derived from typed ligand-order
facts and renderer obligations, not from fixture-local witness logic.

Status: complete for the current non-ring star-shaped tetrahedral cases:
`implicit_h_tetrahedral_center` and `quaternary_tetrahedral_center` are now
`unified_reference_tetrahedral_atom_stereo_obligations`. The proof records
derive `@` / `@@` from source ligand order, traversal/emitted ligand order,
implicit-H placement, renderer inputs, and semantic parse-back. Ring/tetrahedral
interactions are now promoted as
`unified_reference_ring_tetrahedral_monocycle_obligations`, using the same
tetrahedral proof inputs composed with shared ring traversal closure events and
labels.

### 130f: Define EnumS Benchmark Artifact

Goal: create the first explicit semantic-enumerator benchmark plan/artifact.

This is not needed for semantic correctness, but it is required before any
release-facing EnumS speed claim. The artifact should record command, machine,
versions, repeats, case set, policy set, domain labels, output counts, and
timings.

### 130g: Harden RDKit Comparison Boundary

Goal: keep RDKit writer parity diagnostics useful but non-authoritative.

Comparison labels should keep distinguishing intersection, SouthStarOnly, and
RDKitParityOnly. Any package-facing docs must say equality with RDKit writer
support is not the South Star goal.

### 130h: Re-run Export Gate

Goal: process `South Star 118` only after enough authority migration has landed
to make an export/no-export decision meaningful.

If blockers remain numerous, the gate should close as no-export with exact
remaining blockers. If blockers are reduced to a deliberately scoped subset,
the gate can decide whether an experimental private-or-public surface is
acceptable.
