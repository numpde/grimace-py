# Provisional `MolToSmilesEnumS`

`MolToSmilesEnumS` is a provisional name for South Star semantic SMILES
enumeration. It is not a public package API yet.

The current public runtime remains `MolToSmilesEnum`, whose contract is exact
RDKit writer parity for the supported `canonical=False, doRandom=True` regime.
`EnumS` is being developed as a separate semantic enumerator: it should emit
SMILES strings that preserve the input molecule's graph and stereo assignment
under a named annotation policy, even when those strings are not in RDKit's
writer support.

## Contract Boundary

`MolToSmilesEnum` answers:

- what strings are in RDKit's pinned writer support for the supported runtime
  regime?

`MolToSmilesEnumS` should answer:

- what strings are semantically valid for this molecule under a documented
  South Star annotation policy?

Those are different correctness contracts. A South Star string may be valid,
parseable, and stereo-preserving while still being outside RDKit writer parity.
Conversely, an RDKit writer string may be absent from an incomplete South Star
seed enumerator until the semantic traversal surface is expanded.

## Current Status

The current implementation lives behind the private internal
`grimace._south_star` boundary and is used by tests and diagnostics. It is not
imported from `grimace.__init__` and is not a package-ready runtime surface.
The executable private contract is tracked by
`tests.helpers.south_star_domain_manifest.SouthStarDomainManifest`; fixtures and
support-gate tests use that manifest to keep policy names, fixture authorities,
feature areas, and unsupported categories aligned.

The seed enumerator:

- derives traversal candidates from the RDKit molecule graph and South Star
  component marker assignments;
- does not use fixture-positive strings or `MolToSmilesEnum` outputs as
  generation input;
- enumerates roots and child/main-branch orders for the current connected
  acyclic tree subset;
- enumerates simple single/double-bond monocycles with acyclic branches by
  choosing supported graph closure edges, traversing the remaining spanning
  tree, and emitting `ring_open` / `ring_close` events;
- composes disconnected molecules from independently supported connected
  fragment supports under an explicit all-fragment-orders policy;
- varies component-local marker assignments for the supported stereo features;
- applies local carrier-orientation rules for markers emitted through branches
  or reversed tree edges;
- emits `@` / `@@` atom-stereo tokens for the current tetrahedral-center
  subset from traversal ligand-order facts;
- renders strings from traversal events plus solved marker-slot assignments.

The fixture-backed prototype remains in `tests.helpers` as comparison support
only. It is not the implementation strategy for a package API.

## First Supported Shape

The implemented private scope is deliberately narrow:

- connected acyclic molecules;
- nonstereo single/double-bond monocycles with acyclic branches;
- the first ring-stereo monocycle subset, where directional carrier markers may
  be emitted as event-local `ring_open` marker slots, but stereo double bonds
  are not themselves used as ring-closure edges;
- disconnected molecules whose connected fragments are independently supported,
  composed with all fragment orders;
- directional double-bond stereo represented by slash/backslash carriers;
- alkene-style carriers;
- hetero imine or oxime-style carriers using the same directional semantics;
- independent stereo components;
- explicitly coupled components, including shared carrier edges;
- same-side alternate carrier edges under maximal carrier annotation.
- tetrahedral centers with exactly four ligands, including the current
  implicit-hydrogen and quaternary-center slices.

This is not yet support for all RDKit stereo surfaces, all OpenSMILES syntax,
or all legal semantic SMILES for arbitrary molecules.

## Unsupported Cases

Unsupported cases should fail before enumeration with explicit reasons. They
must not silently fall back to `MolToSmilesEnum` or return a partial support set.

Current unsupported categories include:

- query atoms or query bonds;
- unsupported bond types;
- dative or metal-containing stereo surfaces;
- fused/polycyclic rings, reported as `fused_or_polycyclic_ring`;
- ring/tetrahedral interactions, reported as `ring_tetrahedral_interaction`;
- unsaturated ring traversal and ring-closure carrier bases;
- ring stereo;
- aromatic rings, reported as `aromatic_ring_surface`;
- aromatic directional surfaces, reported separately as
  `aromatic_directional_surface`;
- any component whose marker equations cannot be stated locally.

These unsupported categories are classification boundaries, not implementation
targets by themselves. The current near-term ring work is simple monocycles and
explicit ring-closure stereo carrier bases. Aromatic surfaces, fused/polycyclic
ring systems, and ring/tetrahedral interactions require separate semantic
models before enumeration should widen to them.

## Annotation Policy

The current South Star seed targets maximal eligible-carrier annotation:

- if a carrier can express a semantic stereo feature, emit a directional marker
  on that carrier;
- expose marker directions that preserve at least one surviving local semantic
  assignment;
- keep annotation policy separate from component extraction and marker-equation
  solving.

This policy is intentionally modular. Future package work may add minimal,
canonical, or writer-like annotation policies without changing the component
fact model.

The private runtime currently passes a small immutable policy set through the
graph-native EnumS path:

- annotation policy: `maximal_eligible_carrier`;
- fragment-order policy: `all_fragment_orders`;
- output-order policy: `first_occurrence_deduplication`.

These names are engineering contract labels, not a plugin framework. They make
the current behavior explicit in diagnostics and tests while preserving the
existing default support order.

## Complexity Diagnostics

Graph-native EnumS results expose private generation diagnostics:

- fragment count and per-fragment support counts;
- traversal skeleton count;
- marker-slot count;
- local assignment count;
- solved assignment count;
- estimated product size before output-order deduplication.

These diagnostics are engineering guardrails. They help detect accidental
complexity shifts or hidden support-generation changes, but they are not
semantic authority. Fixture support and independent oracles remain the
correctness evidence.

## Conformance Evidence

The current South Star oracle separates four checks:

- RDKit parseability, used as parser evidence;
- non-isomeric graph equivalence;
- isomeric stereo equivalence;
- RDKit-independent annotation conformance for the current directional-marker
  subset.

The annotation conformance basis is
`south_star_current_subset_directional_marker_grammar`. It is narrower than a
full OpenSMILES parser and covers the atom, branch, double-bond, and
slash/backslash forms emitted by the seed enumerator.

Exact support evidence is split by domain:

- `tests/fixtures/south_star_exact_first_domain/first_domain_v1.json` pins the
  connected acyclic directional-marker first domain and is checked against an
  independent test oracle;
- `tests/fixtures/south_star_expanded_support/expanded_domain_v1.json` pins
  expanded semantic support. Saturated-monocycle, ring-stereo monocycle, and
  disconnected-composition cases are checked against independent test oracles.
  The ring-stereo oracle checks closure-event marker slots and parity equations
  by slot id; unsaturated nonstereo monocycles and tetrahedral centers are still
  graph-native regression support with RDKit parse-back graph/stereo equivalence
  as evidence.

RDKit parseability is useful evidence, but it is not the definition of South
Star validity.

## Comparison Diagnostics

Diagnostics compare South Star semantic output with RDKit writer-parity support
using explicit categories:

- `intersection`: accepted by both surfaces;
- `SouthStarOnly`: semantically valid under the South Star policy but outside
  RDKit parity support;
- `RDKitParityOnly`: in RDKit parity support but absent from the current South
  Star seed or policy.

These categories are diagnostics. Equality with RDKit support is not the South
Star goal.

## Package-Readiness Gap

Before `MolToSmilesEnumS` can become a documented package API, the graph-native
enumerator needs a broader molecule and syntax surface:

- polycyclic ring traversal;
- independent completeness oracles for unsaturated nonstereo ring traversal;
- broader ring-closure marker bases, including stereo-double-bond closure edges;
- selectable disconnected-fragment policies beyond the current all-orders
  private default;
- atom text beyond the current organic-subset seed;
- aromatic ring and aromatic directional-surface models, if any;
- a ring/tetrahedral interaction model;
- broader validation of local branch-orientation equations against more
  adversarial carrier topologies;
- independent completeness oracles beyond the first connected acyclic
  directional-marker and saturated-monocycle domains;
- explicit fail-fast checks for every unsupported molecule class;
- complexity diagnostics that expose component counts, local assignment counts,
  affected component counts, and estimated product size.

The provisional name `MolToSmilesEnumS` should remain internal until that
surface is implemented and reviewed. The current private implementation is a
construction path for that future surface, not a public API commitment.
