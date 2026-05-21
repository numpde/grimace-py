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

The provisional private API boundary is
`grimace._south_star.api.mol_to_smiles_enum_s_private(mol, *, policy_set=...)`.
It accepts an RDKit `Mol`, builds `SouthStarMoleculeFacts`, applies the South
Star support gates before enumeration, uses `DEFAULT_SOUTH_STAR_POLICY_SET`
unless a policy set is passed explicitly, and returns graph-native outputs plus
policy names and generation diagnostics. It is deliberately not exported from
`grimace.__init__`.

`SouthStarMoleculeFacts` is the current semantic fact boundary. It owns the
support-gate report, atom and bond text facts, graph topology, extracted
semantic stereo components, carrier opportunities, and tetrahedral center
facts. Runtime generation, component support state, diagnostics, and test
helpers should consume those named facts instead of re-deriving parallel local
views.

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
  be emitted as event-local `ring_open` marker slots and where the stereo double
  bond itself may be used as the ring-closure edge;
- disconnected molecules whose connected fragments are independently supported,
  composed with all fragment orders;
- directional double-bond stereo represented by slash/backslash carriers;
- alkene-style carriers;
- hetero imine or oxime-style carriers using the same directional semantics;
- independent stereo components;
- explicitly coupled components, including shared carrier edges;
- same-side alternate carrier edges under maximal carrier annotation;
- tetrahedral centers with exactly four ligands, including the current
  implicit-hydrogen and quaternary-center slices;
- explicit bracket hydrogen atoms in the narrow neutral, non-isotopic,
  non-radical form emitted as `[H]`.

Atom text is scoped by the `grimace._south_star.atom_text` policy boundary.
The current contract records isotope, element symbol, chirality token,
explicit-hydrogen count, formal charge, radical electron count, atom-map
number, and aromaticity as explicit fields. The only supported bracket atom
texts today are neutral `[H]` and the tetrahedral carbon forms emitted by the
current atom-stereo slice: `[C@H]`, `[C@@H]`, `[C@]`, and `[C@@]`. Isotope,
charge, radical, and atom-map modifiers are deliberately deferred and must fail
before enumeration with named unsupported categories.

This is not yet support for all RDKit stereo surfaces, all OpenSMILES syntax,
or all legal semantic SMILES for arbitrary molecules.

## Unsupported Cases

Unsupported cases should fail before enumeration with explicit reasons. They
must not silently fall back to `MolToSmilesEnum` or return a partial support set.

Current unsupported categories include:

- query atoms or query bonds;
- unsupported bond types;
- atom isotopes, charges, radicals, and atom maps, reported as
  `unsupported_atom_isotope`, `unsupported_atom_charge`,
  `unsupported_radical_atom`, and `unsupported_atom_map`;
- dative or metal-containing stereo surfaces;
- fused/polycyclic rings, reported as `fused_or_polycyclic_ring`;
- ring/tetrahedral interactions, including ring-member chiral atoms and
  ring-adjacent chiral atoms whose ligand order depends on a ring path, reported
  as `ring_tetrahedral_interaction`;
- ring stereo outside the supported monocycle subset, reported as
  `ring_stereo`;
- aromatic rings, reported as `aromatic_ring_surface`;
- aromatic directional surfaces, reported separately as
  `aromatic_directional_surface`;
- any component whose marker equations cannot be stated locally.

These unsupported categories are classification boundaries, not implementation
targets by themselves. The current near-term ring work is simple monocycles and
explicit ring-closure stereo carrier bases. Aromatic surfaces, fused/polycyclic
ring systems, and ring/tetrahedral interactions require separate semantic
models before enumeration should widen to them.

The current aromatic stance is fail-fast exclusion of aromatic RDKit molecule
facts. See `notes/040_south_star_aromatic_boundary.md` for the alternatives and
why kekule-looking input text is not enough when normal RDKit parsing still
sets aromatic flags.

The current polycyclic stance is also fail-fast. Ring-system facts are named,
but fused/polycyclic traversal still needs graph-native choices for spanning
trees, closure edges, label allocation, and closure-event ordering before
support should widen. See `notes/041_south_star_ring_system_model.md`.

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

Policy candidates are named in
`grimace._south_star.annotation_policy.SOUTH_STAR_ANNOTATION_POLICY_CANDIDATES`.
Only `maximal_eligible_carrier` is the current default. `minimal_sufficient` and
`canonical_semantic` are deferred concepts, `rdkit_writer_like` is a comparison
candidate rather than semantic authority, and `no_marker_policy_stub` exists
only to test the policy boundary.

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
- fragment-order count;
- stereo-component count;
- traversal skeleton count;
- marker-slot count;
- local assignment count;
- solved assignment count;
- raw output count before output-order deduplication;
- output count after output-order deduplication;
- deduplication drop count and deduplicated-output ratio;
- estimated product size before output-order deduplication.

These diagnostics are engineering guardrails. They help detect accidental
complexity shifts or hidden support-generation changes, but they are not
semantic authority. Fixture support is evidence against the shared South Star
model. Temporary witness helpers remain useful, but they are not permanent
sources of truth.

Test-only complexity diagnostics may also record per-layer timings for fact
extraction, generation, and conformance checks. Those timings are inspectable
metadata, not correctness thresholds.

Representative non-timing budgets live in
`tests/fixtures/south_star_complexity_budgets/generation_diagnostics_v1.json`.
They pin reviewable diagnostic counts for selected domains so support-surface
expansion cannot hide multiplicative growth behind generic consistency checks.

`tests.helpers.south_star_adversarial_corpus` generates deterministic triage
candidates for roots, branch order, carrier placement, shared carriers, ring
closures, tetrahedral ligand order, disconnected fragments, and unsupported
feature triggers. These candidates are diagnostic inputs only; they do not
define expected support sets.

## Package-Readiness Harness

The private package-readiness command is:

```bash
PYTHONPATH=python:. python3 -m unittest tests.run_south_star_package_readiness -q
```

It aggregates the current pre-public `MolToSmilesEnumS` checks:

- exact support equality where temporary witness helpers exist;
- graph/stereo parse-back checks for regression-backed fixture domains;
- unsupported-feature gate checks;
- policy-name diagnostics;
- generation complexity guardrails.

The harness also exposes a readiness matrix that separates
unified-reference-backed cases, temporary witness-backed cases, and
graph-native regression-backed cases. That distinction is part of the current
maturity signal: temporary witnesses and regression fixtures are useful
evidence, but neither is the final South Star source of truth.

## Conformance Evidence

The current South Star conformance layer separates four checks:

- RDKit parseability, used as parser evidence;
- non-isomeric graph equivalence;
- isomeric stereo equivalence;
- RDKit-independent grammar membership for the declared South Star subset.

The grammar-conformance basis is `south_star_declared_subset_grammar_v1`. It is
narrower than a full OpenSMILES parser and currently covers the atom, branch,
dot-fragment, ring-label, bracket-atom, double-bond, slash/backslash, and
tetrahedral-token forms emitted by the seed enumerator. It is a syntax
membership check only; graph identity and stereo identity remain separate
semantic checks.

Graph and stereo identity currently use a named parse-back boundary:
`rdkit_parser_dependency`, `rdkit_canonical_nonisomeric_parseback`, and
`rdkit_canonical_isomeric_parseback`. That makes the parser dependency explicit:
RDKit parsing is evidence for current graph/stereo identity checks, not the
definition of grammar membership or support completeness.

Exact support evidence is split by domain:

- `tests/fixtures/south_star_exact_first_domain/first_domain_v1.json` pins the
  connected acyclic directional-marker first domain and is checked against a
  temporary witness helper that emits shared traversal/slot records;
- `tests/fixtures/south_star_expanded_support/expanded_domain_v1.json` pins
  expanded semantic support. Saturated and unsaturated nonstereo-monocycle,
  ring-stereo monocycle, and disconnected-composition cases are checked against
  temporary witness helpers that consume shared traversal, slot, and
  fragment-composition records; current star-shaped tetrahedral-center cases
  are checked against a temporary tetrahedral witness helper. The nonstereo
  monocycle witness checks broken ring-edge choice, tree traversal order,
  closure digit placement, and closure bond text. The ring-stereo witness
  checks closure-event marker slots, central-double-bond closure events, and
  parity-equation projections by slot id.

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

The diagnostic helper is `tests.helpers.south_star_comparison`. It may import
the public `grimace.MolToSmilesEnum` writer-parity surface, but core South Star
helpers and semantic tests must not. Diagnostic reports are allowed to expose
differences; those differences are metadata, not semantic failures.

## Public API Promotion Gate

`MolToSmilesEnumS` must remain private until the promotion gate is satisfied.
The gate is intentionally stricter than "the current fixtures pass": each
promoted behavior needs a declared domain, syntax evidence, semantic identity
evidence, support-completeness evidence, unsupported-case coverage, diagnostics,
and release-facing documentation.

The executable readiness entry point is:

```bash
PYTHONPATH=python:. python3 -m unittest tests.run_south_star_package_readiness -q
```

The gate has these required parts:

| Gate | Required evidence |
| --- | --- |
| Private boundary | `MolToSmilesEnumS` is not exported from `grimace.__init__` until all other gates pass. |
| Supported domain manifest | Supported feature areas, support-evidence classes, policy names, and unsupported categories are declared in one manifest. |
| Grammar conformance | Every output is in the declared South Star grammar subset. |
| Semantic identity | Every output parses back to the intended graph and stereo assignment under the named parser dependency. |
| Support evidence classification | Every promoted supported domain classifies support evidence as unified-reference-backed, temporary witness, or graph-native regression witness. Temporary witnesses are not final authority. |
| Unsupported-category completeness | Every out-of-domain molecule class fails before enumeration with a named category. |
| Complexity guardrails | Generation diagnostics expose fragment counts, traversal skeletons, marker slots, assignment counts, solved counts, and estimated products for representative cases. |
| Documentation | Docs name the contract, policy set, supported domains, unsupported domains, parser dependency, and difference from RDKit writer parity. |
| CI command | The readiness runner is listed as the command to run before public export. |
| Release notes | Any release that exports `MolToSmilesEnumS` must state the semantic contract and explicitly distinguish it from `MolToSmilesEnum` RDKit writer parity. |

The test-side gate list is
`tests.south_star.test_package_readiness.SOUTH_STAR_PUBLIC_API_PROMOTION_GATES`.
It records both executable commands and explicit review items so the promotion
bar is not reduced to informal confidence.

## Package-Readiness Gap

Before `MolToSmilesEnumS` can become a documented package API, the graph-native
enumerator needs a broader molecule and syntax surface:

- polycyclic ring traversal;
- selectable disconnected-fragment policies beyond the current all-orders
  private default;
- bracket atom text beyond the current neutral explicit-hydrogen and
  tetrahedral-center slices, with isotope, charge, radical, and atom-map
  modifiers still deliberately deferred;
- aromatic ring and aromatic directional-surface models, if any;
- a ring/tetrahedral interaction model;
- broader validation of local branch-orientation equations against more
  adversarial carrier topologies;
- shared-reference support evidence beyond the current temporary witness cases;
- explicit fail-fast checks for every unsupported molecule class;
- complexity diagnostics that expose component counts, local assignment counts,
  affected component counts, and estimated product size.

The provisional name `MolToSmilesEnumS` should remain internal until that
surface is implemented and reviewed. The current private implementation is a
construction path for that future surface, not a public API commitment.
