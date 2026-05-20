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

The seed enumerator:

- derives traversal candidates from the RDKit molecule graph and South Star
  component marker assignments;
- does not use fixture-positive strings or `MolToSmilesEnum` outputs as
  generation input;
- enumerates roots and child/main-branch orders for the current connected
  acyclic tree subset;
- varies component-local marker assignments for the supported stereo features;
- applies local carrier-orientation rules for markers emitted through branches
  or reversed tree edges;
- renders strings from traversal events plus solved marker-slot assignments.

The fixture-backed prototype remains in `tests.helpers` as comparison support
only. It is not the implementation strategy for a package API.

## First Supported Shape

The intended first supported class is deliberately narrow:

- one connected acyclic molecule;
- directional double-bond stereo represented by slash/backslash carriers;
- alkene-style carriers;
- hetero imine or oxime-style carriers using the same directional semantics;
- independent stereo components;
- explicitly coupled components, including shared carrier edges;
- same-side alternate carrier edges under maximal carrier annotation.

This is not yet support for all RDKit stereo surfaces, all OpenSMILES syntax,
or all legal semantic SMILES for arbitrary molecules.

## Unsupported Cases

Unsupported cases should fail before enumeration with explicit reasons. They
must not silently fall back to `MolToSmilesEnum` or return a partial support set.

Current unsupported categories include:

- query atoms or query bonds;
- tetrahedral atom stereo;
- unsupported bond types;
- dative or metal-containing stereo surfaces;
- disconnected molecules;
- ring stereo and ring-closure carrier bases;
- aromatic directional surfaces;
- any component whose marker equations cannot be stated locally.

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

- ring-closure traversal and marker bases;
- disconnected molecule policy;
- atom text beyond the current organic-subset seed;
- supported aromatic directional surfaces, if any;
- broader validation of local branch-orientation equations against more
  adversarial carrier topologies;
- explicit fail-fast checks for every unsupported molecule class;
- complexity diagnostics that expose component counts, local assignment counts,
  affected component counts, and estimated product size.

The provisional name `MolToSmilesEnumS` should remain internal until that
surface is implemented and reviewed. The current private implementation is a
construction path for that future surface, not a public API commitment.
