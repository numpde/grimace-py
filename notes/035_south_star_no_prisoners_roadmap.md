# South Star No-Prisoners Roadmap

Branch: `south-star`

Date: 2026-05-20

## Purpose

This note fixes the stricter South Star stance.

South Star should not become a looser RDKit writer clone, a fixture-backed
generator, or a post-filtered string sampler. The target is a principled
semantic SMILES language with a declared grammar subset, declared annotation
policy, and exact-by-construction enumeration.

RDKit writer parity remains important comparison metadata. It is not the South
Star oracle.

## Target Contract

Provisional surface:

```text
MolToSmilesEnumS(mol, policy=...)
```

Intended meaning:

```text
enumerate exactly the strings in the declared South Star SMILES sublanguage
that parse to the input molecule's intended graph and stereo semantics under
the declared annotation policy
```

The contract has three parts:

- a molecule domain;
- a SMILES grammar/traversal sublanguage;
- an annotation policy.

The implementation is only correct relative to those three declarations. If a
molecule, syntax feature, or annotation policy is not declared, it must fail
fast with a named unsupported reason.

## Non-Negotiables

- No fixture-positive generation.
- No RDKit writer support as the semantic oracle.
- No hidden RDKit serializer quirks in semantic code.
- No runtime path that generates junk and then depends on parsing/filtering to
  recover correctness.
- No partial support sets for unsupported surfaces.
- Every emitted string must belong by construction.
- Completeness must be measured against the declared grammar and annotation
  policy, not against a handful of examples.
- RDKit parseability is evidence, not definition.
- RDKit writer parity is comparison metadata, not pass/fail authority for South
  Star.

## Current Position

The current branch has crossed the line from exploration into a coherent seed:

- South Star is documented as separate from RDKit writer parity.
- `MolToSmilesEnumS` is documented as a provisional internal name.
- The graph-native seed no longer uses fixture-positive strings or
  `MolToSmilesEnum` output as generation input.
- The seed enumerates all roots and child/main branch-order choices for the
  current connected acyclic tree subset.
- Directional marker assignments come from component marker assignments.
- Local carrier-orientation rules now handle branch/reversed traversal cases.
- Coupled shared carriers have a local traversal admissibility rule.
- Generated outputs are checked by the South Star conformance harness.

That is still not South Star proper. It is the first executable vertical slice.

## First Closed Domain

The first domain should be frozen before adding more feature families:

- one connected acyclic molecule;
- organic-subset atom text currently supported by the seed;
- single and double bonds currently supported by the seed;
- directional double-bond stereo only;
- alkene, imine, and oxime-style carrier semantics;
- independent directional stereo components;
- coupled components through explicitly modeled shared carriers;
- maximal eligible-carrier annotation policy.

Everything else should be rejected before enumeration.

Initial unsupported categories:

- query atoms or query bonds;
- tetrahedral atom stereo;
- unsupported atom text;
- unsupported bond types;
- dative or metal-containing stereo surfaces;
- disconnected molecules;
- rings and ring-closure marker bases;
- aromatic directional surfaces;
- any component whose marker equations cannot be stated locally.

## Proper Layer Split

South Star should be implemented as five separable layers.

### 1. Molecule Facts

Owns graph and semantic extraction:

- atoms;
- bonds;
- bond orders;
- stereo features;
- eligible marker carriers;
- carrier-feature incidence;
- independent/coupled component decomposition.

This layer must not know how strings are rendered.

### 2. Traversal Language

Owns the syntax skeleton:

- roots;
- child ordering;
- main path versus branch choices;
- later: ring closures;
- later: disconnected fragment policy.

This layer should produce traversal events, not final strings.

### 3. Marker Slot Model

Owns all slash/backslash positions made available by traversal:

- slot id;
- graph edge;
- traversal direction;
- syntax position;
- adjacent feature context;
- branch/main/reversed-edge parity contribution.

Every emitted `/` or `\` must come from a named marker slot.

### 4. Stereo Constraint Model

Owns equations over marker slots.

For the first domain, this should reduce to parity constraints:

```text
slot_bit xor traversal_parity xor feature_phase == required_stereo_phase
```

Shared carriers should be ordinary coupled equations. They should not be
special traversal folklore.

### 5. Annotation Policy

Owns which satisfying marker assignments belong to the language.

Initial policy:

```text
maximal eligible-carrier annotation
```

That means every eligible carrier slot required by the policy is marked. Later
policies may be minimal, canonical, or RDKit-writer-like, but they must share
the same fact, traversal, slot, and constraint layers.

## Z3 And Runtime Solver

Z3 should be kept as an independent specification oracle, not as the runtime
dependency.

Use Z3 to:

- encode facts, marker slots, and parity equations for small fixtures;
- enumerate satisfying assignments independently of the runtime solver;
- compare against the custom solver for the same traversal skeleton;
- expose contradictions or underconstrained cases.

Runtime should eventually use a small purpose-built parity/bitset solver for
the declared first domain. The runtime solver should be simple enough to audit.

## Completeness Standard

For the first closed domain, completeness means:

- every supported traversal skeleton is generated;
- every required marker slot for the policy is present;
- every satisfying assignment is rendered;
- no non-satisfying assignment is rendered;
- every rendered string passes graph/stereo conformance;
- the implementation output equals an independent reference enumerator for
  small exhaustive cases.

Fixtures are witnesses and regression cases. They are not the source of truth.

## Immediate No-Regret Implementation Path

### Phase 0: Freeze Domain And Gates

Before reshaping traversal, make the first supported domain executable:

- one public helper or internal guard validates the first closed domain;
- every unsupported category has a named failure reason;
- tests prove unsupported cases fail before enumeration;
- no unsupported case returns a partial support set.

This phase should happen before the event/slot split. Without it, later
architecture work can accidentally widen the language without declaring a
contract.

### Phase 1: Event/Slot Boundary

Introduce explicit traversal events and marker-slot records.

Keep the current emitted strings unchanged, but add tests asserting:

- every emitted directional marker came from a marker slot;
- every marker slot has edge, direction, syntax position, and feature context;
- no slash/backslash is inserted directly by string-local logic.

### Phase 2: Equation Builder

Build parity equations from molecule facts plus traversal marker slots.

The existing carrier-orientation behavior should become data:

- traversal parity;
- feature phase;
- required stereo phase;
- shared-carrier incidence.

The goal is to make branch/reversed-edge handling inspectable as equations.

### Phase 3: Independent Solver Check

Add a Z3-backed reference for small current-domain fixtures.

For each fixture:

- build traversal slots;
- build equations;
- enumerate satisfying assignments with Z3;
- enumerate satisfying assignments with the custom solver;
- assert equality at the assignment level before rendering strings.

This is the main guardrail against replacing one procedural patch with another.

### Phase 4: Runtime-Like Solver

Implement the custom parity solver over the same equation records.

It should expose:

- component count;
- local assignment counts;
- coupling causes;
- affected components per marker decision;
- estimated product size.

The solver must preserve the component-factorized complexity picture.

### Phase 5: Renderer

Render strings from traversal events plus solved marker assignments.

The renderer should not decide stereo semantics. It should only print the
already-selected tokens.

### Phase 6: Exhaustive First-Domain Tests

For small connected acyclic cases:

- enumerate all traversal skeletons independently;
- enumerate all marker assignments independently;
- compare implementation support exactly;
- verify graph/stereo conformance for every output;
- verify selected negative witnesses stay excluded.

### Phase 7: Internal Package Boundary

Only after the event/slot/equation/solver/renderer split is stable, move the
implementation behind an internal package boundary such as `grimace._south_star`
or the eventual Rust/Python equivalent.

Do not expose `MolToSmilesEnumS` publicly yet.

## Major Generalization Milestones

### Rings

Rings are the first serious expansion.

They require:

- ring-closure traversal events;
- marker slots detached from simple parent/child traversal;
- ring-closure numbering policy;
- carrier equations that can reference closure syntax positions.

Rings should be added as traversal-language work, not as string patching.

### Disconnected Molecules

Disconnected support needs a fragment policy:

- semantic fragment independence;
- output ordering policy;
- optional writer-like ordering later.

Do not silently inherit RDKit fragment-order semantics unless an explicit
writer-like policy asks for it.

### Tetrahedral Atom Stereo

Tetrahedral stereo is a different constraint family.

Do not mix it into directional double-bond carrier work until the directional
layer is closed, tested, and internally packaged.

## What Not To Do

- Do not expand public API surface while the implementation still lives as test
  helpers.
- Do not add more local branch/case repairs without translating them into slot
  facts or equations.
- Do not use RDKit string support to define South Star completeness.
- Do not treat parsed-object equivalence as a substitute for a declared
  grammar and annotation policy.
- Do not add rings by mutating rendered strings after traversal.
- Do not add atom stereo to the same solver before directional parity is closed.

## Next Concrete Task

Freeze the first closed domain and add explicit fail-fast gates.

Acceptance:

- supported current fixtures still enumerate;
- unsupported examples fail before traversal;
- failure reasons are named and testable;
- no unsupported case returns partial output.

Then create the event/slot boundary for the current tree subset.

Acceptance:

- traversal can produce an inspectable event stream;
- marker slots are explicit records;
- current output strings are unchanged;
- tests prove directional markers are rendered only from marker slots;
- the note-level layer split is visible in code names.

This is the bridge from a working seed to South Star proper.
