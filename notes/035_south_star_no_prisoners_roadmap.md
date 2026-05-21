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

The South Star reference should also not become a collection of feature-local
mini-oracles. Per-domain oracles are acceptable only as temporary witnesses
while a feature boundary is being understood. The target is one mathematical
support model: one molecule fact model, one traversal/event language, one
constraint model, one annotation-policy layer, and one renderer. Test cases
exercise that model; they do not define it.

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
- No case-by-case authority as the long-term correctness model.
- No runtime path that generates junk and then depends on parsing/filtering to
  recover correctness.
- No partial support sets for unsupported surfaces.
- Every emitted string must belong by construction.
- Completeness must be measured against the declared grammar and annotation
  policy, not against a handful of examples or feature-specific oracle scripts.
- RDKit parseability is evidence, not definition.
- RDKit writer parity is comparison metadata, not pass/fail authority for South
  Star.

## One-Truth Reference Model

The long-term South Star reference is a single mathematical support model. It
should answer:

```text
given molecule facts, traversal events, semantic constraints, and an annotation
policy, which rendered strings belong to the declared South Star language?
```

It should have these shared inputs and outputs:

- `SouthStarMoleculeFacts` as the only molecule/semantic fact boundary;
- a traversal/event stream as the only syntax skeleton;
- marker, atom-stereo, ring-closure, fragment, and atom-text obligations as
  typed constraints over that stream;
- one solver/enumerator over those constraints;
- one annotation policy deciding which satisfying assignments are in the
  language;
- one renderer that prints only from events plus solved assignments.

Feature families such as directional double-bond stereo, tetrahedral atom
stereo, ring closure syntax, disconnected fragments, bracket atom text, and
aromatic text are constraint/fact families inside this model. They are not
separate correctness authorities.

The reference implementation may have internal modules by feature for
readability. That modularity is acceptable only if the modules feed the same
fact/event/constraint/solver/renderer pipeline. A tetrahedral helper may derive
tetrahedral constraints; it should not define a separate tetrahedral support
universe. A ring helper may derive closure events; it should not define a
separate ring support universe.

## Witnesses Versus Authority

Fixtures and per-feature oracle helpers currently serve a useful purpose:

- expose gaps in unsupported surfaces;
- pin representative strings while the general model is incomplete;
- prevent regressions during refactors;
- make feature-specific reasoning reviewable.

They are scaffolding. They should be treated as witnesses for the one-truth
model, not as the final source of truth.

The intended migration is:

1. keep existing feature-local oracle tests as guardrails;
2. extract the common fact/event/constraint concepts they duplicate;
3. replace per-domain expected-support authority with the unified support
   model;
4. keep the old fixtures as regression witnesses against the unified model;
5. delete or demote feature-local oracles once they no longer provide
   independent information.

Adding another feature-local oracle is acceptable only when it is explicitly
temporary and the task also names how its concepts fold into the unified model.
If a new oracle would just add another permanent mini-world, do not add it.

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

South Star should be implemented as separable layers under one reference model.

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

### 6. Renderer

Owns final text emission from traversal events plus solved assignments.

The renderer should not make semantic choices, search for repairs, or decide
which strings belong. It prints the single model's accepted event/assignment
pairs.

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
- the implementation output equals the one-truth reference model for small
  exhaustive cases.

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

### Phase 3: Unified Reference Model

Build the first executable one-truth reference model for the current closed
domain. Z3 may be used as the backend for small exhaustive cases, but the
important point is not Z3 specifically. The important point is that all feature
families feed the same fact/event/constraint vocabulary.

For each fixture:

- build traversal slots;
- build typed constraints;
- enumerate satisfying assignments with the reference model;
- enumerate satisfying assignments with the custom solver;
- assert equality at the assignment level before rendering strings.

This is the main guardrail against replacing one procedural patch with another.
It is also the guardrail against replacing the South Star target with a growing
set of case-by-case independent oracles.

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

- enumerate all traversal skeletons through the shared traversal/event model;
- enumerate all assignments through the shared constraint model;
- compare implementation support exactly;
- verify graph/stereo conformance for every output;
- verify selected negative witnesses stay excluded.

Feature-specific witnesses should become rows in these tests, not separate
support authorities.

### Phase 7: Internal Package Boundary

Only after the event/slot/equation/solver/renderer split is stable, move the
implementation behind an internal package boundary such as `grimace._south_star`
or the eventual Rust/Python equivalent.

Do not expose `MolToSmilesEnumS` publicly yet.

## Major Generalization Milestones

Every milestone below must widen the same one-truth model. The sequence is not
"add one more oracle." The sequence is "add one more feature family to the
shared fact/event/constraint language, then prove the existing witnesses and new
adversarial witnesses follow from that model."

### Rings

Rings are the first serious expansion.

They require:

- ring-closure traversal events;
- marker slots detached from simple parent/child traversal;
- ring-closure numbering policy;
- carrier equations that can reference closure syntax positions.

Rings should be added as traversal-language work, not as string patching.
Ring fixtures should test closure-event choices in the shared model, not define
a separate ring oracle permanently.

### Disconnected Molecules

Disconnected support needs a fragment policy:

- semantic fragment independence;
- output ordering policy;
- optional writer-like ordering later.

Do not silently inherit RDKit fragment-order semantics unless an explicit
writer-like policy asks for it.

### Tetrahedral Atom Stereo

Tetrahedral stereo is a different constraint family.

It should still feed the same event/constraint/reference pipeline. Do not leave
tetrahedral support as a separate fixture-backed mini-oracle once the unified
model can express atom-stereo ligand-order constraints.

## What Not To Do

- Do not expand public API surface while the implementation still lives as test
  helpers.
- Do not add more local branch/case repairs without translating them into slot
  facts or equations.
- Do not keep adding permanent per-domain support authorities.
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

## Current Correction

Recent work added several independent per-domain oracles. That improved the
evidence quality versus graph-native regression fixtures, but it is not the
final South Star shape. The next planning pass should stop treating
"independent oracle per domain" as the roadmap and instead consolidate toward
the one-truth reference model described above.

Near-term tasks should therefore be reframed:

- extract duplicated traversal/rendering concepts from per-domain oracles into
  shared fact/event/constraint records;
- make current feature-local oracles call or compare against that shared model;
- ensure new feature work adds constraints to the common model rather than a new
  standalone oracle;
- keep fixture cases as witness coverage only.
