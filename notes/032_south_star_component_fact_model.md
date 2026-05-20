# South Star Component Fact Model

Branch: `south-star`

Date: 2026-05-20

Task: `South Star 13: Define component fact model`

## Purpose

This note fixes the next South Star architecture target before implementation.
The central abstraction is an independent semantic stereo component, not a
global table of marker rows.

`MolToSmilesEnumS` should eventually enumerate semantic SMILES spellings by
querying a component-factorized support state:

- decompose the molecule into independent semantic stereo components;
- keep component coupling explicit and named;
- let an annotation policy decide which semantically valid markers to emit;
- keep RDKit writer parity as a comparison layer, not a semantic authority.

This is a design target for the South Star path only. The current public
`MolToSmilesEnum` path remains the RDKit writer-parity runtime.

## Core Principle

A molecule decomposes into independent stereo components, except where a named
coupling fact proves otherwise.

For an online token query:

- identify the component or components affected by the candidate token;
- update only those components;
- accept the token only if every affected component still has at least one
  surviving semantic assignment;
- leave unaffected components untouched;
- expose enough counters to detect accidental global-state growth.

This gives the desired complexity picture. Independent components multiply at
the final support-set level, but online support queries should not repeatedly
scan or filter unrelated stereo state.

## Semantic Stereo Component

A `SemanticStereoComponent` should own the local semantic problem for one
connected stereo dependency.

Required fields:

- `component_id`: deterministic local id, stable across one prepared molecule.
- `source_features`: the stereo features owned by the component.
- `eligible_carriers`: single bonds whose directional markers can express
  component stereo.
- `marker_equations`: local relationships between carriers, marker directions,
  and intended stereo assignments.
- `local_assignments`: compact representation of possible local semantic
  assignments.
- `unsupported_features`: explicit reasons this component cannot yet be handled.

The component owns semantics only. It should not contain RDKit serializer
preferences such as unwanted marker pruning, local/traversal writer layers,
or no-marker events unless a later South Star policy explicitly defines those
as semantic facts.

## Source Features

The first supported source feature class should be narrow:

- parsed RDKit molecule;
- explicit double-bond stereo with eligible directional single-bond carriers;
- acyclic alkene-style cases;
- hetero imine or oxime-style cases that use the same directional carrier
  semantics.

Each feature record should include:

- feature id;
- central bond or atom ids;
- intended stereo assignment;
- side domains;
- eligible carrier edges per side;
- evidence source, such as RDKit parsed stereo fields.

Do not start with all RDKit stereo surfaces. Atom tetrahedral stereo, aromatic
directional surfaces, ring-closure carrier bases, metal/dative quirks, query
molecules, and disconnected traversal interactions need explicit support gates
before they become package-facing behavior.

## Eligible Carriers

An eligible carrier is a graph edge that can carry a slash/backslash marker for
one or more source features.

Carrier records should include:

- normalized edge;
- orientation basis relative to the source feature;
- source feature ids affected by the carrier;
- whether the carrier is emitted directly, through a branch, or through a ring
  closure basis;
- unsupported basis reason, if the current model cannot state the basis.

The initial extractor should derive carriers from the molecule and compare them
to current South Star fixture expectations. Fixture carrier edges should become
expected outputs, not support inputs.

## Marker Equations

Marker equations are the component-local semantic constraints that decide
whether a set of directional markers preserves the intended stereo assignment.

They should be expressed independently from RDKit writer policy:

- visible `/` and `\` tokens are semantic observations;
- equivalent global flips may represent the same stereo assignment;
- omission is not a semantic observation unless an explicit annotation policy
  says so;
- RDKit marker minimization or suppression is not part of the equation.

The equation layer is where the South Star path should distinguish
"semantically valid but not RDKit-emitted" from "semantically wrong".

## Component Coupling

Components are independent by default. Coupling must be explicit.

Named coupling causes:

- shared carrier edge;
- shared atom where marker equations are not separable;
- conjugated or shared-carrier systems;
- ring-closure basis where one emitted token constrains several local facts;
- disconnected-fragment traversal facts that affect emitted token basis;
- token observations that constrain multiple components.

If none of these causes applies, a token query for one component must not
inspect or filter another component.

## Unsupported Feature Signals

Unsupported cases should fail fast at the South Star boundary. They should not
silently fall back to RDKit writer parity or produce partial semantic support.

Unsupported-feature records should include:

- category;
- affected atoms or bonds;
- reason;
- whether the limitation is extraction, component coupling, annotation policy,
  or output conformance.

Initial likely unsupported categories:

- tetrahedral atom stereo;
- aromatic directional stereo surfaces;
- ring-closure marker basis;
- dative/metal/query-bond surfaces;
- unsupported disconnected traversal interactions;
- any feature whose semantic equation cannot be stated locally.

## Annotation Policy Boundary

Semantic components answer what marker assignments are valid. Annotation policy
answers which valid assignments are emitted.

The first policy can be maximal eligible-carrier annotation:

- if a carrier is eligible and the component has a surviving marker option,
  require a visible marker;
- if several marker directions preserve semantics through different surviving
  assignments, expose the union of allowed markers;
- keep the policy swappable.

Future policies may be minimal, canonical, or writer-like. They should not
require changing component extraction or marker equations.

## Online Support State

`ComponentSupportState` should represent the current prefix state.

Required query shape:

- input: emitted edge basis plus candidate marker token or no-marker event;
- output: accepted/rejected plus reason;
- output diagnostics: affected component ids, local survivor counts, and
  unsupported-feature reasons.

The state may use rows inside one component if that is the simplest local
representation. It should not use one global row table over the whole molecule
as the primary abstraction.

## Complexity Guardrails

The South Star implementation should expose debug counters early:

- component count;
- affected component count per token query;
- local assignment count per component;
- survivor count per affected component;
- estimated final product size;
- rejected unsupported feature count.

These counters are not performance assertions. They are architecture guardrails:
they tell us whether the implementation is still component-factorized or has
silently regressed into global row filtering.

## Relation To RDKit Parity

`MolToSmilesEnum` remains RDKit writer parity.

`MolToSmilesEnumS` should remain separate and initially experimental/internal.
It may produce semantically valid strings that RDKit's writer does not emit.
Those differences should be reported by comparison diagnostics:

- `SouthStarOnly`;
- `RDKitParityOnly`;
- intersection;
- semantic correctness of both sides;
- representative divergence reasons.

Do not use RDKit exact-string fixtures as South Star pass/fail authority.

## Package-Ready Bar

`MolToSmilesEnumS` is not package-ready until all are true:

- component extraction works for a clearly documented molecule class;
- unsupported features fail fast with explicit reasons;
- the support state is component-factorized and instrumented;
- every emitted string is parseable;
- every emitted string preserves graph and intended stereo semantics;
- the OpenSMILES-style conformance criterion is documented separately from
  RDKit parser behavior;
- comparison reports against `MolToSmilesEnum` exist;
- docs explain the difference between RDKit parity and semantic enumeration.

## Immediate Next Work

1. Define unsupported feature gates before widening extraction.
2. Implement the first component extractor for acyclic directional double-bond
   stereo and hetero imine-style carriers.
3. Replace fixture-provided carrier edges as support inputs with extracted
   component facts checked against fixture expectations.
4. Build `ComponentSupportState` over those extracted components.
