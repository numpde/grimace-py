# South Star Semantic Stereo Investigation

Branch: `south-star`

Date: 2026-05-19

## Purpose

South Star names a semantic investigation target, distinct from RDKit
writer-string parity.

The question is:

> Which SMILES strings are valid semantic spellings of the intended graph and
> stereo assignment, independent of whether RDKit's writer happens to emit
> them?

This note should not be read as a final runtime architecture. Its job is to keep
the investigation from importing North Star assumptions accidentally.

## Starting Definition

For now, South Star means parser-backed semantic correctness:

- the emitted string is syntactically valid SMILES;
- parsing the emitted string reconstructs the intended molecular graph;
- parsing the emitted string reconstructs the intended supported stereo
  assignments;
- slash/backslash markers have a globally consistent interpretation;
- RDKit writer membership is comparison evidence, not the definition of
  correctness.

This leaves several policy choices open. In particular, it does not yet decide:

- whether the eventual semantic policy is maximal, minimal, or something else;
- whether every eligible carrier edge should be marked;
- whether only assignment-selected carrier edges should be marked;
- whether omission can ever be a semantic observation;
- which representation should implement the constraint system;
- whether South Star becomes a public API or remains an internal diagnostic.

## Non-Definition

South Star is not:

- RDKit writer support;
- RDKit's `canonicalizeDoubleBonds` behavior;
- RDKit's unwanted/redundant directional-spec pruning;
- a canonical SMILES normal form;
- a claim that OpenSMILES requires minimal or maximal annotation;
- parser equivalence used as a substitute for RDKit parity in the public
  runtime;
- a post-processing pass over completed RDKit-style outputs.

OpenSMILES validity does not appear to require global minimization of
directional markers. Minimization, maximal annotation, and traversal-conditioned
suppression are writer policies. South Star should make those policies explicit
before choosing one.

## Relation To The North Star

The current North Star for the public runtime is exact RDKit writer-string
support for RDKit's `canonical=False, doRandom=True` regime. That remains the
shipping API target unless explicitly changed.

The South Star investigation should stay separate:

- North Star: exact RDKit writer parity.
- South Star: semantic validity of SMILES/stereo spellings.

Shared infrastructure may be useful, but sharing must be deliberate. Graph
preparation, component discovery, carrier domains, and token algebra may be
semantic. RDKit marker suppression, no-marker events, committed-token parity
filters, and serializer fixtures are writer-policy or comparison evidence unless
proven otherwise.

## Why South Star Is Useful

South Star gives a reference point when RDKit behavior is awkward,
underspecified, or possibly inconsistent.

It can answer:

- whether Grimace's stereo algebra is semantically coherent;
- whether RDKit is omitting semantically valid spellings;
- whether a red RDKit-parity witness is a semantic bug or a writer-policy gap;
- whether a proposed rule belongs in semantic support or RDKit writer policy;
- whether row/propagator machinery is solving the stereo constraint problem or
  only mirroring serializer artifacts.

This is especially important for shared directional markers, conjugated systems,
ring closures, and cases where RDKit mutates directional bond state after it has
already chosen a traversal stack.

## Starting Constraint Vocabulary

The investigation can begin with neutral concepts:

- intended molecular graph;
- supported stereo centers or stereo bonds;
- directional carrier candidates;
- selected or observed carrier edges;
- emitted slash/backslash tokens;
- token basis relative to an emitted edge;
- component-local assignments;
- parser-observed graph/stereo result;
- writer-policy comparison outcome.

These names are intentionally not a full model. They are a starting vocabulary
for separating semantic facts from RDKit writer facts.

## Candidate Policy Questions

The central policy questions remain open:

- What does "sufficiently explicit" stereo annotation mean?
- Is maximal annotation desirable, and if so, maximal over which carrier set?
- Can omission of a marker carry semantic information, or is it only writer
  policy?
- How should shared carriers be represented when one marker constrains several
  stereo centers?
- How should ring-closure marker basis be stated independent of RDKit placement?
- Which aromatic and non-double-bond stereo forms are in the first scope?
- What evidence is enough to call a string semantically valid when RDKit does
  not emit it?

The first implementation work should make these questions observable rather
than answer all of them.

## Tests

South Star tests should be separate from RDKit parity tests.

The first tests should be small semantic witnesses:

- candidate strings that parse to the intended graph and stereo assignment;
- negative strings that parse but invert or lose the intended stereo assignment;
- cases where RDKit emits the string, recorded only as comparison metadata;
- cases where RDKit does not emit the string, if the parser-backed semantic
  result is still correct.

Good early cases:

- isolated alkene or oxime examples where all marked spellings are inspectable;
- a small conjugated/shared-carrier witness;
- a ring-closure witness where marker basis depends on the emitted edge;
- one known RDKit-discrepant witness, clearly marked as comparison evidence.

These tests should not assert equality with RDKit sampled output. If RDKit
parity is relevant, add a separate paired test that states which layer is being
checked.

## Relationship To Current Branch

The `south-star` branch starts from `stereo-constraint-model`, which already
contains useful machinery and North Star baggage.

Potentially useful machinery:

- explicit stereo components;
- carrier-assignment state;
- token-phase assignment state;
- support-boundary survivor queries;
- diagnostics that can compare semantic and RDKit writer layers.

North Star-shaped machinery:

- marker-placement rows modeled around visible RDKit marker subsets;
- `NoMarker` writer events;
- RDKit local/traversal writer layers;
- token-flip adjustments named after RDKit writer behavior;
- committed-token parity filters;
- pinned exact-string RDKit fixtures as runtime authority.

Do not delete these surfaces just because they are North Star-shaped. First
classify them as semantic, comparison-only, reusable scaffolding, or obsolete.

## First-Slice Principle

The first slice should create a clean observation point:

1. Add a South Star semantic test runner that does not depend on RDKit
   exact-string fixtures.
2. Add tiny parser-backed semantic witnesses.
3. Add helper assertions for graph/stereo equivalence.
4. Record RDKit writer membership only as comparison metadata.
5. Add guards or conventions that prevent South Star tests from depending on
   RDKit writer-policy layers.

This first slice should not:

- change public `MolToSmilesEnum` behavior;
- delete RDKit parity fixtures;
- decide the final annotation policy;
- make row/state machinery the assumed representation;
- treat RDKit omissions as automatically wrong.

## Success Criteria

The investigation is on track when:

- semantic tests can pass or fail independently of RDKit exact-string parity;
- every accepted witness has parser-backed graph/stereo evidence;
- every rejected witness has a named semantic reason or open question;
- RDKit writer-policy exclusions are represented separately;
- it is obvious whether a rule is semantic, writer-policy, or still
  unclassified.
