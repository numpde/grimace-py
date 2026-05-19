# South Star Semantic Stereo Enumeration

Branch: `stereo-constraint-model`

Date: 2026-05-19

## Purpose

The South Star is the mathematically clean stereo-enumeration target, distinct
from RDKit writer-string parity.

It asks: what should Grimace enumerate if the goal is not "match RDKit's
serializer quirks", but instead "emit every online SMILES spelling that is
semantically valid under a principled OpenSMILES-style stereo language"?

This note names that target so runtime RDKit parity work and semantic
enumeration work do not keep borrowing each other's concepts accidentally.

## Definition

South Star enumeration is exact online support for graph and stereo semantics
under a maximal directional-marker policy.

For a prepared molecule, Grimace should enumerate prefixes and complete SMILES
strings such that:

- the emitted string is syntactically valid SMILES;
- parsing the emitted string reconstructs the intended graph;
- parsing the emitted string reconstructs the intended stereo assignments for
  all supported stereo centers;
- every directional marker required to make double-bond stereo explicit at the
  chosen traversal boundary is emitted, rather than minimized away;
- slash/backslash choices satisfy one global component constraint system;
- every emitted token is supported online by the current prefix state;
- no completed-string repair, cleanup, or backfill phase is needed.

The "maximal" in maximal directional-marker policy means "mark all required
directional carrier opportunities chosen by the semantic policy", not "spray
markers on chemically irrelevant bonds". It is a deterministic semantic policy
for making stereo explicit, not RDKit's traversal-conditioned minimization.

## Non-Definition

The South Star is not:

- RDKit writer support;
- RDKit's `canonicalizeDoubleBonds` behavior;
- RDKit's unwanted/redundant directional-spec pruning;
- a canonical SMILES normal form;
- a claim that OpenSMILES requires minimal or maximal annotation;
- parser equivalence used as a substitute for RDKit parity in the public
  runtime;
- a post-processing pass over completed RDKit-style outputs.

OpenSMILES validity does not require global minimization of directional
markers. Minimization or maximal annotation is a writer policy. South Star
chooses maximal annotation because it gives a cleaner semantic constraint
problem and avoids making RDKit's pruning look chemically fundamental.

## Relation To The North Star

The current North Star for the public runtime is exact RDKit writer-string
support for RDKit's `canonical=False, doRandom=True` regime. That remains the
shipping API target.

The South Star is a parallel semantic target:

- North Star: exact RDKit writer parity.
- South Star: exact principled SMILES/stereo semantic support.

The two should share graph preparation, component decomposition, carrier
domains, token-direction algebra, and online support-state machinery. They
should not share writer-policy decisions blindly.

A clean architecture should look like this:

1. Build semantic stereo components and constraints.
2. Expose a South Star support query over those semantic constraints.
3. Layer RDKit writer policy as a named projection or additional constraint
   layer.
4. Keep public RDKit parity tests exact-string based.
5. Keep South Star tests parser/semantic based and explicitly named as such.

## Why South Star Is Useful

South Star gives a principled reference point when RDKit behavior is awkward,
underspecified, or possibly inconsistent.

It can answer:

- whether Grimace's core stereo algebra is chemically/semantically coherent;
- whether RDKit is omitting semantically valid spellings;
- whether a red RDKit-parity witness is a semantic bug or a writer-policy gap;
- whether a proposed runtime rule belongs in the semantic layer or RDKit policy
  layer;
- whether row/propagator machinery is solving the actual stereo constraint
  problem or only mirroring serializer artifacts.

This is especially important for shared directional markers, conjugated systems,
ring closures, and cases where RDKit mutates directional bond state after it has
already chosen a traversal stack.

## Constraint Shape

The semantic model should decompose the molecule into independent stereo
components. Within each component, variables describe finite choices:

- selected carrier neighbor for each stereo side;
- directional token basis for each carrier edge;
- per-component orientation or phase;
- ring/branch traversal facts that affect how emitted tokens are interpreted;
- marker obligations required by the maximal annotation policy.

Constraints should enforce:

- each supported stereo center has enough marked carrier information to recover
  its assignment;
- shared carrier edges satisfy every component that observes the same emitted
  token;
- a visible slash/backslash token has one consistent meaning in the emitted
  edge basis;
- omitted marker opportunities are legal only when the South Star policy says
  the edge is not required;
- terminal states have no pending semantic marker obligations.

Rows are one possible finite representation of this state. A native propagator,
bitset domains, or a hybrid may be better later. The representation is not the
goal; the goal is one explicit online constraint boundary.

## Maximal Marker Policy

The starting South Star policy should be intentionally simple:

- if a traversal edge is a required directional carrier for any surviving
  semantic stereo assignment, a marker must be emitted on that edge when the
  output grammar allows it;
- if several marker values are compatible with different surviving assignments,
  branch online;
- if exactly one marker value is compatible, force it;
- if an edge is not a required directional carrier under any surviving
  assignment, do not emit a stereo marker for South Star purposes;
- if an obligation is known but not yet placeable at the current grammar
  boundary, carry it explicitly and discharge it at the next valid boundary.

This policy deliberately avoids RDKit's question: "which otherwise valid
markers should be suppressed as redundant for this traversal?" The South Star
question is instead: "which markers make the semantic stereo assignment explicit
and consistent?"

## Tests

South Star tests should be separate from RDKit parity tests.

Good test classes:

- tiny isolated alkene and oxime cases where all valid marked spellings are easy
  to inspect;
- conjugated/shared-carrier witnesses where RDKit omits some semantically valid
  spellings;
- ring-closure witnesses where marker basis must be interpreted through the
  emitted edge, not only component-local endpoint order;
- parser round-trip tests that verify graph and stereo assignment;
- negative tests for inconsistent slash/backslash assignments;
- cross-checks against Z3 or another exploration oracle for small components.

These tests should not assert equality with RDKit sampled output. If RDKit
parity is relevant, add a separate paired test that states which layer is being
checked.

## Relationship To Current Branch

`stereo-constraint-model` is closer to the South Star than `main` because it
already has:

- explicit stereo components;
- carrier-assignment state;
- token-phase assignment state;
- marker-placement rows;
- marker event facts;
- support-boundary survivor queries;
- diagnostics separating semantic and RDKit writer layers.

But it is not the South Star yet.

The current marker-placement rows are RDKit-policy shaped: they model visible
marker subsets and no-marker events, because the public runtime is trying to
match RDKit writer support. South Star should reuse the component and online
fact machinery, but it should define its own semantic marker-obligation policy
instead of inheriting RDKit suppression/minimization.

## Implementation Direction

The least-regret implementation path is:

1. Keep public runtime on the RDKit parity path.
2. Add South Star as an internal semantic support mode or diagnostic query.
3. Reuse component decomposition and token-direction algebra.
4. Add a semantic marker-obligation model that requires maximal annotation.
5. Add small exact semantic fixtures independent of RDKit writer output.
6. Compare South Star support with RDKit parity support on known witnesses and
   classify differences as either semantic bugs or RDKit writer-policy
   exclusions.
7. Only after this split is stable, decide whether South Star should become a
   public API surface.

## Open Questions

- What exact OpenSMILES subset should define supported semantic parsing?
- Which non-double-bond stereo classes belong in the first South Star slice?
- Should maximal annotation mark every eligible carrier edge or exactly every
  carrier edge selected by the component assignment?
- How should aromatic directional bonds be represented in the semantic policy?
- How much parser-equivalence evidence is enough before treating an RDKit
  omission as a writer-policy exclusion rather than a Grimace bug?

## Success Criteria

South Star is meaningful when:

- it is implemented as online support, not completed-string filtering;
- every accepted string parses to the intended graph and stereo assignment;
- every rejected string has a named semantic reason;
- RDKit writer-policy exclusions are represented separately;
- small witnesses can show South Star support strictly larger than RDKit writer
  support without making the public RDKit parity layer fail;
- the code makes it obvious whether a rule is semantic or RDKit-specific.
