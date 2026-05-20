# South Star Ring Traversal Extension

Branch: `south-star`

Date: 2026-05-20

Task: `South Star 37: Plan ring traversal extension`

## Purpose

Ring support should extend the traversal language. It should not be implemented
as a finished-string mutation, and it should not borrow RDKit writer placement
rules unless a later writer-policy layer explicitly asks for them.

The South Star implementation was initially limited to one connected acyclic
component. `south_star_support_gate_report()` named rings as unsupported with
`ring_molecule`, and directional ring carriers got the more specific
`ring_stereo` category. That fail-fast boundary was correct until ring closure
syntax was represented as first-class traversal data.

## Implementation Checkpoint

The private `south-star` branch now has a first graph-native ring slice:

- connected saturated monocycles with acyclic branches are supported;
- traversal chooses one graph ring edge as the closure edge and traverses the
  remaining spanning tree;
- closure syntax is emitted through `ring_open` and `ring_close` events carrying
  graph edge identity, closure id, label, role, and traversal parent context;
- simple cyclohexane and methylcyclohexane support is pinned in
  `tests/fixtures/south_star_expanded_support/expanded_domain_v1.json`;
- fused rings, unsaturated rings, and ring stereo remain explicitly gated.

This is still only the non-stereo ring-language slice. The marker-slot equation
work for ring-closure carriers has not been implemented.

## Required Concept Split

Graph semantics:
: The molecule graph has cycle edges. A traversal chooses a spanning tree plus
  non-tree closure edges. This layer does not know closure digits or string
  placement details.

Traversal language:
: A traversal emits atom, tree-bond, branch, and ring-closure events. Closure
  events are not patches to an already-rendered string; they are syntax-bearing
  events with graph edge identity and local syntax position.

Ring numbering policy:
: A policy assigns SMILES closure labels to closure events. The initial South
  Star policy can be deterministic and simple; it does not need to match RDKit's
  writer ordering. The only hard requirement is internally consistent paired
  labels for each closure edge.

Directional marker slots:
: A slash/backslash marker on a ring-closure carrier must be represented as a
  marker slot on the closure event itself or on an adjacent tree-bond event.
  Marker-slot equations should reference those syntax positions directly.

Semantic constraints:
: Directional double-bond constraints should still be solved as component-local
  marker equations. Adding rings must not introduce a global post-render filter
  over completed strings.

## Event Shape

The current event vocabulary is:

- `atom`
- `bond`
- `branch_open`
- `branch_close`

Ring support should add explicit event kinds instead of overloading `bond`:

- `ring_open`: first syntax occurrence for a closure edge;
- `ring_close`: second syntax occurrence for the same closure edge.

Both events should carry:

- normalized graph `edge`;
- `begin_atom_idx`;
- `end_atom_idx`;
- `begin_parent_idx`;
- closure label assigned by numbering policy;
- syntax position such as `ring_open` or `ring_close`;
- optional `marker_slot`.

The renderer should render a closure event from its event data:

- optional solved marker assignment;
- bond text if needed for explicit non-single closure bonds;
- ring label, including `%NN` or parenthesized extension if the policy later
  permits labels beyond one digit.

The renderer should not search the string for a place to insert the closure.

## Traversal Algorithm Sketch

For a chosen root and child-order policy:

1. Build a spanning-tree traversal over atoms.
2. Classify each graph bond as a tree edge or closure edge.
3. For each closure edge, choose the two endpoint syntax events where the
   closure label appears.
4. Allocate closure labels by policy.
5. Emit atom, tree-bond, branch, and ring-closure events in one pass.
6. Build marker slots from event-local syntax positions.
7. Build marker equations from graph component facts and marker slots.
8. Solve assignments before rendering.
9. Render only from traversal events plus solved assignments.

This keeps the complexity split intact: traversal choices generate event
skeletons, semantic stereo components generate local assignments, and the solver
connects marker slots to component equations.

## Numbering Policy

The first South Star policy should be deliberately boring:

- assign labels in first-encounter order during traversal;
- reuse a label exactly twice, once at each endpoint event;
- fail fast if the current renderer cannot represent the needed label range;
- expose the chosen label in the event so tests can inspect it.

Do not start by mimicking RDKit's closure-label reuse or traversal ordering.
That belongs in a named writer-policy layer if needed later.

## Marker Equation Implications

Ring closure marker slots need the same invariant as tree-bond marker slots:

- every required marker slot appears in the traversal event stream;
- every marker slot is assigned exactly one marker by the solver;
- the solver equations reference slot ids, not rendered strings;
- impossible slot assignments fail before rendering.

The existing `SouthStarMarkerSlot` shape is close, but `syntax_position` must
accept ring positions and the slot id must include closure-event identity. A
ring closure can put the visible directional marker on a syntax occurrence that
is not a parent-child tree edge, so equation construction must not assume every
marker slot came from a tree bond.

## First Fixtures

Start with non-stereo rings before stereo rings:

1. `C1CCCCC1`: graph closure and label pairing only.
2. `C1CCC(C)CC1`: branch plus ring closure.
3. `C1=CCCCC1`: explicit double closure/bond text handling, no directional
   marker semantics yet.
4. `C1/C=C\\CCCC1`: directional ring carrier semantics.

The first three should prove traversal/numbering/rendering without forcing the
stereo equation layer to change. The fourth should be the first marker-slot
equation fixture for closure syntax.

## Guardrails

- Keep `ring_molecule` fail-fast until at least non-stereo ring traversal is
  represented by events.
- Keep `ring_stereo` fail-fast until closure marker slots feed the same solver
  path as tree-bond marker slots.
- Do not add a parser-backed accept/reject filter to make ring outputs look
  correct after rendering.
- Do not add RDKit closure ordering as an implicit default.

## Minimal Implementation Path

1. Introduce closure-event dataclasses or extend `SouthStarTraversalEvent` with
   explicit ring fields while keeping current acyclic behavior unchanged.
2. Add a traversal test that constructs a tiny synthetic closure event stream
   and verifies rendering comes from event data.
3. Add a non-stereo cyclohexane fixture once traversal can emit paired closure
   events.
4. Add marker-slot equations for a ring directional carrier only after the
   non-stereo closure language is stable.
5. Repeat the complexity checkpoint with ring fixture rows separated from
   acyclic rows.
