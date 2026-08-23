# Visible component composition

## Scope

This slice removes the connected-only non-stereo spelling adapter without
adding new chemistry or CSP relations. The graph walker remains responsible
for component lifecycle; the visible layer owns only `.` spelling and its
forced pending root atom.

The structural completion relation is typed before disconnected spelling is
admitted:

```text
CloseBranch
    -> restore a suspended parent
    -> visible ")"

FinishComponent
    -> restore no parent
    -> silent normalization
```

An empty graph is already accepted. The first component root emits its atom
text directly. After a component finishes, each remaining root candidate
publishes `.` and a successor already committed to that root; the selected
atom text is the successor's sole pending visible operation. Equal dot text
therefore preserves distinct root-order branches.

## Ownership boundaries

- `WriterState` derives and applies typed graph completion candidates.
- `PreparedNonStereo` validates only atom-text and bond-token bindings.
- `NonStereoWriterState` silently consumes `FinishComponent`, spells
  `CloseBranch`, and owns `ComponentRootAtom` pending syntax.
- Ring labels must be clean at every component boundary and may be reused by
  the next component.
- Whole-support component products remain test-only evidence and never enter
  `choices()`.

## Alternatives rejected

- Inferring branch return from whole-graph completeness fails when a component
  ends before the graph does.
- Publishing a zero-length component-completion choice leaks forced
  normalization into the decoder alphabet.
- Emitting `.` before selecting a root would add a spelling-only state and
  collapse semantically distinct equal-dot choices.
- Recursively searching suffixes to validate component liveness would violate
  the one-step online kernel boundary.

## Acceptance gates

- the empty graph has support `{ "" }`;
- isolated distinct atoms enumerate every component order;
- identical atom text preserves semantic root-order paths despite one terminal
  text;
- branch closure emits `)` while top-level completion is silent;
- equal `.` choices retain different committed roots and preserve their source;
- two cyclic components close all labels and reuse label 1;
- a test-only Cartesian component oracle matches live disconnected tree text;
- bounded disconnected exploration reports no writer invariant failure.

## Deferred work

Chemical atom rendering, bracket grammar, aromatic and bond-elision policy,
ring-endpoint factors, parser equivalence, public APIs, persistent label
storage, and pending-continuation caching remain outside this slice.
