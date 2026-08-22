# Writer liveness and exact-component plan

## Scope

This revision closes three remaining contract gaps after authoritative
successor choices landed:

1. a nonaccepted state must not report an ordinary empty choice set;
2. a local role restriction must not revalidate every solver domain;
3. exact-search component discovery must not rebuild immutable factor-graph
   metadata for every changed-variable seed.

It does not add suffix search, support construction, disconnected rendering,
new factor types, pending-continuation caches, or persistent ring-label state.

## Liveness classification

`choices()` remains one-step and source-local. It attempts the one structural
candidate batch and classifies a zero-success batch as one of:

```text
accepted state
    -> ordinary empty choice set

at least one semantically valid candidate has no selected-dialect spelling
    -> typed spelling failure

every candidate contradicts immediate writer/CSP consistency
    -> typed internal live-state invariant failure

solver backend failure
    -> typed backend failure
```

Candidate spelling is checked only after the structural transition and its
successor validation. This prevents an unspellable but semantically
contradictory candidate from being misclassified as dialect exhaustion.

The invariant failure is deliberately not repaired by recursively asking
whether later states reach acceptance. Future feasibility must eventually be
represented by compact writer facts and factors. Until then, a discovered
nonterminal dead end is explicit evidence that the live-state model is
incomplete, not a terminal decoder result.

## Solver boundary

The complete solver-domain shape assertion remains at initial solver
admission. A successful `ConstraintSolver::restricted` transition is trusted
to preserve that admitted contract. Writer code still reads the particular
role domains required by local frontier and transition logic, but it does not
scan unrelated variables after every restriction.

Backend conformance tests retain full-domain checks. A counting test pins that
one writer role restriction does not call the solver-neutral `domain()` method
once per prepared variable.

## Immutable exact-component plan

The constraint model stays solver-neutral. `NativeSolverState::initial`
compiles one immutable, `Arc`-shared exact-search plan from the immutable
factor graph.

The plan treats binary-relation variables as the semantic core. Binary
relations connect core variables directly; a spanning-tree factor connects
all core variables in its scope and contributes all of its variables to the
projected set. This yields disjoint descriptors:

```text
pure cyclic binary component
    -> core variables + binary factor IDs

mixed semantic/structural component
    -> core variables + all variables projected by touched spanning factors

pure acyclic binary component
pure spanning-tree component
isolated variable
    -> no exact-search descriptor
```

Every projected variable maps directly to at most one descriptor. Transition
time collects descriptor IDs from changed variables, deduplicates those IDs,
and runs only those exact filters. A pure spanning-tree writer model therefore
performs no semantic-component discovery or exact semantic search after its
ordinary structural projection.

Mixed exact search subsumes separate binary-component exact search for the
same descriptor: it branches over every unresolved semantic-core variable,
propagates all binary and spanning factors, and unions projected support over
surviving assignments. Pure cyclic binary descriptors continue to use the
specialized local binary search.

## Alternatives rejected

- Recursively validating that each successor reaches a terminal string would
  reconstruct support and violate the online kernel boundary.
- Returning an unclassified empty vector would keep terminality, contradiction,
  and dialect exhaustion observationally identical.
- Keeping the full shape assertion under `debug_assertions` would still make
  ordinary development and test transitions globally linear.
- Caching dynamically discovered components in each solver state would add
  invalidation and duplicate immutable metadata across persistent forks.
- Putting native exact-search descriptors into `ConstraintModel` would leak a
  backend execution plan into the solver-neutral factor definition.

## Acceptance gates

- accepted states alone return an ordinary empty choice set;
- all-candidate contradiction returns an explicit invariant failure;
- all-candidate ring-label exhaustion returns a typed spelling failure;
- an unspellable opening still does not suppress a valid sibling closure;
- backend failure remains distinguishable;
- role restriction performs no whole-model solver-domain scan;
- exact descriptors are compiled once and shared by solver forks;
- a pure spanning-tree model runs zero mixed-component searches;
- one changed mixed component does not execute a disconnected descriptor;
- native and exhaustive mixed-domain projections remain equal.

## Deferred measurements

Pending deterministic transitions intentionally remain recomputed during
prevalidation and later token emission. Ring-label assignments intentionally
remain a small cloned `BTreeMap`. Counters and profiles should establish that
either cost matters before adding validated pending deltas or persistent label
storage.
