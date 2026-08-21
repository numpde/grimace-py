# Authoritative successor choices and hardening lanes

## Starting split

South Star 2 began this revision with two transition paths for the same visible move:

1. `ConnectedNonStereoWriterState::choices()` derives a `StructuralFrontier` and publishes a descriptor.
2. `advance()` calls `choices()` again, then invokes `WriterState` helpers which derive the frontier again before applying the transition.

That split is the immediate correctness problem. A descriptor can become stale, ordinary candidate contradiction is represented as the solver's opaque error, and source-local structural facts such as a ring closure's first endpoint are rediscovered during advancement.

## Single source of truth

The authoritative path will be:

```text
ConstraintSolver
    returns Consistency separately from backend Failure

WriterState
    derives one ephemeral StructuralCandidate batch
    applies one supplied candidate without deriving another source frontier

ConnectedNonStereoWriterState
    attempts every candidate, including spelling state
    filters semantic rejection
    aborts on backend failure
    publishes text plus an already-valid successor
```

No durable candidate batch, capability object, continuation cache, or public solver adapter is introduced. Dense state clones remained temporarily as a correctness baseline until the residual-state shape was established.

## Ownership boundaries

- `ConstraintSolver` owns finite-domain consistency outcomes and backend failures.
- `WriterState` owns structural candidate derivation and structural/CSP transition effects.
- `ConnectedNonStereoWriterState` owns visible spelling, pending-token state, ring-label resources, and successor publication.
- Preparation owns immutable topology and, in the residual-state lane, immutable initial component metadata.
- Traversal owns live residual partition and writer frames; it must not become a CSP factor or traversal planner.

## Serious alternatives rejected for this slice

- Keeping descriptor choices plus a checked `advance`: preserves duplicate semantic paths.
- Caching `StructuralFrontier` in writer state: turns an ephemeral derivation into durable state and creates invalidation work.
- Searching every variable touched by a spanning-tree factor: makes semantic coupling exponential in all graph bonds.
- Installing general dynamic connectivity: exceeds the required atom-deletion residual update.
- Optimizing current dense arrays before defining residual partition shape: risks optimizing stores that the locality revision replaces.

## Minimal-regret sequence

1. Separate `Consistency` from solver `Failure`.
2. Derive one source-local structural candidate batch carrying closure facts.
3. Attempt candidates internally and publish successor-bearing visible choices.
4. Delete semantic `advance` revalidation and pin rejection/failure/sibling behavior.
5. Add exact semantic-core search through the spanning-tree projector.
6. Retain prepared component metadata and add a live residual partition, differentially checked against full recomputation.
7. Introduce paged copy-on-write stores, persistent frames, and candidate overlays only after the live-state shape is stable.

## Potential regrets to monitor

- Successor-bearing choices amplify dense clone cost until persistent stores land.
- A generic kernel failure wrapper may be premature while the only non-asserting failure is the solver backend; keep the visible error generic over `S::Failure` initially.
- Candidate rejection reasons should remain private until diagnostics have a concrete consumer.
- Residual component identifiers must be stable only within one live state; treating them as public or canonical identity would overconstrain later storage.

## Acceptance gates

- One source frontier derivation per `choices()` call.
- Returned choices contain valid successors and require no source state to apply.
- Candidate contradiction is filtered; backend failure aborts the batch.
- Equal text does not merge distinct successors.
- Mixed equality-plus-triangle contradiction matches exhaustive projection.
- Incremental residual attachments match test-only full recomputation.
- Locality counters show no unconditional whole-molecule scan and no full successor materialization for rejected candidates after persistent storage lands.

## Implemented locality shape

The persistent store is a private radix tree with 64-value leaves and at most 32 children per index node. A state clone shares one root; a write path-copies only the index path and the touched leaf. Domain values, bond progress, visited words, live atom-to-component IDs, and residual-component slots use this store.

Residual components keep immutable ordered member lists plus small live metadata. The paged atom-to-component store remains the live membership authority. A no-split deletion therefore changes only paged IDs and component metadata; an actual split walks and rebuilds only the affected component.

Writer frames and their suspended stack nodes are `Arc`-shared; restoring a parent therefore shares the complete frame as well as its ancestors until a later local mutation. Candidate attempts use ordinary state clones as ephemeral transactions: persistent roots are initially shared, candidate-local writes path-copy their touched pages, contradictions drop the tentative fork, and only successful forks become visible successors.
