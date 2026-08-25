# South Star 2: clean Rust incremental constraint walker

## Normative status

This note is the governing architectural decision. It incorporates the
successor-bearing choice revision recorded in note 115 and the liveness and
failure-classification clarification recorded in note 116. Those later notes
retain the rationale and implementation history; the contract below is
normative.

## Decision

South Star 2 is a new Rust implementation. It is not a refactor, extraction,
or continuation of the South Star 1 internal architecture.

South Star 1 has only three roles:

1. a source of difficult SMILES examples;
2. an external oracle and corpus for differential tests;
3. a record of mistakes to avoid, especially whole-support construction,
   proof machinery, and certification entering the transition path.

South Star 2 does not inherit South Star 1's modules, artifacts, envelopes,
evidence records, residual-store representation, or qualification machinery.
Existing code may inspire tests or semantics; it does not constrain the new
architecture.

## Purpose

South Star 2 is an online SMILES walker driven by a small incremental finite-
domain constraint engine.

Its fundamental operation is:

```text
current compact writer state
-> derive one ephemeral batch of source-local structural candidates
-> tentatively apply each candidate's local traversal and constraint deltas
-> propagate the affected constraint neighborhood
-> discard candidate contradiction and abort on backend failure
-> publish only {text, already-valid compact successor} choices
```

Producing the next choices must not require enumerating remaining strings,
compiling the reachable state graph, counting completions, generating proofs,
or materializing support artifacts.

## Implementation sequence is not a capability model

Development will proceed through small internal slices, but temporary
incompleteness must not become semantic taxonomy.

Do not introduce a public type, error, registry, surface, admission rule, or
qualification concept merely to describe work that has not landed yet. In
particular, avoid durable concepts such as:

```text
unsupported cycle
unsupported component count
unsupported surface
capability admission
envelope-qualified molecule
```

when they only mean that the current private implementation is incomplete.
Incomplete kernels remain private until their contract is coherent.

The prepared representation accepts ordinary molecular graphs, including:

```text
an empty graph
multiple disconnected components
cyclic components
```

Tests may arrive first for an empty graph, an atom, a path, a branch, a cycle,
and then multiple components. That is test order, not a hierarchy of supported
molecule classes.

Internally, the semantic runtime distinguishes:

```text
invalid input
valid state
candidate contradiction
spelling or dialect failure
constraint-backend failure
private implementation incompleteness
internal defect
```

Candidate contradiction is filtered during choice generation; it is not a
caller-visible choice result. An ordinary empty choice list means acceptance.
A nonaccepted state with no successful candidate is an explicit invariant or
spelling failure, not silent terminality.

Private implementation incompleteness is a temporary internal outcome for a
locally reached semantic relation that has not yet landed. It is distinct from
contradiction and invariant failure, but it does not define a public capability,
admission class, or molecule taxonomy.

## Two cooperating kernels

South Star 2 consists of:

```text
specialized graph walker
+
small incremental constraint solver
```

The graph walker owns procedural SMILES structure:

- active atom and component progression;
- branch returns and pending entries;
- visited atoms and written bonds;
- open and paired ring endpoints;
- ring-label allocation and reuse.

The constraint solver owns unresolved finite-domain relationships:

- tetrahedral token and local-order parity;
- directional carrier signs;
- shared directional sites;
- ring-endpoint compatibility;
- other small cross-event relations.

Ordinary graph bookkeeping is not converted into CSP variables merely for
uniformity.

## Rust-first boundary

The primary implementation is a pure Rust crate with no RDKit or Python
runtime dependency.

```text
RDKit or another source
-> adapter
-> Rust-owned PreparedMolecule
-> Rust walker
```

The kernel must be testable directly in Rust. PyO3 bindings come only after the
transition kernel is correct and locally efficient.

## Solver-neutral semantic model

"Swappable solver" means that South Star owns the meanings of variables,
domains, factors, and deltas independently of a backend.

```text
VariableId
Domain
FactorId
FactorDefinition
EdgeRolePartition
Consistency<S>
```

A backend executes these typed definitions. The API does not expose arbitrary
Z3, SAT, or other backend expressions.

The first native backend should use:

```text
bitset domains
typed low-arity factors
variable-to-factor adjacency
affected-factor work queue
factor-specific propagators
cheap branch cloning
```

A simple brute-force implementation may serve as a tiny-CSP oracle in tests.
External SAT or SMT adapters may be added later, but they must not shape the
kernel API.

## Prepared definitions and live state

Preparation owns immutable molecule-local data:

```rust
pub struct PreparedMolecule {
    graph: PreparedGraph,
    constraints: ConstraintModel,
    // Later: immutable traversal and event-effect indexes.
}
```

The live state contains only evolving writer and solver facts. The current
non-stereo composition is conceptually:

```rust
pub struct WriterState<S> {
    prepared: PreparedMolecule,
    traversal: TraversalState,
    constraints: S,
}

pub struct NonStereoWriterState<S> {
    surface: PreparedNonStereo,
    structural: WriterState<S>,
    labels: RingLabels,
    pending: Option<PendingEmission>,
}
```

The live solver is carried forward directly. It is not serialized to a semantic
snapshot and reconstructed after every choice.

Live graph and solver stores use sparse persistent forks where transition
measurements justify them. Candidate attempts are ephemeral transactions over
the source state: rejected forks are dropped, while successful forks become
published successors.

## Semantic choices

A structural frontier entry is a private candidate. It may contradict when its
combined writer and CSP effects are attempted, and it is never published by
itself.

A visible choice is an emitted token paired with the successor that has already
survived the complete immediate transition:

```rust
pub struct Choice<S> {
    pub text: TokenText,
    pub successor: S,
}
```

Two choices may emit the same text and lead to different successor states. The
kernel preserves both. Text grouping is a separate convenience view.

Generating choices:

1. derives the source candidate frontier exactly once;
2. forks the compact live state for one candidate;
3. applies the candidate's structural and spelling commitments;
4. restricts all immediately decided variables in one batch;
5. propagates and, where required, exactly solves the affected factor
   component;
6. validates the combined immediate successor once;
7. discards semantic contradiction, records spelling rejection, or aborts the
   whole choice batch on backend failure;
8. publishes only the successful `{text, successor}` values.

Consuming a returned choice merely selects its contained successor. It does not
rerun satisfiability and cannot reveal ordinary semantic contradiction. Equal
text must not merge distinct successors.

Immediate successor validity is not recursive suffix proof. If a published
successor later has no semantic continuation, choice generation reports an
invariant failure. The eventual stronger liveness property must come from
compact walker or CSP feasibility facts, never recursive suffix probing inside
ordinary choice generation.

## Constraint lifecycle and completeness

Prepared variables and factors are static by default. Writer events narrow
their domains monotonically. Factor activation or retirement is an optional
mechanism to introduce only when a concrete semantic relation cannot be
represented correctly and locally by static prepared factors and restrictions.
It is not part of ordinary choice application, and retirement must never be
used merely as an optimization that widens represented assignments.

Queue-based propagation is the normal path. A successor is published only after
the candidate attempt establishes a satisfiable current CSP. Arc consistency
alone is not complete for every cyclic factor graph, so the native solver:

1. propagate to a fixed point;
2. identify the affected factor component;
3. search that component when propagation cannot establish satisfiability;
4. project surviving values back into domains.

This is local CSP solving, not enumeration of SMILES suffixes.

Any future factor type must define whether it belongs to the exact semantic
search core or provides its own exact extension projector. A backend remains
private until it satisfies the solver-neutral exactness contract for every
factor type it accepts. Backend incompleteness is not a public molecule
capability or error.

## Terminality and state identity

Terminality is read directly from the compact state. It requires:

- every graph component represented;
- no pending branch or child entry;
- no unresolved ring endpoint;
- every structural obligation discharged;
- every represented bond decision resolved as required by its completed event;
- the current prepared constraint model remains consistent.

The transition kernel does not require a canonical serialized state identity.
If a future memoized client needs one, its semantic key must contain the
relevant compact facts, such as:

```text
structural traversal state
ring labels and pending visible syntax
current variable domains
any concretely introduced factor-lifecycle state
```

It excludes queues, trail history, watch order, learned clauses, allocation
identity, and other backend-specific details.

## No proof system in the hot path

The transition kernel does not construct:

- cryptographic digests;
- JSON terms;
- obligation manifests;
- branch or terminal certificates;
- count certificates;
- continuation assets;
- whole-support artifacts.

An optional typed trace may record events, domain reductions, any concretely
introduced factor lifecycle, and contradictions. Tracing is diagnostic and
does not define semantics.

## Enumeration and counting are clients

Support enumeration recursively follows semantic choices and emits accumulated
text at terminal states. Exact counting is a separate memoized client over
canonical states.

Neither operation is called by ordinary next-choice generation. Continuation
compilation, if useful later, is an optional cache built from the walker rather
than the walker architecture.

## Straight-line implementation sequence

### 1. Foundations

Establish the Rust-owned graph, solver-neutral constraint definitions, and a
private native backend. Keep provisional internals private.

### 2. General non-stereo traversal

Implement one structural transition system for ordinary molecular graphs,
including:

```text
empty graph
component progression
atoms and bonds
branches
ring endpoints and labels
terminality
```

Simple fixtures land first, but no separate tree walker, cycle surface, or
connected-only public API is created.

### 3. CSP-backed writer semantics

Add factor semantics in a straight line:

1. ring-endpoint text compatibility;
2. tetrahedral token/parity;
3. directional carriers and shared directional sites;
4. directional ring endpoints.

Each feature extends the same transition architecture.

### 4. Differential validation

Compare South Star 2 externally with original Grimace and South Star 1 for:

```text
semantic choices
text-grouped choices
terminal strings
stereochemical results
```

South Star 2 does not call either prior implementation internally.

### 5. Optional clients and bindings

Only after local transitions are correct and measured, add lazy enumeration,
memoized counting, portable snapshots, and PyO3 bindings.

## Governing rule

Every design decision is tested against this question:

> Can the walker produce the next legal semantic choices by inspecting and
> propagating only the current structural state and the small affected
> constraint neighborhood?

If the answer requires whole-support construction, global automaton
compilation, proof generation, count-DAG construction, or artifact
materialization, that work does not belong in the South Star 2 kernel.
