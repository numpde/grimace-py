# South Star 2: clean Rust incremental constraint walker

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
-> enumerate structurally possible semantic choices
-> apply one local traversal delta
-> apply one local constraint delta
-> propagate the affected constraint neighborhood
-> return a compact successor state or contradiction
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

The semantic runtime should normally distinguish only:

```text
invalid input
valid state
contradiction
internal defect
```

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
ConstraintDelta
ConstraintSnapshot
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

The live state contains only evolving writer and solver facts:

```rust
pub struct WalkerState<S> {
    traversal: TraversalState,
    rings: RingState,
    constraints: S,
}
```

The live solver is carried forward directly. It is not serialized to a semantic
snapshot and reconstructed after every choice.

Start with measured cloning of compact vectors and bitsets. Add copy-on-write,
arenas, or persistent deltas only when profiling shows that cloning is the
bottleneck.

## Semantic choices

A primitive choice is a semantic transition, not just emitted text:

```rust
pub struct Choice {
    pub text: TokenText,
    pub action: WriterAction,
}
```

Two choices may emit the same text and lead to different successor states. The
kernel preserves both. Text grouping is a separate convenience view.

Applying a choice:

1. forks the compact live state;
2. applies the structural delta;
3. derives the typed constraint delta;
4. restricts affected domains;
5. activates or retires affected factors;
6. propagates locally;
7. returns the successor or contradiction.

## Constraint lifecycle and completeness

Factors may be activated, narrowed, and retired as writer events occur. A
factor is retired only when the semantic relationship it represents has been
resolved; retirement must not widen domains.

Queue-based propagation is the normal path. A successor presented as legal
must nevertheless have a satisfiable active CSP. Arc consistency alone is not
complete for every cyclic factor graph, so the native solver must eventually:

1. propagate to a fixed point;
2. identify the affected factor component;
3. search that component when propagation cannot establish satisfiability;
4. project surviving values back into domains.

This is local CSP solving, not enumeration of SMILES suffixes.

Until complete satisfiability handling exists, the native backend remains
private. Its incompleteness is not a public molecule capability or error.

## Terminality and state identity

Terminality is read directly from the compact state. It requires:

- every graph component represented;
- no pending branch or child entry;
- no unresolved ring endpoint;
- every structural obligation discharged;
- every required semantic factor satisfied or retired.

The canonical state key contains:

```text
structural traversal state
ring state
active variable domains
active factors
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

An optional typed trace may record events, domain reductions, factor lifecycle,
and contradictions. Tracing is diagnostic and does not define semantics.

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
