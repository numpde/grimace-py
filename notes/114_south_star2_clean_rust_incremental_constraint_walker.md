# South Star 2: clean Rust incremental constraint walker

## Decision

South Star 2 is a new Rust implementation. It is not a refactor, extraction,
or continuation of the South Star 1 internal architecture.

South Star 1 has only three roles:

1. a source of examples of difficult SMILES semantics;
2. an external corpus and oracle for differential tests;
3. a record of mistakes to avoid, especially proof machinery, whole-support
   construction, and certification entering the transition hot path.

South Star 2 must not inherit South Star 1's module boundaries, artifact
formats, envelopes, evidence records, residual-store representation, or
qualification machinery merely because those implementations already exist.
Existing pieces may serve as vague reference, inspiration, or proof of
concept. They are not architectural constraints.

## Purpose

South Star 2 is an online SMILES walker driven by a small incremental
constraint engine.

Its fundamental operation is:

```text
current compact writer state
-> enumerate structurally possible semantic choices
-> apply one local traversal delta
-> apply the corresponding local constraint delta
-> propagate only the affected constraint neighborhood
-> return a compact successor state or contradiction
```

The walker must produce the next legal semantic choices without enumerating
remaining strings, compiling the whole reachable state graph, generating
proofs, or materializing support artifacts.

## Two cooperating kernels

South Star 2 consists of:

```text
specialized graph walker
+
small incremental finite-domain constraint solver
```

The graph walker owns inherently procedural SMILES structure:

- the current atom;
- the branch stack;
- visited atoms and written bonds;
- pending child entry;
- open and paired ring endpoints;
- ring-label allocation;
- component progression.

The constraint solver owns unresolved semantic relationships:

- tetrahedral token and local-order parity;
- directional carrier signs;
- shared directional sites;
- ring-endpoint compatibility;
- other small cross-event finite-domain relationships.

Ordinary traversal bookkeeping must not be turned into CSP variables merely
for uniformity. The CSP contains only relationships that benefit from explicit
constraint propagation.

## Rust-first implementation

The primary implementation belongs directly in Rust.

The core has no RDKit dependency. RDKit is an adapter that converts a molecule
into a stable Rust-owned prepared representation:

```text
RDKit molecule
-> input adapter
-> SouthStar2PreparedMolecule
-> Rust walker
```

The Rust kernel must be testable without Python. PyO3 bindings come only after
the transition kernel is correct and locally efficient.

Rust is preferred because the relevant properties are:

- compact memory layouts;
- bitset finite domains;
- cheap state cloning or structural sharing;
- predictable allocation;
- adjacency-based propagation;
- no Python-object traffic during walking;
- straightforward later exposure through PyO3.

## Solver-neutral semantic model

"Swappable solver" means the constraint model is solver-neutral. It does not
mean that the production runtime initially depends on an external CSP package.

South Star 2 defines its own semantic model:

```text
VariableId
Domain
FactorId
FactorDefinition
ConstraintDelta
ConstraintSnapshot
```

The walker owns the meanings of variables and factors. Solver implementations
execute the same typed semantic deltas.

The API must not expose arbitrary backend expressions such as Z3 formulas.
Each backend implements the known South Star factor types.

A suitable solver contract is conceptually:

```rust
pub trait ConstraintSolver: Clone {
    type Snapshot: Clone + Eq + Hash;

    fn introduce_variable(
        &mut self,
        variable: VariableId,
        domain: Domain,
    ) -> Result<(), SolverError>;

    fn activate_factor(
        &mut self,
        factor: FactorId,
    ) -> Result<(), SolverError>;

    fn deactivate_factor(
        &mut self,
        factor: FactorId,
    ) -> Result<(), SolverError>;

    fn restrict_domain(
        &mut self,
        variable: VariableId,
        allowed: Domain,
    ) -> Result<(), Contradiction>;

    fn propagate(
        &mut self,
    ) -> Result<PropagationSummary, Contradiction>;

    fn domain(&self, variable: VariableId) -> Domain;

    fn semantic_snapshot(&self) -> Self::Snapshot;
}
```

The exact trait may evolve, but the semantic boundary is mandatory.

## Initial production solver

The first production backend should be a custom incremental finite-domain
solver:

```text
bitset domains
typed low-arity factors
variable-to-factor adjacency
affected-factor work queue
factor-specific propagators
cheap branch cloning
```

Potential variables and factor definitions receive stable integer IDs during
preparation. Factor definitions are shared by all branches. Live state stores
only current domains, active-factor status, and incremental solver metadata.

The default implementation should not enumerate Cartesian products for every
factor on every update when a specialized propagator can prune directly.

A deliberately simple brute-force backend should exist in tests as an
independent oracle for tiny CSPs. A Z3 or SAT adapter may be added later for
independent checking, explanations, or unsatisfiable cores. Neither external
backend should shape the runtime API.

## Static preparation and compact live state

Preparation owns immutable molecule-local definitions:

```rust
pub struct PreparedMolecule {
    graph: PreparedGraph,
    traversal_metadata: TraversalMetadata,
    variable_definitions: Vec<VariableDefinition>,
    factor_definitions: Vec<FactorDefinition>,
    event_effects: EventEffectIndex,
}
```

The live state is compact:

```rust
pub struct WalkerState<S> {
    traversal: TraversalState,
    rings: RingState,
    constraints: S,
}
```

The live solver must be carried forward directly. It must not be serialized to
a semantic snapshot and reconstructed after every choice.

Start with measured cloning of compact vectors and bitsets. Introduce
copy-on-write pages, arenas, or persistent deltas only when profiling proves
that ordinary cloning is the bottleneck.

A reversible trail remains useful for depth-first enumeration and counting,
but a branch-preserving decoder needs sibling states that can coexist.

## Semantic choices

A choice is a semantic transition, not merely text:

```rust
pub struct Choice {
    pub text: TokenText,
    pub action: WriterAction,
}
```

Writer actions include:

```text
emit root atom
emit child atom
emit bond
open branch
close branch
open ring endpoint
pair ring endpoint
emit component separator
finish molecule
```

Applying a choice:

1. forks or clones the compact state;
2. applies the structural traversal delta;
3. derives a typed constraint delta from the writer event;
4. restricts affected domains;
5. activates or deactivates affected factors;
6. propagates the affected neighborhood;
7. returns the successor or contradiction.

Two semantic choices may emit the same token text while producing different
states. The primitive interface preserves those branches. A separate
convenience layer may group choices by emitted text.

## Constraint lifecycle

Factors are introduced, restricted, and discharged as the writer progresses.
Examples:

- a tetrahedral token/parity factor is resolved when the relevant local order
  closes;
- a directional carrier factor is narrowed when a carrier mark is emitted;
- a directional site factor is discharged when all relevant carriers are
  resolved;
- a ring-endpoint compatibility factor exists between endpoint creation and
  pairing.

The native solver should remove discharged factors from the active adjacency
index. External SAT or SMT adapters may emulate factor activity through
activation literals or assumptions.

## Propagation completeness

Queue-based local propagation is the normal path.

A presented successor must nevertheless have a satisfiable active CSP. Local
arc consistency alone may not prove this for cyclic factor graphs. Therefore
the native solver should:

1. propagate to a fixed point;
2. identify the affected connected factor component;
3. if the component remains unresolved and cyclic, perform bounded search
   inside that component;
4. project surviving assignments back into domains.

This is local CSP solving, not enumeration of remaining SMILES suffixes.
Complexity must depend on the affected constraint component, not on the global
number of terminal strings.

## Terminality

A walker state is terminal only when:

- the molecular graph has been completely represented;
- no branch return or child entry is pending;
- no ring endpoint remains unresolved;
- all structural obligations are discharged;
- all required semantic factors are satisfied or discharged.

Terminality is read directly from the compact state. It is never inferred by
enumerating suffixes.

## State identity

South Star 2 owns a canonical semantic state key derived from:

```text
structural traversal state
ring state
active variable domains
active factors
```

The key excludes:

```text
propagation queues
trail history
watch ordering
learned clauses
memory layout
backend-specific solver state
```

Different solver implementations reaching the same semantic state must expose
the same canonical identity. This identity supports state merging, counting,
snapshots, and differential tests.

## No proof system in the hot path

The transition kernel must not construct:

- cryptographic digests;
- JSON terms;
- obligation manifests;
- branch or terminal certificates;
- facts-bound proof records;
- count certificates;
- continuation assets;
- whole-support artifacts.

An optional typed trace may record:

```text
writer event
restricted variables
activated factors
deactivated factors
domain reductions
contradiction
```

Tracing must be optional and must not define transition semantics.

## Enumeration and counting are clients

Full support enumeration is an optional traversal over the walker:

```text
walk semantic choices recursively
-> emit the accumulated string at terminal states
```

Exact counting is a separate memoized client over canonical states:

```text
count(state) = 1, if terminal
count(state) = sum(multiplicity * count(successor)), otherwise
```

Neither operation may be called by ordinary next-choice generation.

Continuation compilation, if later useful, is an optional cache built from the
walker. It is not the walker architecture.

## Straight-line implementation sequence

### Phase 1: kernel contract

Create a new Rust module or crate containing:

```text
prepared graph
traversal state
ring state
semantic choices
advance
terminality
constraint model
solver trait
native solver skeleton
```

No Python API, support enumeration, counting, proofs, or assets.

### Phase 2: structural writer

Implement exact connected non-stereo traversal through the final interfaces:

```text
atoms
bonds
branches
ring labels
root selection
terminality
```

Use an empty constraint model. This proves the walker architecture before
stereo complexity is added.

### Phase 3: CSP-backed semantics

Add factors in this order:

1. ring-endpoint text compatibility;
2. tetrahedral token/parity;
3. acyclic directional carriers;
4. shared directional sites;
5. directional ring endpoints.

Each feature adds factor definitions and event-to-delta rules. It must not add
a second transition architecture.

### Phase 4: differential validation

Compare South Star 2 externally against original Grimace and South Star 1 for:

```text
semantic next choices
text-grouped choices
terminal strings
rooted support
stereochemical output
```

South Star 2 must not call either prior implementation internally.

### Phase 5: optional clients

Only after the local transition kernel is demonstrably cheap, add:

```text
lazy support enumeration
memoized completion counting
snapshots
PyO3 bindings
```

## Governing performance rule

Every design decision is tested against this question:

> Can the walker produce the next legal semantic choices by inspecting and
> propagating only the current structural state and the small affected
> constraint neighborhood?

If the answer requires whole-support construction, global automaton
compilation, proof generation, count-DAG construction, or artifact
materialization, that work does not belong in the South Star 2 kernel.
