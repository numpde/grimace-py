Constrain development with an **executable semantic contract**, not merely a roadmap.

## 1. One operation may decide legality

Every writer step must have this form:

[
R'=\operatorname{normalize}
\bigl(
\operatorname{project}_{\text{live}}
(
\operatorname{propagate}
(
\operatorname{restrict}(R,C_a)
)
)
\bigr),
]

where (R) is the current factorized residual and (C_a) is the constraint imposed by one selected writer action.

The mandatory invariant is:

[
\llbracket R' \rrbracket
========================

\left{
x_{\mathrm{future}}
\mid
\exists x_{\mathrm{closed}}:
(x_{\mathrm{future}},x_{\mathrm{closed}})
\in \llbracket R\rrbracket
\land C_a
\right}.
]

An action is legal exactly when this residual is nonempty. No topology classifier, stereo post-filter, counting routine, or snapshot routine may make an independent legality decision.

## 2. Features must be added as relations, not molecule classes

Every new feature must specify four things:

1. **Variables** representing unresolved decisions.
2. **Small factors** expressing compatibility between those variables.
3. **Writer actions** that restrict those factors.
4. **Discharge conditions** saying when variables can be projected away.

For example:

* tetrahedral stereo: neighbour-order and parity/marker variables;
* double-bond stereo: compatibility between the two carrier variables;
* ring closure: compatibility between the opening endpoint, closing endpoint, bond text, and any delayed stereo carrier.

A proposal such as “add fused-ring support” is therefore too coarse. The acceptable proposal is closer to:

> Add a residual factor relating two simultaneous closure lifecycles, define which actions restrict it, and show when each endpoint variable is discharged.

This prevents topology-specific implementations from gradually replacing the common engine.

## 3. Walk one branch; preserve unresolved choices symbolically

The operational rule should be:

```text
one selected transition support
→ one exact residual update
→ one exact successor state
```

Selecting a branch does **not** require choosing every future ring, traversal, or stereo decision. Those remain existentially represented in their factors.

Same-text aggregation may exist for frontier presentation, counting, or audit summaries, but it must remain outside the semantic kernel. It must not determine the successor or collapse branch-local residual state.

## 4. Put a hard complexity discipline into the API

Logical relations do not automatically make the general problem polynomial. The practical guarantee is narrower:

> Never pay for combinations of decisions unless those decisions are currently coupled by a shared live variable.

Production code should therefore have no operation resembling:

```text
enumerate_assignments()
all_stereo_configurations()
all_traversals()
join_all_factors()
```

The permitted operations should be limited to:

```text
restrict a factor
semijoin factors sharing a variable
propagate through the affected connected component
project a discharged variable
merge equivalent weighted rows
test emptiness
```

Each action should touch only the factor-graph component containing variables constrained by that action. Independent stereo centres must remain separate and must not produce a Cartesian product.

The engine should record explicit complexity measures:

* largest factor cardinality;
* largest factor scope;
* size of the affected connected component;
* number of live interface variables;
* induced width encountered during propagation;
* relational work performed per transition.

There is no honest universal “non-combinatorial” guarantee for arbitrarily coupled chemistry. Instead, public support should be limited to a declared width/cardinality envelope and fail closed when execution would leave it. The boundary is then mathematical and operational, rather than a list of molecule topologies.

## 5. One engine must serve every observable

These must all consume the same residual transition function:

```text
frontier legality
advance
EOS
count
stream
snapshot/resume
diagnostics
capability evidence
```

A change is architecturally invalid if it requires:

* a second stereo traversal;
* generation followed by filtering;
* a separate counting recurrence;
* topology-specific scheduling;
* recomputation from the whole molecule during replay;
* capability inference that bypasses the executed transition.

Capabilities should be emitted as evidence that a particular residual rule or factor lifecycle was exercised. They describe execution; they do not replace residual semantics.

## 6. Make every feature pass the same acceptance gate

A development slice should not merge unless it demonstrates:

**Semantic exactness**

A small exhaustive oracle confirms that the factorized residual denotes exactly the compatible completions.

**Factorization**

Adding (n) independent stereo obligations causes approximately additive stored-state growth, not (2^n) materialized assignments.

**Compatibility**

Deliberately coupled examples propagate constraints across shared carriers and reject incompatible combinations immediately.

**Lifecycle correctness**

Every introduced variable has a precise creation point, live period, and discharge point. EOS is impossible while any required factor remains live.

**Single-engine agreement**

Frontier replay, count, stream, and snapshot resumption all produce results consistent with the same transition sequence.

**Complexity evidence**

Tests assert bounds on factor rows, active-component size, or propagation work—not merely runtime on a few examples.

**No semantic special case**

The diff adds a variable, factor, compatibility rule, or generic reduction operation—not a branch keyed to “fused”, “bridged”, or another whole-molecule category.

The current capability-audit slice should therefore remain instrumentation around branch-local live transitions. It should not become another topology semantics layer. That follows the brief’s existing requirements that writer execution stay live, avoid pre-enumeration, preserve one weighted machinery across observables, and move authority away from topology-centric gating. 

The unit of progress becomes:

> **The residual engine can maintain one additional local compatibility relation under online writing, within a measured complexity envelope.**


