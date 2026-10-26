# Exact relational computation as the foundation of the online writer

2026-10-26

## Decision and scope

Build the computational core around a maintained exact relation, from which
supported immediate choices are enumerated. Do not build a walker around an
unresolved completion predicate and later supply a planner wherever local
constraints are insufficient.

This records the architectural direction discussed after notes 126–128. It
proposes a concrete algebra and an initial implementation boundary; it does
not claim that the required molecular encoding or the strong complexity
target of note 127 has been solved. It changes no existing runtime behavior
and does not redefine the language currently supported on `main`.

The failure mode to prevent is an incremental physical-factor layer whose
hard cases still depend on future visit schedules, branch-end plans, or
recursive hypothetical writer execution. A factorization alone does not
prevent that failure: symbolic computation can conceal the same work.

## Behavioral and relational obligations

Fix the molecular interpretation and syntax before choosing an algorithm.
For the research target, retain note 127's labelled, single-character model:

- `G` is the target molecular graph with distinct physical atom identities.
- `q` records the actual prefix and labels of atom occurrences already begun.
- `Live_G(q)` means that a complete labelled serialization extends `q`.
- `Next_G(q)` contains exactly the raw immediate character/atom-label steps
  whose resulting states are live.
- EOS is available exactly when the current labelled prefix is accepted.

Distinct physical successors remain distinct even when they emit the same
character. Multiple satisfying assignments witnessing one successor do not
create duplicate choices. A coarser writer action interface requires an
explicit refinement argument; its bounds do not automatically establish
the single-character result. Normalized physical observations have finite
domains even when the syntax admits unbounded numeric padding. Padding and
partial tokens still require actual lexical records and charged updates.

Let `Z_G` denote the physical observation variables fixed in note 127:
normalized atom lexemes, local incidence orders, and each edge's direct or
ring representation and associated endpoint annotations. Let `R_G(q)` be
the joint tuples of these observations induced by complete serializations
extending `q`. The compiler must construct factors `F_G(q)` satisfying

\[
\pi_{Z_G}\operatorname{Sol}(F_G(q))=R_G(q).
\]

Equality is required after initialization and every raw candidate update,
including dead candidates. Agreement on a feasibility bit is insufficient.
The physical tuple can erase global ordering information, so advancing the
actual grammar state need not be expressible as unary conditioning alone.

Variables have physical or actual-control owners: atoms, edges, actual open
ring slots, actual stack frames, and the current lexical record. Their
domains describe bounded local alternatives and actual attachments. Future
global visit positions, future branch-end times, future atom subsets, and
continuation strings are excluded. Actual stack order remains legitimate
procedural state. Identifier storage and access costs remain charged.

## Kernel boundary

The kernel receives finite relations, explicit edits, and indexed families
of labelled immediate edits:

```text
compile(factors, composition, indexed_choice_families) -> representation
supported(representation) -> iterator of supported edit handles
commit(representation, returned_handle) -> updated representation
```

Relations support insertion, removal, restriction, and replacement. A
returned handle identifies a supported edit at the current representation
version and the delta needed to install it. Committing it maintains the
representation without repeating a completion decision. Other handles can
be discarded after selection; retaining a branching history needs separate
storage accounting. Stale handles must not be applied to another version.

The kernel has no writer callback, hypothetical transition function, suffix
generator, or arbitrary feasibility oracle. Factor evaluation itself must
be an explicit finite table or specified relation operation with accounted
cost. A predicate named `CompatibleFutureTraversal` is not an implementation.

The adapter manages actual parser/graph records and maps immediate steps to
edits. It must prove the correspondence with `Next_G(q)` and implement the
separate acceptance/EOS decision. Neither the adapter nor factor generation
may move the forbidden future search outside the nominal kernel boundary.

## Exact boundary relations

Start with explicit boundary tables so that both semantics and costs are
inspectable. Partition base factors among leaves of a binary composition
tree. For a region `t`, let `F_t` be all base factors below it, `B_t` the
variables that must remain exposed, and `I_t` its internal variables. Store

\[
M_t(B_t)=\exists I_t\;\bigwedge_{f\in F_t}f.
\]

Each row is a **joint** boundary assignment extendible inside that region.
It is not a collection of independently supported values. For children
`l,r`, compose by natural join and existential projection:

\[
M_t=\pi_{B_t}(M_l\Join M_r).
\]

Natural join combines assignments agreeing on common variables; projection
removes coordinates and duplicate rows. Every variable shared across the
cut must remain exposed until its last relevant use. A decomposition that
forgets such a variable is invalid. Zero-ary tables distinguish true
(`{()}`) from false (the empty relation).

For example, over binary variables,

\[
R(p,u)=[p=u],\qquad Q(u,q)=[u\ne q]
\]

give `pi_{p,q}(R join Q) = [p != q]`. Eliminating `u` preserves its
correlated consequence without choosing a complete assignment.

Keep the base factors and their dependency structure. A later restriction
on `u` invalidates the affected summaries; the projected table alone cannot
reconstruct that update. Likewise, a zero-ary root is only a feasibility
summary, not the full physical relation. The retained factor system and
composition represent that relation; discarding everything except the root
would violate the projected-relation obligation.

Already-built components with disjoint variable sets retain a product node
and cached nonemptiness, with additive storage and composition work. Do not
materialize their Cartesian product. Disconnected molecular components are
not automatically independent: grammar control and ring resources can
couple them. Discovering independence is charged work.

Sparse tables or specialized parity/symbolic operators can replace explicit
tables later, provided they implement the same relation with stated costs.
Variable elimination and FAQ/InsideOut provide relevant algebraic background
[3]; they do not supply the missing molecular construction.

## Supported actions without a rejected-candidate scan

A Boolean root does not by itself give an efficient successor iterator.
Maintain two summaries at every region:

- `M_t^0`: boundary assignments feasible with no candidate edit in the region.
- `M_t^1`: boundary assignments feasible with exactly one candidate edit in
  the region, existentially ranging over its action owners.

At an action-owning leaf, keep the unchanged module relation `L_t^0` and
the replacement relation `L_{t,a}` for each currently applicable action `a`.
Project them to the same boundary; `M_t^1` is the union of the replacement
summaries. A leaf with no candidates has an empty one-edit summary.

For internal nodes:

\[
M_t^0=\pi_{B_t}(M_l^0\Join M_r^0),
\]

\[
M_t^1=\pi_{B_t}\left[
 (M_l^1\Join M_r^0)\cup(M_l^0\Join M_r^1)
\right].
\]

The unchanged sibling ensures that exactly one action is applied. There is
no `M_l^1 join M_r^1` term. Action identity stays at its unique owning leaf,
not in a global variable ranging over all actions.

Enumeration needs an unchanged outside context, not just `M_t^1`. Let
`C_t` be the feasible boundary assignments from factors outside `t`.
Start with true at an empty-boundary root and derive contexts on descent:

\[
C_l=\pi_{B_l}(C_t\Join M_r^0),\qquad
C_r=\pi_{B_r}(C_t\Join M_l^0).
\]

Descend into `t` only if `C_t join M_t^1` is nonempty. At a leaf, test the
bounded local alternatives against `C_t`, emitting each supported action
identity once. The descent traverses owners, not solution witnesses, so
overlapping witness sets produce no duplicate outputs. A whole unsupported
region is skipped by one bounded relation computation.

Compute contexts on demand along the iterator's path, or use an explicitly
charged lazy cache. Eagerly refreshing every outside context after one edit
could take linear work and invalidate the proposed update bound.

This construction has substantive prerequisites:

1. Each action has exactly one owner and its **whole joint effect** is a
   replacement inside that module. An action changing several distant
   factors cannot be represented as independent one-edit alternatives.
   Grouping those factors may enlarge the interface and its cost.
2. Interfaces cover the union of scopes needed by unchanged and candidate
   relations. New variables or changed scopes need a valid interface update
   or charged rebuilding; they cannot be silently projected away.
3. Alternatives per leaf are parameter-bounded, or a further exact action
   index with proved bounds is supplied. An unbounded leaf scan would restore
   the rejected-candidate problem.
4. Candidate-family generation, applicability changes, and index maintenance
   are included in the cost. A local actual step need not change only a local
   set of candidate records; that locality must be established.

Dynamic enumeration on trees [4] is relevant precedent for maintained answer
representations, not a theorem that these prerequisites hold for SMILES.

## Conditional costs and their limits

Suppose a valid composition tree with `n` modules is supplied, its height is
`O(log(2+n))`, domains have size at most `d`, and every join/projection,
including context construction and candidate replacement, involves at most
`k` variables. Assume parameter-bounded alternatives per module and bounded
word-sized tuple operations. Then straightforward explicit tables give:

| Operation | Conditional work/storage |
| --- | --- |
| Initial summaries and their storage | `O(n d^{O(k)})` |
| Maintenance after `r` changed module records | `O(r d^{O(k)} log(2+n))` |
| Iterator delay, including first result and exhaustion | `O(d^{O(k)} log(2+n) + output size)` |

The update bound follows by recomputing affected ancestor paths. The delay
bound follows by pruning empty subtrees and traversing at most a tree-height
path between supported leaves, with bounded local alternatives. Neither
bound permits deferred uncharged reconstruction at iterator startup.

Here `r` includes changes to both base relations and candidate-family
records. These bounds exclude no hidden work: producing those changes,
constructing or rebalancing a valid decomposition, and reading input tables
must be added explicitly. Merely balancing an arbitrary tree does not prove
that its interfaces retain width `k`. Multiword values require their actual
operation and storage costs.

There is no established small bound on `k` for the molecular problem.
Bounded molecular degree does not imply bounded factor interaction width.
Nor is the required relation between `r` and the actual parser delta proved.
Note 127 seeks bounds depending on fixed semantic parameters and actual
changes, with near-linear preprocessing and polylogarithmic delay. A result
with an additional width parameter is a weaker, explicitly qualified result.

Width-dependent knowledge compilation can require exponential size for
specified representation classes [5]. That is not a lower bound for every
possible SMILES algorithm. It does explain why calling a representation a
circuit or cache is not an efficiency argument.

If width or update scope becomes unaffordable, expose the failing relation
and cost. Improve the encoding/operators or state a weaker theorem. Do not
silently switch to a schedule planner, and do not report resource exhaustion
or an unsupported encoding as mathematical infeasibility.

## The unresolved molecular construction

The first domain construction must confront finite reusable ring resources,
irreversible endpoint phases, actual branch-return ownership, and constrained
incidence order **jointly**. A writer with nonbinding ring capacity followed
by later resource and stereo patches does not settle this interaction.
Stereo order/configuration relations may initially be supplied as explicit
finite tables, but their coupling to traversal and resource use must already
be represented. Specified and unspecified stereo remain part of the chosen
identity contract.

For every factor/template, specify:

| Obligation | Required evidence |
| --- | --- |
| Variables and owners | Concrete domains with physical/actual meanings |
| Tuple semantics | Explicit local relation, not a completion predicate |
| Construction | Finite algorithm and cost, without a future planner |
| Actual update | Identified base and action-index edits with charged work |
| Boundary | All surviving correlations and their representation sizes |
| Correctness | Soundness and completeness for the fixed physical projection |

In particular, prove that every satisfying physical tuple is realizable by
an allowed completion, and that every allowed completion supplies such a
tuple. Local consistency tests and a collection of successful examples do
not establish either implication globally.

Two obstacles remain separate: existence of the required locally updatable
physical encoding, and affordable exact computation on that encoding. The
relation kernel resolves neither by assumption.

## First implementation and acceptance boundary

Implement a small finite-relation kernel: natural join, projection, product
nodes, dependency-aware replacement, zero/one-edit summaries, on-demand
outside contexts, and an iterator returning versioned action handles.
Develop the coupled molecular encoding against this interface alongside it.
Do not surround a missing encoding with a production walker that needs a
completion-search rescue path.

Validate algebra against direct finite relation semantics: preservation of
joint correlations, empty and zero-ary relations, replacement after internal
projection, outside-context rejection, duplicate witnesses for one action,
exactly-one-edit composition, independent products, and invalidated handles.
Check first-result/exhaustion behavior on large rejected regions and charge
action-index maintenance explicitly. Tiny exhaustive domain examples can
serve as test oracles; they are not runtime dependencies or proofs of the
general molecular construction.

Only after the encoding and its scope are justified does the writer consume
the kernel: select a supported edit, update actual records and relations,
then emit the associated text. Counts, determinization, sampling, persistent
branch trees, and full streams need their own semantics and cost arguments;
Boolean feasibility summaries do not establish those capabilities.

The deliverable is an exact computational algebra with inspectable costs and
a separately justified physical encoding. Any remaining obstruction must be
identified as a specific relation, interface, update, or bound.

## References

The earlier notes are on `south-star-2`; these links are pinned because they
are not files on `main` at the time of this note.

1. [Note 127: Factored online SMILES generation — formal problem](https://github.com/numpde/grimace-py/blob/4ebf915564bcb0c209514fb73a6fb0e2cb53ca09/notes/127_factored_online_smiles_formal_problem.txt).
   Defines the interpretation, physical observation relation, owner restrictions,
   exact online interface, and strong work/storage/delay targets.
2. [Note 128: Exact incremental SMILES](https://github.com/numpde/grimace-py/blob/4ebf915564bcb0c209514fb73a6fb0e2cb53ca09/notes/128_exact_incremental_smiles.tex).
   Companion formalization from the preceding discussion.
3. Mahmoud Abo Khamis, Hung Q. Ngo, Atri Rudra.
   [FAQ: Questions Asked Frequently](https://arxiv.org/abs/1504.04044).
   Variable elimination and width-sensitive aggregate-query evaluation.
4. Antoine Amarilli, Pierre Bourhis, Stefan Mengel, Matthias Niewerth.
   [Enumeration on Trees with Tractable Combined Complexity and Efficient Updates](https://arxiv.org/abs/1812.09519).
   Dynamic enumeration for queries represented by tree automata.
5. Antoine Amarilli, Mikaël Monet, Pierre Senellart.
   [Connecting Width and Structure in Knowledge Compilation](https://arxiv.org/abs/1709.06188).
   Width-dependent upper and lower bounds for specified compiled representations.
