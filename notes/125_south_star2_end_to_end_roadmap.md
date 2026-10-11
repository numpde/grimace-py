# South Star 2 end-to-end roadmap

Date: 2026-10-11

Baseline: `fa9b0cf7` (`south-star-2`), Qualify directional ring endpoint
composition.

## Outcome

Build an online molecular SMILES decoder through this complete path:

```text
RDKit molecule
-> independently prepared molecular facts and declared writer policy
-> Rust semantic walker
-> all compatible states for the selected text prefix
-> next token texts, successor selection, and EOS
-> accepted SMILES
```

EOS means that the current text prefix is an accepted complete string. The
decoder must also retain longer continuations when another compatible state
can emit more text.

Add the new decoder alongside the existing public APIs, preserving their
behavior. Replacing an existing runtime is a separate migration decision.

Note 114 remains the governing kernel architecture. This roadmap orders its
implementation and product validation; it does not replace that contract.
The Rust walker computes immediate semantic choices without constructing full
support, suffix plans, proofs, counts, or continuation assets. Molecular
preparation owns chemical facts and spelling policy. The text-prefix client
owns ambiguity between semantic states that emit the same text.

## Current evidence and limits

The baseline implements graph-general traversal, disconnected composition,
token-by-token ring endpoints, persistent state forks, exact finite-domain
projection, ring-coupled tetrahedral order and parity, selectable directional
carriers, and directional ring placement. The crate's 224 library tests passed
locally during roadmap review:

```bash
cargo test -p grimace-south-star2 --lib --offline --quiet
```

The walker remains private. No South Star 2 molecular adapter or text-prefix
client exists at this baseline. Its prepared inputs supply atom strings,
tetrahedral parity strings, and directional relations. Agreement on those
inputs does not establish molecular ingestion, chemical spelling, or RDKit
writer support. The existing Python API still uses the older implementations.

Independent non-stereo conformance covers all 76 labelled simple graphs with
zero through four atoms and curated explicit ring cases. Stereo tests add
independent factor and selected transition checks. These are strong bounded
checks, not a general proof that every combined stereo successor has an
accepting continuation.

The current CI Rust command is `cargo test --lib`. Cargo workspace metadata
selects only the root `grimace-kernel` package by default, so that command does
not run the South Star 2 tests.

Persistent forks and affected-factor queues establish useful locality, but a
spanning-tree projector can scan a whole connected molecular component.
Zero mixed-search calls or sparse page copies alone do not establish bounded
local choice cost as molecule size grows.

## Milestones and acceptance gates

### 1. Protect existing evidence

Add an explicit South Star 2 library-test step to the existing Rust CI job.
Preserve the root package and existing Python, parity, and packaging lanes.

Exit: the intended CI workflow runs both Rust packages' library tests. Keep
local verification distinct from an observed remote CI result.

### 2. Prepare and walk real molecules privately

Add the smallest independent molecular preparation boundary and test-only
integration path. Begin with unrooted `CCO`, a branch, a cycle, and a small
mixed-stereo case using the kernel's existing relations. Include unspecified
stereo controls early. Keep incomplete integration out of public exports.

The initial molecular comparison targets RDKit `2026.03.1` in the pinned
`canonical=False, doRandom=True` writer regime. Declare its precise spelling,
rooting, and component-order policy with the first case. Preparation must own
atom and bond spelling, hydrogen conventions, reference ligand orders,
directional carrier eligibility, endpoint orientation, root semantics, and
component order. Use older Grimace and South Star 1 externally for differential
evidence; do not call their derivation machinery inside the new walker.

Alongside ingestion, challenge a bounded set of reachable states combining
tetrahedral order, selectable directional carriers, ring placement, pending
tokens, label reuse, and disconnected components. Use independent primitive
inputs and transition expectations. Write down the compact continuation
argument: why represented graph decisions and stereo assignments admit an
executable remaining writer schedule. Do not add recursive suffix searches to
production choices.

Exit:

- independently checked preparation facts and spelling;
- graph and stereo parse-back, including preservation of unspecified stereo;
- exact semantic successor agreement on the bounded mixed corpus;
- source and sibling isolation, correct failure precedence, and no unexplained
  reachable nonterminal dead ends in that corpus;
- a stated scope for the continuation argument and remaining evidence gaps.

A general liveness theorem is not a prerequisite for this private experiment.
Its bounded results must not be presented as a universal legal-prefix guarantee.

### 3. Decode selected text without losing hidden alternatives

Implement a separate client that retains every semantic state compatible with
the selected token sequence. Kernel choices already preserve distinct
successors with equal text; the client must preserve that behavior when users
select text rather than a particular branch.

For `CCO` with no fixed atom root, the initial `C` can arise from either carbon
root. Selecting
one arbitrarily loses valid suffixes. Token selection must retain both, expose
the union of subsequent token texts, and allow EOS whenever any retained state
accepts. Accepted states contribute EOS; other states may still contribute
longer continuations.

Do not compile the whole language to implement this client. Introduce state
merging only when its semantic equivalence and need are established. Measure
frontier growth on ordinary symmetric molecules rather than assuming it stays
small.

Exit: selected token sequences retain all compatible successors, EOS and token
union are correct, and accepted strings equal the independently pinned support
for small declared cases. Preserve string support separately from witness
multiplicity; counting is not part of this milestone.

### 4. Close demonstrated semantics and cost gaps

Keep two correctness gates separate:

- semantic preservation: the output parses to the intended graph and stereo,
  including absence of newly assigned stereo;
- writer parity: the output belongs to the exact declared, version-pinned
  RDKit writer language.

Use complete expected sets with defensible provenance for bounded exact parity.
Random RDKit samples and agreement with prior implementations are useful
falsification evidence, but do not establish complete RDKit support. Record
each discrepancy against the declared policy before choosing a correction.

Investigate whether directional marks for configured stereo can assign an
otherwise unspecified potential alkene. First obtain a real witness or show
that the prepared carrier inventory and policy prevent it. Add a negative
constraint only if the molecular semantics require it; do not invent a factor
merely to satisfy a checklist. Unspecified tetrahedral centers should receive
plain atom spelling at preparation.

Measure costs alongside milestones 2 and 3, rather than waiting for completion:

- preparation, first choices, and representative later choices;
- spanning-projector atom and edge visits, latent-factor checks, exact-search
  branches, and persistent page copies;
- the number of compatible states retained after text selection.

Hold one local transition fixed while growing disconnected spectator material,
connected remote graph structure, distant inactive stereo centers, and active
coupled centers. Attribute connected-component work separately from genuine
semantic coupling and prefix ambiguity. Timing experiments stay outside
correctness assertions.

Exit: the declared bounded semantic and parity cases pass; unrelated
disconnected material does not increase semantic work for a fixed already-active
transition; connected growth is measured and explained; any unacceptable
demonstrated cost is corrected and
revalidated against independent projections. Do not promise constant-time
exact CSP solving or choose a replacement solver before measurements justify
it.

Initial and component-boundary frontiers may legitimately gain root choices
when disconnected atoms are added. Account for those required choices
separately from propagation for an unchanged active transition.

### 5. Expose the smallest coherent Python decoder

Once the private molecular and text-prefix path passes its gates, add a narrow
PyO3 bridge for preparation, choices, token selection, and acceptance. Keep the
Rust walker and its native tests independent of RDKit and Python. Python owns
RDKit interop; the compiled extension owns runtime transitions.

Public exposure requires a stated language and liveness scope, correct
equal-text handling, preserved stereo, measured frontier costs, and honest
input, spelling, backend, and invariant failures. Existing public APIs must
retain their behavior.

Exit: installed wheel and source-distribution checks exercise the new path;
normal next-choice operation does not call older enumerators, construct full
support, compile assets, or search suffixes; public examples describe the
implemented contract and its verified limits.

## Decisions from peer deliberation

Three focused peers reviewed semantics, solver locality, and product
integration. Their objections changed the implementation order:

- Bring real molecules and mixed-state validation together now, rather than
  completing an indefinite sequence of symbolic kernel slices first.
- Treat text-prefix ambiguity as a required decoder behavior, not a cosmetic
  grouping view or a counting problem.
- Establish language policy and negative stereo semantics during preparation;
  parse-back correctness cannot substitute for writer parity.
- Measure connected-component work and prefix-frontier growth before asserting
  scalable local decoding.
- Keep the private demonstrator available for bounded experiments while the
  stronger public language and continuation contracts are being established.

Counts, probabilities, portable snapshots, continuation caches, alternative
solver backends, and proof machinery remain later clients. They must not become
prerequisites for the first useful online molecular decoder. Revisit them when
the working decoder or a concrete consumer establishes their required contract.

## Working method and progress

Use small coherent slices with delegated authorship, review, QA/QC, and
validation; the coordinator orchestrates and adjudicates findings. Apply
relevant QA/QC skills sequentially. Verify each finding, fix confirmed issues,
and rerun affected earlier lenses. Continue the full relevant pass until it
produces no new actionable findings; an initial clean review is not final
acceptance. Preserve unrelated dirty work.

Update this note with verified progress, decisions, and outstanding gates.
Add another note only when a durable architectural decision needs a separate
home. Commit reviewed slices with explicit paths; do not infer publication from
a local commit or local test result.

| Milestone | Status | Evidence or next action |
| --- | --- | --- |
| Roadmap | Independently reviewed | Source audit and fresh reader teach-back passed; root-frontier cost scope clarified. |
| CI coverage | Implemented and independently audited | Explicit new-crate step; 86 root and 224 South Star 2 tests passed locally. Remote CI not observed. |
| Molecular preparation and mixed validation | In progress | Native cross-crate interface audited; 230 crate tests pass. Molecular adapter and mixed-state validation remain pending. |
| Text-prefix decoder | Native client implemented and independently audited | For unrooted `CCO`, selecting `C` retains both carbon roots, unions next-token texts, and accepts EOS when any retained state accepts. The 230 South Star 2 tests pass locally. The seeded mixed-frontier EOS test checks only client choice/acceptance behavior across deliberately unrelated prepared graphs; it establishes no reachable-prefix liveness claim. Private Python prefix binding, ordinary symmetry-growth measurements, and exact pinned support comparison remain pending. |
| Private native Python bridge probe | Implemented and independently reviewed | Private `_core` classes expose fixed atom/bond preparation, accepted state, and owned successor choices. Five bridge tests and both Rust test packages pass locally. Direct wheel and sdist-rebuilt wheel installations each passed those five bridge tests plus 15 public smoke tests, with imports verified from site-packages. This validates the private bridge packaging path, not molecular preparation, text-prefix decoding, or the complete molecular path. |
| Semantics and cost closure | Pending | Establish separate parse-back, exact parity, and measurement lanes. |
| Python interface and installed artifacts | Pending | Follow the private molecular and prefix gates. |

The native interface preserves typed spelling and stereo inputs and publishes
text with its validated successor. Solver and traversal stores remain private.
Sequential code-beauty, error-contract, and anti-slop reviews found no
actionable issues after the full pass stabilized. Formatting and diff checks
pass. Ordinary Clippy completes, but a strict `-D warnings` run fails on
existing implementation diagnostics (dead code and four Clippy lints outside
the new interface); that strict gate remains open.
