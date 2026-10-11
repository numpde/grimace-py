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
relevant QA/QC skills sequentially. Verify candidate, patch, and runtime import
identity; confirm each finding against its owning contract before fixing it, then
rerun affected earlier lenses. Keep source-bounded writer parity separate from
semantic parse-back. Finish only after the final relevant pass yields no new
actionable findings. Preserve unrelated dirty work.

Update this note with verified progress, decisions, and outstanding gates.
Add another note only when a durable architectural decision needs a separate
home. Commit reviewed slices with explicit paths; do not infer publication from
a local commit or local test result.

| Milestone | Status | Evidence or next action |
| --- | --- | --- |
| Roadmap | Independently reviewed | Source audit and fresh reader teach-back passed; root-frontier cost scope clarified. |
| CI coverage | Focused source and installed-artifact gates added and locally exercised | The workflow explicitly selects bounded South Star 2 exact support, the public prefix contract, and bridge/preparation regressions. The installed-package runner also selects public-prefix and exact-support tests. Remote CI has not been observed. |
| Molecular preparation and mixed validation | Bounded C/O preparation and six pinned exact-support cases independently audited; mixed-state validation pending | The RDKit adapter prepares neutral, nonaromatic aliphatic C/O graphs with nonaromatic single bonds, checks implicit H against graph valence, and supplies fixed atom text without consulting the SMILES writer. Eleven preparation tests and one native parse-back test cover CCO, CC(C)O, and C1CC1; the emitted strings parse to the intended graph and preserve unspecified atom stereo. A bounded probe prepared 874 sanitized nonaromatic neutral C/O single-bond graphs with 1-4 atoms; RDKit implicit H matched C4/O2 valence minus degree for all, while 9 aromaticized candidates were outside this slice. A pinned RDKit 2026.03.1 fixture now establishes exact support for CCO, CC(C)O, C1CC1, C.O, and two atom-renumbered CC.O inputs. Source-bounded root/child-order and component-construction analysis gives complete upper bounds; a positive seeded writer witness replays every expected output. Native support equals all six sets, including C.O and both renumberings. This exact parity claim covers only those fixed non-stereo, single-bond inputs and atom orders. `WriterRegime` owns component mapping with exactly three private spellings: `generic_any_order`, `generic_input_order`, and `rdkit_random_noncanonical`; the RDKit preparation adapter selects `rdkit_random_noncanonical` and thus `InputOrder`. Under that regime, valid configured directional requests return `Incomplete` pending RDKit anchoring, after invalid input is checked. The measurement probe now requires `--writer-regime` under schema v2; its historical `2ae79ec` measurements are unchanged and a regime-specific rerun remains pending. Broader chemistry and mixed-state validation remain open. |
| Tetrahedral preparation and native transport | Bounded C/O tetrahedral slice independently audited | Preparation freezes tetrahedral facts into immutable native input, using actual incident-bond order and the RDKit CW/CCW tag to derive the reference parity; the single non-graph hydrogen occupies the final reference position. Exact support matches both pinned 16-string enantiomer sets, with renumbering checked separately. Graph/hydrogen/chirality parse-back and ambiguous-prefix sibling ownership are separate semantic tests. The boundary supports degree-three carbon with exactly one non-graph hydrogen; wider hydrogen forms, bracket spelling, and broader chemistry remain unsupported. |
| Stereo writer evidence | Pinned evidence-only fixture independently audited | Complete source-bound plus positive-witness sets cover 16/16 tetrahedral cases and 20 mixed alkene/tetrahedral cases. Exhaustive E- and Z-butene evidence each checks six source-derived writer schedules yielding three strings; the Z fixture records RDKit’s controlling-direction flip before traversal-flip compensation. Two simple unspecified-stereo controls are separate from the triene writer quirk: the in-scope triene output parses with stereo assigned to its otherwise unspecified middle alkene, so this records a membership/parse-back discrepancy and does not establish universal unspecified-stereo preservation. No native comparisons are part of this evidence-only slice; stereo transport and native anchoring remain separate gates. |
| Text-prefix decoder | Native client and private Python transport implemented and independently audited | Selecting a token text retains every matching native state, unions their next-token texts, and accepts EOS when any retained state accepts. Python exposes these operations and `frontier_size`, the count of retained native states. Seven bridge, eleven preparation, and one parse-back test pass against the compiled extension. A generic disconnected `C.C.C` client test exercises hidden-state retention, token union, and EOS; it is not authority for exact RDKit writer support. An `AnyOrder` baseline measurement at `2ae79ec` covered 11 cases with five repeats and 90 accepted semantic paths. Four disconnected ethane components reached a retained physical frontier of 192 before `choices()` generated a 384-state successor batch; five components reached 80 before a 480-state batch. Fixed local `Walker` transitions measured roughly 3-6 microseconds with no large spectator-cost growth observed. These capped measurements do not prove constant-time behavior, attribute all costs, or establish RDKit parity; the probe did not select pruning or state merging. These are historical `AnyOrder` baseline results; no new measurements are claimed for the regime adapter. Rerun with the RDKit-selected writer regime. Broader exact pinned support comparison remains pending; the six bounded cases are recorded under molecular preparation. The seeded mixed-frontier EOS test establishes no reachable-prefix liveness claim. |
| Public prefix decoder API | Minimal additive `grimace.decoding.start(mol)` API independently audited | The public API exposes immutable prefix states with `is_accepted` and token-text `choices()`, backed by the native client without old enumeration engines or support snapshots. Complete output sets match pinned writer support for eleven bounded C/O and stereo cases; separately mapped graph/hydrogen/stereo checks cover semantic identity, unspecified-stereo controls, and the known triene parse-back discrepancy. Exhaustive reachable-prefix liveness, EOS union, and sibling ownership are checked only across this bounded corpus. RDKit 2026.03.1 is checked when `start` is called; imports and the existing API remain usable under actual RDKit 2026.03.4. Broader chemistry, remote CI observation, and the corrected regime-specific measurement remain open. |
| Private native Python bridge probe | Implemented and independently reviewed | Private `_core` classes expose fixed atom/bond preparation, accepted state, and owned successor choices. Five bridge tests and both Rust test packages pass locally. Direct wheel and sdist-rebuilt wheel installations each passed those five bridge tests plus 15 public smoke tests, with imports verified from site-packages. This validates the private bridge packaging path, not molecular preparation, text-prefix decoding, or the complete molecular path. |
| Native directional anchoring and preparation | Bounded C/O directional transport and prospective anchor independently audited | Python preparation transports configured double bonds as typed immutable native facts with endpoint/carrier bond IDs and E/Z phase; it rejects unsupported or ambiguous carrier references, and the native wire requires integer IDs plus an actual boolean phase. The native writer installs eligible carrier requirements as CSP constraints before token publication; visitation and candidate ordering are unchanged, and generic symmetry remains intact. The anchor scope requires one carrier per endpoint, distinct endpoints and carrier bonds across relations, and acyclic directional double/carrier bonds; outside that scope the writer reports `Incomplete`. Bounded E-butene, Z-butene, and mixed alkene/tetrahedral grouped-prefix sets match the pinned fixture/source rows; these exact comparisons remain separate from semantic parse-back. The pinned triene writer quirk remains: a witness can assign direction to its otherwise unspecified middle alkene. Integrated checks passed 38 focused Python tests, six pinned C/O/stereo evidence tests, 243 South Star 2 Rust tests, and 86 root Rust tests. No broader RDKit parity or unspecified-stereo preservation is claimed. |
| Semantics and cost closure | Pending | Keep parse-back, exact parity, and measurement separate. Native work counters and the literal continuation argument remain under implementation; no native-allocation proof is claimed. |
| Python interface and installed artifacts | Bounded public API and direct/sdist artifact gates passed locally; remote CI unobserved | Direct and sdist-rebuilt wheels each passed 101 installed-package tests with one skipped test. Runtime modules imported from each venv’s site-packages with RDKit 2026.03.1; the isolated test source was confirmed through `tests.__file__`. The 38-test private source regression suite passed, and unsupported RDKit versions fail the pinned parity/support gates. |

The native interface preserves typed spelling and stereo inputs and publishes
text with its validated successor. Solver and traversal stores remain private.
Sequential code-beauty, error-contract, and anti-slop reviews found no
actionable issues after the full pass stabilized. Formatting and diff checks
pass. Ordinary Clippy completes, but a strict `-D warnings` run fails on
existing implementation diagnostics (dead code and four Clippy lints outside
the new interface); that strict gate remains open.
