# South Star 2 end-to-end roadmap

Date: 2026-10-11

Historical baseline: `fa9b0cf7` (`south-star-2`), Qualify directional ring endpoint
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

## Installed RDKit and version-keyed writer qualification

The installed RDKit supplies molecule facts; Rust continues under one fixed
`rdkit_random_noncanonical` convention. The public runtime does not select a
policy based on the RDKit version or certify arbitrary serializer parity.
Exact support is qualified separately with version-keyed South Star 2 fixtures.
The earlier eleven-case public comparison selected only the 2026.03.1 fixture
and is historical; current tests select the fixture matching the installed
version. The present bounded .1 and .4 evidence covers only represented
molecules, atom order, and supported facts. Runtime success alone does not
qualify an arbitrary release or unrepresented writer behavior.

The public tests and named qualification runners require the installed
version's exact-support and stereo-evidence fixtures. Missing either fixture
family fails the gate rather than falling back to 2026.03.1. Preparation may
also need revision if a future RDKit changes how supported facts are
represented or interpreted. Broader chemistry and future releases require
their own source-backed, version-keyed fixtures and independent semantic and
writer-parity checks.

## Historical baseline evidence and limits

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

At that baseline, the CI Rust command was `cargo test --lib`. Cargo workspace metadata
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
actionable findings. Strict Clippy remains diagnostic feedback, not a roadmap
acceptance gate. Preserve unrelated dirty work.

Update this note with verified progress, decisions, and outstanding gates.
Add another note only when a durable architectural decision needs a separate
home. Commit reviewed slices with explicit paths; do not infer publication from
a local commit or local test result.

| Milestone | Status | Evidence or next action |
| --- | --- | --- |
| Roadmap | Independently reviewed | Source audit and fresh reader teach-back passed; root-frontier cost scope clarified. |
| CI coverage | Version-selected South Star 2 qualification is wired for the checked .1 and .4 fixture sets; remote CI unobserved | The source-tree parity job validates every checked-in upstream serializer source manifest and file digest before its pinned parity gates; the full source-contract test passed locally. The legacy installed-package aggregate and broad pinned parity remain on RDKit 2026.03.1. Separate wheel and sdist lanes install RDKit 2026.03.4 and select only the public-prefix, bounded South Star 2 exact-support, preparation, semantic-control, and source-evidence gates; they do not run the legacy aggregate at .4. Both fresh local artifact forms passed their installed gates on actual .1 and .4 interpreters. No remote CI result is claimed. |
| Molecular preparation and mixed validation | Six pinned C/O support sets and bounded mixed validation independently audited; broader chemistry pending | The RDKit adapter prepares neutral, nonaromatic aliphatic C/O graphs with nonaromatic single bonds, checks implicit H against graph valence, and supplies fixed atom text without consulting the SMILES writer. Eleven preparation tests and one native parse-back test cover CCO, CC(C)O, and C1CC1; the emitted strings parse to the intended graph and preserve unspecified atom stereo. A bounded probe prepared 874 sanitized nonaromatic neutral C/O single-bond graphs with 1-4 atoms; RDKit implicit H matched C4/O2 valence minus degree for all, while 9 aromaticized candidates were outside this slice. A pinned RDKit 2026.03.1 fixture now establishes exact support for CCO, CC(C)O, C1CC1, C.O, and two atom-renumbered CC.O inputs. Source-bounded root/child-order and component-construction analysis gives complete upper bounds; a positive seeded writer witness replays every expected output. Native support equals all six sets, including C.O and both renumberings. This exact parity claim covers only those fixed non-stereo, single-bond inputs and atom orders. `WriterRegime` owns component mapping with exactly three private spellings: `generic_any_order`, `generic_input_order`, and `rdkit_random_noncanonical`; the RDKit preparation adapter selects `rdkit_random_noncanonical` and thus `InputOrder`. Under that regime, configured directional requests are anchored within the bounded native carrier scope; unsupported valid requests return `Incomplete` after invalid input is checked. The measurement probe requires `--writer-regime` under schema v2; historical `2ae79ec` `AnyOrder` results remain unchanged, and the separate RDKit-regime measurements are recorded under semantics and cost closure. Broader chemistry remains open. |
| Tetrahedral preparation and native transport | Bounded C/O tetrahedral slice independently audited | Preparation freezes tetrahedral facts into immutable native input, using actual incident-bond order and the RDKit CW/CCW tag to derive the reference parity; the single non-graph hydrogen occupies the final reference position. Exact support matches both pinned 16-string enantiomer sets, with renumbering checked separately. Graph/hydrogen/chirality parse-back and ambiguous-prefix sibling ownership are separate semantic tests. The boundary supports degree-three carbon with exactly one non-graph hydrogen; wider hydrogen forms, bracket spelling, and broader chemistry remain unsupported. |
| Stereo writer evidence | Pinned evidence-only fixture independently audited | Complete source-bound plus positive-witness sets cover 16/16 tetrahedral cases and 20 mixed alkene/tetrahedral cases. Exhaustive E- and Z-butene evidence each checks six source-derived writer schedules yielding three strings; the Z fixture records RDKit’s controlling-direction flip before traversal-flip compensation. The two pinned simple unspecified-stereo controls pass parse-back without assigned stereo. The in-scope triene remains a known RDKit writer/semantic discrepancy: the supported writer string can add direction to its otherwise unspecified middle alkene. This behavior is retained under the exact writer-parity contract and is not a universal stereo-preservation promise. These fixture expectations are evidence-only; typed native transport and anchoring are covered separately in the directional preparation and native work-count checks. |
| Text-prefix decoder | Native client and private Python transport implemented and independently audited | Selecting a token text retains every matching native state, unions their next-token texts, and accepts EOS when any retained state accepts. The private Python bridge exposes these operations and `frontier_size` as a diagnostic. Public prefix states expose only `is_accepted` and `choices()`; `grimace.decoding.start(mol)` creates them. Seven bridge, eleven preparation, and one parse-back test pass against the compiled extension. A generic disconnected `C.C.C` client test exercises hidden-state retention, token union, and EOS; it is not authority for exact RDKit writer support. An `AnyOrder` baseline measurement at `2ae79ec` covered 11 cases with five repeats and 90 accepted semantic paths. Four disconnected ethane components reached a retained physical frontier of 192 before `choices()` generated a 384-state successor batch; five components reached 80 before a 480-state batch. Fixed local `Walker` transitions measured roughly 3-6 microseconds with no large spectator-cost growth observed. These capped measurements do not prove constant-time behavior, attribute all costs, or establish RDKit parity; the probe did not select pruning or state merging. These `AnyOrder` results remain historical; separate RDKit-regime measurements are recorded under semantics and cost closure. Timings do not establish writer parity. The public API compares eleven bounded exact-support cases; broader parity remains pending. The original seeded mixed-frontier EOS test alone establishes no liveness claim; separate public tests check the eleven-case corpus. |
| Public prefix decoder API | Minimal additive `grimace.decoding.start(mol)` API independently audited | The public API exposes immutable prefix states with `is_accepted` and token-text `choices()`, backed by the native client without old enumeration engines or support snapshots. Complete output sets match the installed RDKit version’s South Star 2 fixtures for the bounded C/O and stereo cases; separately mapped graph/hydrogen/stereo checks cover semantic identity, unspecified-stereo controls, and the known triene parse-back discrepancy. Exhaustive reachable-prefix liveness, EOS union, and sibling ownership are checked only across these bounded cases. The runtime does not select a writer policy or certify arbitrary installed serializers; strict fixture qualification fails when required version-keyed evidence is absent. Broader chemistry, remote CI observation, and workload-driven cost attribution remain open. |
| Private native Python bridge probe | Implemented and independently reviewed | Private `_core` classes expose fixed atom/bond preparation, accepted state, and owned successor choices. Five bridge tests and both Rust test packages pass locally. Direct wheel and sdist-rebuilt wheel installations each passed those five bridge tests plus 15 public smoke tests, with imports verified from site-packages. This validates the private bridge packaging path, not molecular preparation, text-prefix decoding, or the complete molecular path. |
| Native directional anchoring and preparation | Bounded C/O directional transport and prospective anchor independently audited | Python preparation transports configured double bonds as typed immutable native facts with endpoint/carrier bond IDs and E/Z phase; it rejects unsupported or ambiguous carrier references, and the native wire requires integer IDs plus an actual boolean phase. The native writer installs eligible carrier requirements as CSP constraints before token publication; visitation and candidate ordering are unchanged, and generic symmetry remains intact. The anchor scope requires one carrier per endpoint, distinct endpoints and carrier bonds across relations, and acyclic directional double/carrier bonds; outside that scope the writer reports `Incomplete`. Bounded E-butene, Z-butene, and mixed alkene/tetrahedral grouped-prefix sets match the pinned fixture/source rows; these exact comparisons remain separate from semantic parse-back. The pinned triene writer quirk remains: a witness can assign direction to its otherwise unspecified middle alkene. Integrated checks passed 38 focused Python tests, six pinned C/O/stereo evidence tests, 243 South Star 2 Rust tests, and 86 root Rust tests. No broader RDKit parity or unspecified-stereo preservation is claimed. |
| Primitive continuation evidence | Bounded mixed-state continuation evidence integrated; broader continuation scope remains open | Independent fixture ligand-permutation, parity, and directional-phase expectations pin selected mixed immediate-successor semantic domains. Finite walks cover a selectable tetrahedral child, directional ring, and ring-coupled tetrahedral component; within these fixtures every reachable nonaccepted state has choices and the explored walks reach acceptance. A disconnected ring-coupled walk reaches the completed-first-component boundary, checks that labels reset, then exhausts the selected remaining suffix without cross-producting all first-component histories; a retained-label reuse witness is also covered. This is a bounded qualification, not a universal liveness theorem or general chemistry proof. This symbolic evidence is separate from the eleven real-molecule RDKit parity cases and proves neither general chemistry correctness nor universal liveness. |
| Native work-count evidence | Bounded test-only observer integrated and validated; bounded gate closed | Twelve opted-in records matched the frozen probe results. The shared-lineage observer reports choices, projector visits, exact-search/factor work, writer-frontier counts, and copy-on-write index-node/value-page copies; explicit reset separates each measurement, all 11 choice-batch records report zero index-node copies, while the twelfth record verifies rejected-candidate work remains observable. Five fixed CCC transitions (bare, disconnected spectators, inactive stereo, and connected extensions) each return one successor with zero projector visits and exact runs; they record eight traversal value-page copies and no solver-page copies. The triangle seed performs 12 edge/atom/second-edge visits in each of the first-edge, atom, and second-edge passes, unchanged by disconnected plain or inactive tetra material; each pass has 36 visits for a connected tail and 52 visits with a distant inactive tetra, plus eight layout checks. Adjacent-center activation narrows domains `[24, 8]` to `[1, 1]` as active families change `[false, true]`→`[true, true]`, with two local factor revisions and no exact-search run; the shared bond was already singleton. A rejected forced-cycle candidate records three edge visits and no successor. Copy-page counts are selective instrumentation, not all Rust allocations; quotient internals are excluded. These bounded observations justify no optimization and establish no universal complexity, spectator-independence, exact-search scaling, or allocation claim. |
| Semantics and cost closure | Bounded semantic and cost evidence recorded; broader closure pending | Keep parse-back, exact parity, and measurement separate. The opt-in probe reports source-prepared writer timing and choice counts for 15 cases with five timing repeats; `timing-final.json` is authoritative and supersedes the earlier `timing.json`. It separates full-walk totals, fixed CCC transitions, and component completion: for a fixed CCC transition with 0–4 ethane spectators, the 0→1 interior medians are 2.525, 2.715, 2.715, 3.006, and 3.065 μs; the 1→2 transition completes CCC and is reported separately. Seeded writer witnesses sample accepted paths and do not prove complete support. The separate memory pass uses one fresh process per case: `tracemalloc` covers Python-tracked heap, while GNU time reports whole-process peak RSS; neither isolates Rust allocation. These descriptive measurements establish neither constant-time behavior nor zero spectator effect. The bounded native work-count gate is recorded above; broader chemistry, shared directional rings, new RDKit versions, universal liveness, exact-search scaling, and native allocation remain follow-ups. |
| Historical native work-count artifact qualification | Previously passed for the native work-count candidate; not the current public-prefix candidate | The earlier direct and sdist-rebuilt wheels each reported 101 installed-package tests, with one identified opt-in South Star 1 diagnostic skip. The installed runner selected smoke, public-prefix, exact public invariants, pinned parity, and South Star 2 exact support. That candidate’s private source suite passed 38 tests and its Rust workspace tests passed (root 86, South Star 2 246 with one ignored); these are historical results, not the current artifact run. |
| Historical public-prefix artifact qualification | Results apply to the earlier fixed-2026.03.1-fixture consumer; not the current version-selected fixture gates | Fresh direct and sdist-rebuilt wheels built from candidate `e9bcbaa1` had identical 162-entry non-RECORD/non-SBOM payloads and `_core` SHA256 `8a26e652ad7a2efd26ea3bd00b0d009fcfec66c089d3bd291d8cce759ed1f24c`. Under RDKit 2026.03.1, each ran the then-current full installed-package runner: 102 tests, with one identified opt-in `WriterDefaultStereoAuditSlowTest` skip (`SOUTH_STAR1_RUN_SLOW=1`). The old consumer did not have the 2026.03.4 South Star 2 fixtures, so its strict exact-support gate failed closed on that version; this was missing qualification evidence, not a demonstrated runtime incompatibility. A rolling 2026.09.1 run established public API compatibility only. These are historical results and do not qualify the new version-selected fixture workflow. No remote CI was run. |

The native interface preserves typed spelling and stereo inputs and publishes
text with its validated successor. Solver and traversal stores remain private.
The scoped end-to-end implementation and bounded qualification gates are
complete for the declared corpus and local runtime checks, subject to final
independent note review and commit. Exact writer support remains version-keyed;
the South Star 2 consumers now select fixtures for the installed version and
fail when either required fixture family is absent. The bounded 2026.03.4
fixture and source evidence is captured separately from runtime/API success. Fresh
direct-wheel and sdist-rebuilt-wheel artifacts had identical 162-entry payloads
(excluding RECORD/SBOM; payload digest `31b04c1cbf3f736aed677aa253211530368041533872298a8ee333b751eb7a51`) and identical `_core` SHA256
`8a26e652ad7a2efd26ea3bd00b0d009fcfec66c089d3bd291d8cce759ed1f24c`. Each
passed the full installed-package aggregate on RDKit 2026.03.1: 104 tests total,
including one named opt-in South Star 1 slow diagnostic skip. After force-reinstalling
RDKit 2026.03.4, each passed dependency checks and installed-package provenance;
the four selected workflow modules ran 34 tests and the pinned stereo-evidence
runner ran four more, all without skips. The independent full upstream-source
contract test passed once. These are local artifact qualifications, not remote
CI results. The scoped version-selected implementation and declared local
qualification gates are complete; broader chemistry, shared directional rings,
newer RDKit versions, universal liveness, exact-search scaling, and native
allocation remain followups.
The 2026.09.1 result above qualifies only public API behavior. Broader chemistry,
shared directional rings, newer RDKit versions, universal liveness, exact-search
scaling, and native allocation remain followups. Sequential code-beauty,
error-contract, and anti-slop reviews found no actionable issues after the full
pass stabilized. Formatting and diff checks pass. Ordinary Clippy completes,
but a strict `-D warnings` run reports existing implementation diagnostics
(dead code and four Clippy lints outside the new interface); those pre-existing
diagnostics remain.
