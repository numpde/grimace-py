# South Star 2 end-to-end roadmap

## 2026-10-12 four-bond tetrahedral carbon admission

Generic preparation now admits tetrahedral carbon with four graph-bond ligands
and no non-graph hydrogen alongside the existing three-bond, one-hydrogen
shape. It transports the actual RDKit incident-bond order as the four-ligand
reference; the native order and parity model already supports this shape.
Selected opposite enantiomers are checked as semantic prefix witnesses in
`GENERAL` and `COMPACT`. The version-qualified RDKit writer language retains
its prior one-hydrogen tetrahedral admission, so these semantic witnesses do
not extend exact RDKit writer support.

Root qualification at `7ef6b938` used the unchanged compiled extension (SHA-256
`aa7ffeded8af04d330f71ad4e0d5a50bf1224ce6a71bd00a6c27dd5c45b1b33b`).
The bounded preparation, preparation-semantics, standard-SMILES, Kekule,
compact-language, and public-prefix modules passed 76 tests in 29.311 seconds;
the runner reported `verified_no_live_members` and a sampled process-tree peak
of 199.75 MiB. The alternate ligand-order witnesses test two enantiomers each
for a four-halogen center and a ring center, in both generic languages, with
reverse atom renumbering and wrong-marker rejection. This is finite semantic
coverage, not an exhaustive language or RDKit writer-support result.

## Tetrahedral Forest candidate plan

Admit tetrahedral facts to generic Forest when no directional relation is present. Preserve the Tree role rule; in Forest, each residual chemical group can contain several traversal incidences, while the actual per-bond traversal/ring decisions and topology factor own role consistency. Keep the entry bond as the tetrahedral entry ligand and do not treat a dot or serialization predecessor as a ligand. Qualify the candidate with bounded native tests for real chiral molecules, opposite parity, sibling incidences, completed support, and rejected premature acceptance. This slice does not change E/Z Forest admission or claim general writer parity.

Candidate evidence: the isolated offline wheel exercised version-keyed semantic witnesses in `GENERAL` and `COMPACT` for both halogenated enantiomers, a ring center with two traversal children in one chemical residual component, and adjacent tetrahedral centers. Each selected path reached acceptance, parsed back with the source chirality, and rejected the opposite marker. The earlier exhaustive tetrahedral semantic loops now explicitly cover the no-dot tree-like sublanguage; their collectors fail with a named budget if this scope grows. These checks do not validate leading-dot tetrahedral branches, all Forest paths, or RDKit writer support for the dot spellings.

The bounded candidate Rust library suite passed 293 tests with one ignored. After preserving the existing unrestricted `CC(O)CO` C/O dot witness with its measured case-specific 1.5-million-state and 250,000-output budgets, the selected five public Python modules passed 53 tests in 30.9 seconds against the candidate release extension; the runner reported verified cleanup. The larger C/O budget does not apply to the no-dot stereo loops. This is selected semantic and regression qualification, not exhaustive Forest support or liveness proof.

After copying the five reviewed source files and candidate extension into the root checkout, bounded offline root `cargo test --lib` passed 86 tests, and the same selected five Python modules passed 53 tests against that extension. Both root runs reported verified process-tree cleanup. The root Rust count is a distinct suite from the candidate South Star 2 Rust count above.

## Generic Forest sibling traversal slice

Historical generic-sibling problem, resolved before `599bb352`: the generic Forest writer rejected `C(CC1)CCC1` for the same cyclohexane graph that accepted `C1CCCCC1`: one chemical residual component had several parent incidences, but the frontier treated that grouping as a one-entry traversal limit. The generic sibling slice kept the residual partition as the chemical grouping owner. In Forest, it offered each enterable incidence as a branch and allowed ring openings alongside it; inline became available only after the chosen incidence consumed the parent's last unresolved incidence. A branch start fixed all remaining parent incidences to Traversal in the same solver transition, because parent ring spelling had ended. Restoring a parent recomputed its attachments from the live residual partition after the child could split that component. Tree and stereo admission remained as they were at that baseline. The earlier separate tetrahedral Forest draft was not integrated; the candidate described above is a new slice built on the verified generic frontier.

Branch admission also checks a local entry conflict: another suspended parent may already have a Traversal-only bond into the same unvisited atom. A nested dot branch that would reserve a second entry is immediately rejected; the source prefix can still open a ring before branching. This is a necessary immediate constraint, not a suffix search or a general completeness proof.

Root checkout qualification for this slice: the candidate wheel `grimace_py-0.1.10-cp312-cp312-manylinux_2_34_x86_64.whl` has SHA-256 `b40264c54557fd12490e66ed0513a33f24d4e9757a3f8029fae77611af24bcad`; its sole `grimace/_core` extension was extracted into the ignored local Python package path and has SHA-256 `411cc429eb9461c52d4ffd55ec7d21f5e5e131d44035098d05a069f566ac6c2a`. Bounded offline root `cargo test --lib --offline` passed 86 tests. Bounded root preparation, compact, Kekulé, standard SMILES, and public-prefix Python suites passed 52 tests against that extension and RDKit 2026.03.4. Both runners verified process-tree cleanup. These checks establish the represented paths and fixtures, not universal Forest grammar completeness. The tetrahedral Forest draft was not integrated, and mixed aromatic spelling remains deferred until its atom facts and hydrogen/valence semantics are defined.

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

## Atom spelling and the standards-language base

The next public API direction keeps one native atom-spelling state machine.
Python snapshots intrinsic atom facts from RDKit; Rust derives spelling
context from the prepared graph and owns lexical choices alongside traversal.
The default public prefix language is the implemented OpenSMILES subset, not a
claim of complete grammar or chemistry coverage. The optional
`rdkit_random_noncanonical` style applies source-backed restrictions to that
same spelling state, preserving the separately qualified writer-parity lane.
It does not select or certify the installed RDKit serializer.

Tetrahedral parity remains a native graph constraint; `@` versus `@@` is a
selected spelling context, not a duplicate atom string supplied by Python.
Atom labels are generated by the native grammar. Query atoms and custom
RDKit-only labels remain outside the preparation boundary. Aromatic admission
must use concrete graph facts and explicit eligibility, not a universal
half-bond valence formula. Leading-zero numeric spellings can create
unbounded default-language paths, so exhaustive finite fixture walks must use
the named writer style or a deliberately bounded prefix check.

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

## Published-package support delta investigation

The opt-in probe `tmp/exploration/south_star2/003_compare_published_support_delta.py`
compares this source revision's public `grimace.decoding.start(mol)` prefix
walk with the actual PyPI `grimace-py` 0.1.18 wheel
(SHA256 `14589d18295e418b5570566953250b5b039a4c20c79ef7815eb221e98ff239ee`)
under RDKit 2026.03.4. It verifies the extracted wheel members and extension
hash, and checks the installed current core and preparation hashes. Rerun it
with the pinned 2026.03.4 environment and the same wheel and extracted root;
the JSON report is an external run artifact, not a checked-in expected set.
Each lane/case runs in a fresh subprocess with a three-second timeout and 2 GiB
address-space limit. The prefix walk has a 20,000-state cap. Outcomes and
failures are retained by case. Seeded RDKit strings are positive membership
witnesses only. The published package's enumerated strings are its observed
baseline, not independently source-proven complete RDKit support. The walker
records acceptance and still visits available choices; a small accepted-and-
expandable state check guards this boundary. The latest run had no worker
timeouts or prefix-cap exhaustion.

The 45-case matrix combines C/O controls, ordinary unsupported feature probes,
and the nine curated examples in the published package's former timing corpus.
All nine published decoders initialize; current South Star 2 accepts only
`C1COCCC12CO2` from that group. The other eight fail preparation because of
aromatic bonds, unsupported elements, or directional carrier constraints.
Across the matrix, 19 small cases completed both exact-set comparisons: 14
sets agree and five cyclic unsaturated inputs have strict SS2 supersets. The
published package has no output missing from SS2 in those five. For
`C1=CCCCC1`, for example, SS2 additionally emits `C=1CCCCC1` and
`C=1CCCCC=1`; the other extra spellings are recorded in the run JSON. This is
a standard-valid spelling delta: OpenSMILES §3.4 lists these cyclohexene
ring-bond forms. Daylight §§3.2 and 3.3 also define explicit hydrogen atoms
and directional-bond spelling for partial double-bond stereo. These are not
invalid SMILES or a chemical defect. They are outside the published package's
observed writer style, but are members of the intended standard-defined output
language. The five cyclic cases are `C1=CCCCC1`
(6 published, 8 SS2 strings), `C1=CCCC1` (5, 7), `C1=CCCCCC1` (7, 9),
`C1=CC=CCC1` (6, 10), and `C1=CCCCC1O` (24, 32).

The separate configured E/?/E triene case has seven outputs in each lane, with
equal string sets. Mapped bond facts show that every output preserves the two
configured outer E bonds; four outputs assign E and three assign Z to the
otherwise unspecified middle bond. A seeded writer witness demonstrates each
middle assignment. An explicit-H spelling parses with E/?/E when hydrogens are
retained, then `Chem.RemoveHs` yields the same heavy-atom graph and mapped
stereo facts. RDKit's default parser removes those hydrogens during parsing and
assigns the middle bond, so this observation is scoped to parse-then-normalize
order; it is not a general claim that default parsing preserves the unset
stereo. The observed outputs show a per-string difference in whether the
unspecified center receives a directional assignment; its collective meaning
under the intended language and hydrogen-normalization rules remains an open
semantic question. This is not evidence that default parsing is defective or
that the center is generally unrepresentable. See [OpenSMILES
§3.4](https://opensmiles.org/opensmiles.html) and [Daylight SMILES theory
§§3.2–3.3](https://www.daylight.com/dayhtml/doc/theory/theory.smiles.html).

The bounded matrix also shows where the current preparation boundary lies:
neutral saturated and unsaturated C/O controls, tetrahedral C/O cases, simple
C/C and C/O directional cases, and selected shared-double/ring cases prepare;
N/S/P/B/halogens, charges, explicit hydrogen state, isotopes, atom maps,
radicals, aromatic bonds, and triple bonds are rejected. A short adjacent
directional-chain request reaches native `NotImplementedError` for the
currently declared anchor scope. These are observed admission facts for this
candidate and RDKit version, not a complete chemistry classification.

The intended South Star 2 end state is to generate the standard-defined valid
SMILES representations for the intended molecule first, with any requested
style applied as a separate optional subset. The ring-open-only,
ring-close-only, and matching-both-endpoint spellings therefore belong to the
core language when allowed by the relevant standard; a closure-only preference
is a style filter, not a universal correction. Published/RDKit output is useful
comparative evidence for a style subset, not the authority for the core
language. The current runtime remains incomplete and entangles some writer
style with generation; this investigation does not implement or claim that
target. Moving the public contract to this target requires an explicit API and
documentation transition, not silently treating parsed-object equivalence as
exact writer parity.

The native core must model the applicable standard grammar, not enumerate a
finite glyph list and call it complete. OpenSMILES v1.0 (2016-05-15) §3.1.4
explicitly equates the isotope spellings `[2H]`, `[02H]`, and `[002H]`; §3.1.7
defines atom classes as arbitrary integers, defaults an omitted class to zero,
and gives `:005` as a numeric interpretation example. Section 3.10 discusses
minimum parser width for atom-class fields; it is not ring-label syntax.
OpenSMILES §3.4 limits ring tokens to one digit or `%` followed by exactly two
digits, matched numerically, so leading-zero ring aliases are finite. The
document does not give a separate formal `NUMBER ::= DIGIT+` production;
unbounded zero-prefixed aliases for isotope and atom-class fields follow by
interpreting their values numerically, not by quoting a grammar production.
As a concrete parser observation, RDKit 2026.03.1 and 2026.03.4 accept `[2H]`
but reject `[02H]` and `[002H]`, despite those explicit OpenSMILES examples.
Parser acceptance is comparative runtime evidence, not authority for the
intended standard language. This supports an incremental lexical-state design
with finite next-character choices and arbitrarily long valid numeric fields,
without whole-language enumeration. Evidence and runtime resource limits cap
an execution or fixture walk; they do not cap the language definition. Ring
bond labels also need pair-consistent alternatives, including a mark at either
endpoint or matching marks at both. See [OpenSMILES v1.0,
§§3.1.4, 3.1.7, 3.4 and 3.10](https://raw.githubusercontent.com/opensmiles/OpenSMILES/master/opensmiles.asciidoc).

The grammar has structural serialization choices beyond graph traversal over
connected components. OpenSMILES §3.7.1 documents dot-separated forms such as
`CC` and `C1.C1` for the same molecule, and `CCC` and `C1.C12.C2`; dot segments
therefore cannot be assumed to represent disconnected graph components.
Standard-defined serialization events must be modeled independently from the
current component-first/traversal schedule. The first substantial native step
should establish incremental grammar state and those event transitions, then
allow an optional style predicate to intersect the valid paths. It must not
assume one fixed glyph renderer or equate graph components with dot segments.

The target language still needs a precise, reviewed account of graph identity
and equivalence across hydrogen forms, aromatic and Kekule forms, charge and
atom-class fields, maps, isotopes, and partial stereo. These are standards and
semantic questions, not fields to silently normalize using RDKit's default
parser. In particular, the triene observations above are per-string facts;
collective equivalence under hydrogen normalization remains unresolved. The
pinned RDKit 2026.03.4 source manifest at
`tests/fixtures/rdkit_upstream_serializer_sources/2026.03.4/manifest.json`
supplies provenance for comparative `Canon.cpp` ring-stack traversal (lines
991–1022) and `SmilesWrite.cpp` stack-bond emission (lines 430–458), but these
sources do not define the target core language. See [OpenSMILES
§§2.2, 3.1.4, 3.1.7, 3.4, 3.7.1](https://opensmiles.org/opensmiles.html) and
[Daylight SMILES theory §§3.2–3.3](https://www.daylight.com/dayhtml/doc/theory/theory.smiles.html).
The source-to-native design and the standard/language boundary require review
before behavior changes.

## Candidate integration plan for a standards-language public base

The selected next public shape is
`grimace.decoding.start(mol, *, style=None)`. The default explores the
standard-language choices implemented by this release;
`style="rdkit_random_noncanonical"` restricts the same grammar and writer
state transitions to the named RDKit-style path. This is an incremental subset,
not a claim of complete SMILES support. Style is a single named choice, not a
registry or a second serializer. RDKit exact-support fixtures and parity gates
remain attached to the explicit style path. Existing exhaustive fixture walks
must therefore pass that style explicitly; the standard base has infinite
numeric aliases and cannot be validated by unbounded support enumeration.

The ownership chain should be one-way: Python snapshots RDKit molecular facts
into immutable records; Rust consumes those facts and owns atom spelling,
bond spelling, graph traversal, and online lexical state; the optional style
restriction filters transitions in that same writer model. The new native
`AtomFacts`/`AtomSpellingContext` boundary owns element, aromaticity, optional
isotope, charge, non-graph hydrogen count, graph bond-order sum, and normalized
atom-class value. Tetrahedral text markers are context chosen by traversal, not
chemical facts. Python must reject query atoms and custom `smilesSymbol` or
`_supplementalSmilesLabel` values until a supported meaning is defined. It
must not ask RDKit's SMILES writer to provide a fallback token.

Preparation must keep graph hydrogen atoms distinct from bracket H counts and
preserve the source facts needed to interpret explicit and implicit hydrogen
forms. Isotope absence is distinct from isotope zero. Atom-class absence has
numeric value zero for the core language, so omitted and explicit zero forms
are both candidates; source annotation presence is relevant only to a style
restriction and must travel in style context alongside, not inside, chemical
atom facts. Charge, atom maps/classes, radical state, and aromaticity must be
preserved or rejected at the snapshot boundary, never silently discarded.
Ordinary valence checks can use explicit graph bond facts and normalized
hydrogen counts. Aromatic bond values such as 1.5 are representational
bookkeeping, not a general valence formula for fused rings or heteroaromatics;
aromatic atom eligibility and Kekule alternatives need dedicated semantic
facts and tests rather than arithmetic over `GetBondTypeAsDouble()`.

The Rust atom-spelling state emits finite nonempty lexical pieces and retains
prefix state until one atom spelling is complete. The molecule writer must
carry that state through ordinary choices without previewing suffixes or
enumerating support. Text selection continues to retain every successor with
the selected piece. OpenSMILES permits omitted or explicit single-bond marks,
single/double/triple/aromatic bond forms where chemically applicable, and ring
bond spelling at either endpoint or matching marks at both endpoints. Dot
separated traversal segments can also encode connected graphs; they are not
limited to graph components. These topology and lexical alternatives belong
to the standard writer model, while RDKit endpoint placement and component
ordering belong to its named style restriction.

Integration can proceed in reviewable units without preserving duplicate atom
text authorities:

1. Route current C/O and tetrahedral input through the fact-based spelling
   state and PyO3 bridge. Preserve the existing exact sets only under the
   explicit RDKit style, with stereo marker selection in context rather than
   fixed Python parity strings.
2. Expand the Python snapshot and native fact consumers for ordinary elements,
   triple bonds, and bracketed charge, isotope, hydrogen, and atom-class forms.
   Keep source-valid leading-zero isotope/class cases in grammar tests even
   when RDKit rejects a spelling; normative fixtures own those expectations.
3. Add aromatic/Kekule representation and connected dot/ring serialization
   as separately tested semantic behavior, including fused and heteroaromatic
   valence controls. Do not report the default base as complete outside its
   tested grammar subset.

For each unit, semantic parse-back and graph/H/stereo facts remain separate
from exact RDKit-style fixture equality. Test default choices with bounded
prefix walks and grammar witnesses; test exact fixture support only with the
explicit RDKit style. Liveness claims remain bounded to explored prefixes.
Normal choices must be native incremental transitions and must not invoke
RDKit, a prior enumerator, a suffix search, or asset construction. Rebuild and
exercise both wheel and sdist artifacts after the native bridge is complete.

The comparison is intentionally small and version-specific. It does not
establish full RDKit support equality, general SMILES validity, broad chemical
coverage, or performance. Exact enumerations are limited to the 19 completed
small cases and one separate triene; the larger curated and feature cases are
admission or single-witness observations. The two package lanes share the
same installed RDKit version, but the published and current implementations
remain distinct artifacts. This investigation records the intended language
direction; it does not change the implementation or silently alter the current
public contract.


## Candidate implementation progress (2026-10-12)

The N3 prerequisite is committed as `a46881b5`. The pending candidate now
routes ordinary atom and bond choices through native incremental transitions:
ordinary elements, charges, supported aromatic forms, triple bonds, isotopes,
and atom classes are covered within the tested subset. This remains a subset
of the standard language. Generic ring labels admit independent spellings in
the full 110-spelling domain for labels 0 through 99; the optional
`rdkit_random_noncanonical` restriction remains distinct from those choices.

The final native candidate passed 259 tests with one ignored test; focused
assertion and directional-oracle checks then passed one and six tests,
respectively. A fresh offline sdist-to-wheel build from the frozen V2 exact
substantive Python, Rust, and test bytes qualified the artifact; maturin added
`Cargo.package.readme` metadata, and two empty `.gitkeep` files were omitted.
The rebuilt wheel installed in ordinary RDKit 2026.03.1 and 2026.03.4
environments, with imports proven. The 2026.03.1 lane ran 138 tests, with one skipped and all others passing,
in 121.408 seconds; the 2026.03.4 selected scope passed 46 tests
in 0.114 seconds and does not qualify broader parity. Every cited bounded run
verified cleanup. The wheel SHA-256 is
`1804bf2c5abe7fdfc6dff69f43dc8c48dd740e261b18b892aa9ee927fe82d30d`.
Production runtime bytes were unchanged from the previous V1 direct-wheel
qualification; V2 changed four test files and this note. Final root checks
passed: native 259 tests (one ignored), Rust 86 tests, and checkout Python 36
tests using a freshly rebuilt ignored extension. All bounded runs verified
cleanup.
Logs: `/tmp/grimace-final-artifact-qualifier-20261012/rebuilt-31-final.log`
and `/tmp/grimace-final-artifact-qualifier-20261012/rebuilt-34-final.log`.

This milestone implements a standards-derived chemistry subset with an
optional RDKit writer style, not complete standard-language or stereo support.
Choices return incremental lexical pieces. Ring-label interior prefixes are
retained by the lexical state; the public choices API does not thereby expose
every character-interior prefix. Grammar-only completion does not establish
that every reachable prefix has a full molecular suffix. No ordinary
`NativeSolver` counterexample has been established, and a general liveness or
full-suffix theorem remains unproved. Kekule alternatives, dot-connected ring
topology, and remaining stereo restrictions still need separately scoped
semantic qualification.


## 2026-10-12 finite support-growth diagnostic

An isolated native diagnostic at `/tmp/grimace-support-growth-071a34e5`
walked exact finite supports for fact-correct C/O graphs. These families use
elided single bonds, connected graphs, no stereo, and either bare atoms or one
fact-correct bracket alternative per atom; isotope and class numerics are
excluded. They are not full Generic support (leading zeros make that infinite),
a style API, or runtime caps. Counts are observed finite walks, not a universal
independence claim.

| Graph | Atoms | Ring lexemes | Unique SMILES | Accepting paths |
| --- | --- | --- | ---: | ---: |
| CO | bare | 1 | 2 | 2 |
| CO | brackets | 1 | 8 | 8 |
| CCO | bare | 1 | 4 | 4 |
| CCO | brackets | 1 | 32 | 32 |
| CC(C)O | bare | 1 | 6 | 12 |
| CC(C)O | brackets | 1 | 96 | 192 |
| C1CO1 | bare | 1 | 3 | 6 |
| C1CO1 | brackets | 1 | 24 | 48 |
| C1CO1 | bare | 4 | 24 | 48 |
| C1CO1 | brackets | 4 | 192 | 384 |
| C1CO1 | bare | 110 | 390 | 780 |
| C1CO1 | brackets | 110 | 3120 | 6240 |

Ring families are canonical `1`, `{0,1,%00,%01}`, and 110 lexemes for numeric
IDs 0–99. Opening and closing use the same numeric ID: the latter families
admit 8 and 130 pairs, not 16 and 12100. Branch/ring paths can outnumber strings
twofold; this does not prove equal full futures. CCO prefix `C` retains two
states with different next texts (`C` versus `(`), demonstrating necessary
state retention.

Ring-1 bare had 702 pre-group emitted successors and 13 admitted text groups;
ring-110 brackets had 41906 pre-group emitted successors and 13909 admitted
text groups, with 28697 cumulative retained states, maximum frontier 4,
and 13910 token-prefix nodes. Pre-group choices include valid Generic options
excluded by this finite witness; they are not production waste. These are
independent choice-call bookkeeping counts, neither raw structural candidates
or solver calls nor runtime timings.

Checks cover independent CO bare/all-eight-bracket, CCO, and ring expectations,
projection, and group conservation. Budgets of 100000 prefixes, 2000000 states,
and depth 128 fail rather than truncate. Native tests passed 260 with one
ignored before the last assertion; the final focused test passed with verified
cleanup. Final root native verification passed 260 tests with one ignored and
verified cleanup; the twelve matrix rows matched. The final log is
`/tmp/grimace-support-growth-final-root.log`; diagnostic results are retained
at `/tmp/grimace-support-growth-results.txt`. Reproduce from the repository checkout:

```sh
python3 scripts/run_bounded_tests.py --wall-seconds 120 --rss-mib 2048 --cpu-seconds 120 --address-space-mib 4096 -- cargo test --offline -p grimace-south-star2 bounded_support_growth -- --nocapture
```

Decision: the next narrow investigation is aromatic/Kekule alternatives;
published 0.1.10 benzene `kekuleSmiles=True` was confirmed locally. No measured
bottleneck justifies new compact policy. Do not start that investigation or a
refactor as part of this diagnostic.

## 2026-10-12 planned aromatic/Kekule contract

Generic semantics come before style qualification. For qualified, currently
admitted aromatic facts, support should include all feasible integer bond-order
assignments preserving source valence, H, charge, connectivity, isotope, class,
and stereo, while retaining source-aromatic witnesses. Each aromatic component
chooses Aromatic or Kekule representation as a whole; this is not a claim of
all mixed aromatic/uppercase SMILES coverage. Implementation and qualification
remain planned.

Python transports cached `GetTotalValence()` as an optional aromatic-atom fact
appended to the strict ten-field tuple; nonaromatic atoms carry `None`. Source
chemical aromatic flags remain immutable. Rust derives
`double_demand = source_integer_total_valence - nongraph_H - fixed_nonaromatic_graph_order_sum - aromatic_degree`;
graph H belongs in the fixed sum. Supported roles are demands zero or one;
other demands fail descriptively, without invented roles or silent defaults.

The existing `ConstraintModel` owns an exclusive aromatic matching primitive,
component modes, and separate chemical edge-order domains
`{Aromatic, Single, Double}`. Factor-variable exclusivity is checked. General
nonbipartite matching feasibility and edge projection bypass generic exact
assignment search for this factor. Evolving domains live only in the existing
`PagedStore`, with no second writer or constraint store. Atom notation context
is separate from source aromatic facts; both ring ends constrain the same
chemical edge variable. Existing placement, direction, and stereo variables
retain their responsibilities.

Reject runtime RDKit Kekulization/writer fallback, precomputed assignment
catalogs or multiple RDKit-prepared graphs, mutable aromatic flags, duplicate
element/valence role tables, bipartite-only matching, and local valence checks
without global feasibility.

Qualification must cover both benzene matchings; fused naphthalene and
nonbipartite azulene; pyridine, pyrrole, N-substituted pyrrole, furan, thiophene;
charged N/C and exocyclic C=O; source-aromatic wildcard/P/As/Se/B cases;
isotope/class/H preservation; ring double-bond placements; independent
component modes; infeasible partial commitments; and mode/order correlations.
Semantic parse-back checks stay separate from exact style support. The RDKit
style currently rejects aromatic input; preserve that narrow admission until
pinned style fixtures qualify published Kekule parity.


## 2026-10-12 aromatic/Kekule implementation progress

The planned component contract is now implemented in the native runtime:
source aromatic facts remain immutable, while a general matching factor
projects Aromatic/Single/Double edge domains through the existing domain store.
Default `style=None` offers source-valence-preserving Kekule assignments and
aromatic spelling per connected aromatic-bond component, including independent
component choices. Atom notation uses separate representation context; joint
order/placement commitments and both ring ends share the same edge order,
with global residual feasibility. There is no runtime assignment catalog,
RDKit writer/Kekulize call, or new public `kekuleSmiles` option. This component
contract does not cover all mixed lowercase/uppercase valid SMILES.

A later preparation invariant requires every admitted source-aromatic component
to have at least one assignment of its 0/1 double demands to its source
aromatic edges. Source lowercase spelling still emits aromatic bonds without
choosing a particular integer assignment, but the component must have a
latent assignment consistent with its fixed nonaromatic bond orders and
non-graph hydrogens. Otherwise Source spelling could admit a ring whose atom
valences cannot be satisfied, such as five neutral aromatic carbons each with
one hydrogen and total valence four. A component with all zero demands needs
no selected double edges. This is an integer-valence consistency check for
the currently modeled aromatic facts; it does not establish aromaticity or
broader mixed-representation support.

Qualification after integration at `57778418`: the focused native aromatic
suite passed 12 tests, and the full South Star 2 Rust suite passed 297 with
one ignored before the final even-demand assertion, which passed in the
focused suite. Root Rust passed 86 tests. The six selected public Python
modules passed 78 tests against the freshly compiled local `_core` extension
(SHA-256 `e754167e0103002c1a7ee6c6c8d0e7e860621c5de1b619a324528a6efe3922ca`).
The two integrated root runners verified process-tree cleanup; logs are
`/tmp/grimace-aromatic-feasibility-root-integrated-rust.log` and
`/tmp/grimace-aromatic-feasibility-root-integrated-python.log`.

Preparation transports cached source total valence and normalizes flagged
aromatic single/double bonds after caller Kekulization with flags retained;
contradictory flags, endpoints, or marks fail. Qualification resolved source
carbonyl valence, normative bare B/P/wildcard spelling, and directional-carrier
ownership; source H1 N/P stays bracketed, and shared `*` retains both modes.
Existing unsupported input/stereo limits remain, including explicit rejection
of chemical directional marks on aromatic bonds. The optional RDKit style
still rejects aromatic inputs; semantic checks do not qualify Kekule writer
parity or prove general prefix full-suffix liveness.

Native verification passed 279 tests with one ignored. A fresh wheel passed
39 selected methods on each of RDKit 2026.03.1 and 2026.03.4 (13 public,
21 preparation, five Kekule methods). All semantic parse-back checks passed
for 22 fixture cases and 26 integer/mixed-component witnesses, including source
H, isotope/class/charge, adjacent tetrahedral/E/Z stereo, and exograph H.
Direct wheel bytes and import provenance were verified; bounded cleanup was
verified. Evidence is under
`/tmp/grimace-kekule-artifacts-24dfbfdb/direct-v2-source-manifest.json`.
Fresh sdist qualification is complete: a wheel rebuilt from the extracted
sdist with a fresh target produced the same core bytes as the direct wheel.
Ordinary site-packages installs matched source provenance and current RDKit
versions. The rebuilt artifact passed the full installed suite on RDKit
2026.03.1 (147 tests, one skipped, 121.132 seconds) and 55 selected methods on
2026.03.4 (all passed, 0.365 seconds); the latter is selected coverage, not a
full-suite or Kekule writer-parity claim. Install/test budgets were 300 seconds
for the full suite and 120 for selected coverage, with verified cleanup.
Artifact hashes and provenance are recorded in
`/tmp/grimace-kekule-artifacts-24dfbfdb/sdist-wheel-provenance-v1.json`;
`rebuilt-31-final.log` and `rebuilt-34-final.log` in that directory hold results.
The next scoped investigation remains connected-dot/ring topology.

## 2026-10-12 compact language implementation and validation

The public entry point becomes
`start(mol, language=SmilesLanguage.GENERAL)`, with enum members `GENERAL`,
`COMPACT`, and `RDKIT_RANDOM_NONCANONICAL`. Strict enum input replaces `style`;
there is no compatibility alias. `GENERAL` keeps the current default's broad
supported semantic subset. The RDKit member retains its separate, complete
qualified writer contract and existing admission limits; semantic equivalence
does not establish exact RDKit support.

Rust owns language rules. Python prepares source facts and transports the
selected enum name. `COMPACT` is generated directly by native grammar, state,
and constraints, without completed-string filtering or a new suffix solver.
It retains traversal, rooting, stereo choices, aromatic representation, and
whole-component Kekule choices while restricting spelling as follows:

- Use a bare atom when source facts permit it; otherwise use brackets.
  Preserve isotope and atom-map metadata, including native isotope `Some(0)`
  and an explicitly present source map property zero as `:0`. Public RDKit
  isotope zero has already been normalized to absence during preparation.
- Render decimal fields without padding. Omit H0, spell H1 as `H`, and spell
  larger hydrogen counts as `Hn`. Omit neutral charge; spell magnitude one as
  `+` or `-`, and larger magnitudes as a sign followed by decimal magnitude.
  Tetrahedral markers are only `@` and `@@`.
- Omit a bond symbol exactly when the endpoint interpretation yields the
  intended bond, using selected emitted atom categories and representation
  modes rather than source aromatic flags alone. For source aromatic edges
  incident to `*`, emit an explicit `:`; omit a fixed single wildcard bridge
  when its endpoint interpretation is single. `GENERAL` may admit a sanitized
  omission in some cases; that is a supported writer behavior, not a
  standards mandate. For ordinary ring bonds, omit at both ends when that
  interpretation is correct; otherwise emit exactly one symbol at either
  endpoint, excluding duplicate `Ring11` spelling. Double/triple bonds need
  one symbol, and a fixed single bond between aromatic atoms needs one `-`.
  Directional ring plans and marker multiplicity remain governed by their
  existing constraints, outside ordinary-symbol minimization; this adds no
  stereo minimizer.
- Allocate the smallest free ring number in `0..99`, starting at zero. Use
  single digits for `0..9` and `%10` through `%99` for the rest; reuse a number
  after closure. Preserve closure-event order and existing ring capacity.

Compact denotes this spelling policy, not canonicalization, globally shortest
SMILES, or exact RDKit writer support. A public dataclass or orthogonal writer
and spelling axes was considered and deferred until real consumers need those
combinations: exposing them now would introduce undefined combinations. The
enum deliberately keeps qualified language contracts together. Future demand
for independently selectable policies is a reason to revisit that boundary.

The implementation keeps compact spelling inside native atom and bond
construction, rather than filtering completed choices. Preparation preserves
the public isotope-zero normalization boundary, while native internal isotope
`Some(0)` remains spellable. Review corrected several compact test witnesses
so they describe the actual source/emitted facts.

Final bounded validation passed: native South Star 2 ran 283 tests with one
ignored; the root Rust suite ran 86; the installed RDKit 2026.03.1 package suite
ran 119 with one skip; the selected RDKit 2026.03.4 installed and candidate
checkout scopes each ran 61. After integration, the root checkout's direct
2026.03.4 smoke ran 26 tests. Every bounded run exited zero with
`verified_no_live_members` cleanup. Logs are in
`/tmp/grimace-compact-artifacts-05d831bf/` (`native-south-star2.log`,
`native-root.log`, `installed-31.log`, `python-selected-installed-34.log`,
`python-selected-checkout.log`, and `final-wheel-build.log`); the integrated
root smoke log is `root-smoke.log` in that directory. The final wheel SHA256 is
`00b4f091f07c83429e9b1cbd4be8f0b43d5f1b26700b1813b1ba649bb2ab3ffd`; its
verified extension SHA256 is
`04d968bfe80f7496626b05cfad0efa8cefb09de3675d9183b208516b92416bfe`.
These checks cover the recorded finite scopes and do not claim universal
full-suffix liveness or remote CI results.

## 2026-10-12 principled language contract and next slice

The target contract now lives in
[`docs/api/smiles-language.md`](../docs/api/smiles-language.md). It separates
admitted molecular facts and explicit rejection, OpenSMILES v1.0 syntax and
semantic identity, constructive soundness and coverage, and the narrower
compact spelling policy. Its coverage table marks current structural and
stereo gaps; `GENERAL` remains a provisional name, not a completeness claim.
The next design decision is how to represent serialization segments as a
spanning forest on admitted nonaromatic, stereo-free graphs: the current
`SpanningTreeFactor` enforces `n-1` traversal edges, whereas connected-dot
forms require ring obligations across forest segments independently of
chemical connected components. This blocks a local dot-transition patch.
No implementation or universal proof is claimed by this documentation step.

Kernel checkpoint after contract commit `b01eb3f2`: a shared traversal-topology
factor now supports Tree and Forest requirements. Both reject forced traversal
cycles and remove optional traversal values that would close one; Tree also
requires connectivity and forces quotient bridges. The public writer still
builds Tree factors. An independent exhaustive role-projection test covers
small cyclic and disconnected Forest graphs. The bounded Rust library suite
reported 284 passed, 1 ignored, with verified process cleanup. This validates
the factor's role projection, not joint writer liveness or connected-dot output.

## 2026-10-12 top-level segment qualification in progress

The native forest path now supports top-level connected-dot serialization for
admitted nonaromatic, stereo-free inputs in `GENERAL` and `COMPACT`. Public
semantic witnesses include `C1.C1` for `CC`, `C1.C12.C2` for `CCC`, and compact
smallest-label equivalents. Nested branch dots and aromatic or stereo inputs
remain outside this slice. A fixed-budget native diagnostic may explicitly
report incomplete enumeration, so it is not a global exhaustive-support or
prefix-liveness result.

A candidate wheel passed the focused public compact-language file: 12 tests,
with verified process cleanup. The compact `CO` support set grew from 2 to 4
strings and the three-member ether ring set from 3 to 39. All enumerated
outputs parsed back to the source graph; enumeration used at most 10,000
prefix states and 1,000 outputs per molecule (actual 13 and 147 states).
These fixture sets record bounded candidate-runtime support, not an
independent exhaustive oracle. Native South Star 2 validation subsequently
passed 287 tests with one ignored and verified cleanup. An initial candidate
root Rust run passed 67 tests and failed 19 because that checkout lacked its
Python `_core` extension; those failures were an environment setup result.
After extracting only the verified candidate wheel extension into the root
source package, the actual root `cargo test --lib --offline` passed 86 tests,
and the root Python public smoke passed 28 tests across the compact-language
and public-prefix integration files. Both root runs used the bounded runner
and verified no live descendants. These finite results do not establish
universal support or prefix liveness.

The next forest slice admits terminal branches on admitted nonaromatic,
stereo-free inputs in `GENERAL` and `COMPACT`. Native witnesses reach accepted
`C(C)`, `C(C(C))`, `C(C)(C)`, and `C(C)C`; public witnesses check the first
three against their source graphs. Bounded compact enumeration of `C1`, `CO`,
and `C1CO1` yielded 2, 19, and 236 prefix states, respectively, with every
accepted output parsing back and the prior expected sets retained. The compact
fixture adds 23 parenthesized strings for `CO` and `C1CO1`. The qualified
candidate wheel passed the focused public compact-language file (13 tests).
That candidate did not yet cover nested branch dots or leading dots; aromatic
or stereo cases also remained open. These checks did not establish maximal
support.

The next qualification must distinguish standard syntax from RDKit parser
acceptance and RDKit writer parity. [OpenSMILES v1.0's branch grammar and
§3.7.1](https://opensmiles.org/opensmiles.html) allow `branch ::= '(' dot chain ')'`:
`C(.C)` has the graph of `C.C`. RDKit 2026.03.4 was observed to reject that
spelling as input, so its parser cannot be the sole validity check for these
outputs. The temporary test-only `_small_compact_graph` witness in
`tests/south_star2/test_compact_language.py` interprets only C/O, implicit
single bonds, single-digit ring labels, branches, and dots. It rejects other
tokens, constructs and sanitizes an RDKit graph without parsing the emitted
SMILES through RDKit, and compares canonical identity to the source. It should
eventually fold into the shared fact/event/constraint model. The witness is
neither a runtime fallback nor a full OpenSMILES validator, exhaustive support
oracle, or prefix-liveness proof. It caught the invalid `C0(.C00)1.O10`:
ring tokens after a branch violate this grammar even if graph reconstruction
would yield the intended graph. The branch phase was then corrected.

The expanded candidate qualified admitted nonaromatic, stereo-free top-level
and branch-local dots, including leading-dot branches, in `GENERAL` and
`COMPACT`. The compact fixture's bounded output counts were 1 for `C`, 8 for
`CO`, and 228 for `C1CO1`, from 2, 27, and 963 prefix states respectively
(limits: 1,000 outputs and 10,000 states per molecule). Every output passed
the grammar witness and reconstructed the source graph; the previous fixture
sets remain subsets. The candidate public compact-language file passed 14
tests; native library tests passed 289 with one ignored. These are bounded
candidate results, not global maximal-support or prefix-liveness proof. The
integrated root checkout then passed the bounded native South Star 2 library
suite (289 passed, one ignored), root Rust library tests (86 passed), and 30 selected public tests from
`tests.south_star2.test_compact_language` and
`tests.integration.test_public_prefix_api`. Both runners reported verified
process cleanup. The compiled extension used for the root public test came
from the qualified candidate wheel with SHA256
`b06e9e3144d170d09c36fe3a81d1663858e186934a3a3ca34aa6658e39a1eb69`.

## 2026-10-12 aromatic forest qualification

The next candidate extends `GENERAL` and `COMPACT` Forest serialization to
supported stereo-free aromatic graphs. Aromatic matching remains prepared over
each source aromatic-bond component, so a dot does not reset its source or
whole-component Kekule mode or its bond-order demands. Native witnesses cover
benzene and pyrrole rings split across segments, including ring closure and a
rejected bond distribution that exceeds one carbon demand. Public semantic
witnesses cover these paths and independent components; they check graph
identity only for positive outputs. Mixed lowercase/uppercase spelling within
one aromatic component and stereo Forest serialization remain open.

The frozen candidate passed 290 native tests with one ignored, six isolated
public Kekule tests, 14 compact-language tests, and six selected standard tests.
The integrated root checkout then passed 86 root Rust library tests and 42
selected public tests across Kekule semantics, compact language, standard
SMILES semantics, and public prefix integration. All bounded runners reported
verified cleanup. The root public run used the extension extracted from the
qualified wheel (wheel SHA256
`19caf8e874d8cebf4962c7afebfae2dae22d7f8ad4119b078f67a941b5594935`,
extension SHA256
`da832e22ac87c41e6f163adec9736b4185c8cee92d15338161e85dc099e63036`).
These are finite semantic witnesses, not exhaustive support or RDKit
writer-parity evidence.

## 2026-10-12 prefix work relative to visible output

At HEAD `bf3c165f`, a bounded probe followed only the selected `C.C...C`
token path for 2–7 identical disconnected methane components. In both
`GENERAL` and `COMPACT`, seven components retained 5,040 native states at the
single accepted text `C.C.C.C.C.C.C`; each intermediate prefix had at most two
visible next-token choices. The retained counts after successive `C` tokens
were 7, 42, 210, 840, 2,520, 5,040, and 5,040. These counts measure retained
walker entries from atom-identity histories; they do not establish that all
retained states are distinct. For this family their remaining visible
continuations coincide under permutation of the identical components. The
observation does not establish a general complexity or hardness bound.

`TextPrefix` stores a `Vec<Walker>` and `grouped_successors` expands every
state's choices before grouping by token text, without merging successors
(`rust/south_star2/src/prefix.rs`). Repeated `choices()` calls recompute the
grouping. Root and dot-root candidate generation considers each eligible
unvisited atom (`writer_state.rs`); the selected dot commits an atom identity
before emitting its identical text (`nonstereo_writer.rs`). Thus this chosen
path retains factorially many internal histories while showing a constant
number of next texts. Candidate attempts can also be rejected after solver
work. The mixed exact solver may search unresolved core assignments; its
topology projector does not enumerate spanning trees (`native.rs`). An
automorphism-aware continuation quotient is a possible direction, subject to
a proof that traversal, constraints, ring labels, pending spelling, and writer
regime preserve the same future visible language. Simple caching would reduce
repeated calls but leave the retained frontier growth.

Reusable probe: `tmp/exploration/south_star2/004_count_identical_component_prefix_states.py`.
It caps components at seven and asserts a 10,000-state frontier limit after
each selection; `choices()` materializes successors before that assertion, so
this is not a memory hard cap. Run it
through `scripts/run_bounded_tests.py --wall-seconds 60 --rss-mib 2048
--cpu-seconds 60 --address-space-mib 4096 -- env PYTHONPATH=python:.
python3 tmp/exploration/south_star2/004_count_identical_component_prefix_states.py`,
redirecting output to a temporary log. The bounded run exited zero with
`verified_no_live_members` cleanup. It did not enumerate the language or
measure latency.

### Distinct labeled paths: native work audit at `7d4d5a26`

The factorial methane example above groups different atom-identity histories
under one token text. If atoms are labeled and each distinct traversal is an
output path, those permutations are legitimate work. Its 5,040-state frontier
therefore does **not** establish redundant work under this cost model. A
comparison must hold a labeled path fixed and count internal work that does
not produce or prove a distinct labeled continuation.

Preparation does not eagerly form a product of tetrahedral choices across
centers. Each center has one 24-value ligand-order domain and one 8- or
16-value bond-role-pattern domain, with a root layout factor and one factor
per possible entry bond (`tetrahedral.rs:6,31-33,61-64,66-107`;
`nonstereo_writer.rs:935-985`). At atom entry, the writer tries parity and
available aromatic modes, restricts those domains, and activates the chosen
layout (`nonstereo_writer.rs:2973-3005`). These are compact local constraints;
the 24 permutations are distinct labeled local ligand orders when emitted.

An isolated copy of HEAD `7d4d5a26` ran the existing ignored native
work-count probe at one fixed labeled root/transition and at an unresolved
triangle prefix. The triangle produced two choices. Its first-edge, atom, and
second-edge topology counters each recorded 12 visits for the three-atom
graph, 36 for a connected nine-atom tail, and 52 for a connected 13-atom
tail with a distant inactive tetrahedral center. The last case also checked
eight potential layout factors.
All these cases reported zero mixed exact runs and zero mixed branch points.
The adjacent-center transition activated the second center and narrowed its
order/pattern domains from 24/8 values to singletons through two local layout
factor revisions and 16 potential-layout checks, again with zero mixed exact
runs. A forced-cycle candidate returned no successor after three first-edge
visits. Those visits were work incurred while rejecting that candidate; the
lack of a successor does not show whether the work can be avoided. The counts
show input-size dependent projector scans on this fixed path, not a factorial
or redundant same-path computation bound.

The probe's first run failed at its existing expectation that the adjacent
center choice text was `[C@H]`; the actual next grammar token was `[`. In the
isolated copy only, that assertion was changed to accept the observed `[` so
the work-count record could be read. The second run passed one ignored test
with verified process cleanup. The failed and corrected run logs are
`/tmp/grimace-complexity-probe.log` and
`/tmp/grimace-complexity-probe2.log`; the failed run also reported verified
cleanup. No repository runtime or test file was changed.
The corrected probe is diagnostic evidence, not a passing test of the
repository's unchanged assertion. To reproduce, archive HEAD into an isolated
directory, change only the probe assertion at
`nonstereo_writer_tests.rs:7588` to accept `[` as the observed choice text,
and run there:

```sh
python3 scripts/run_bounded_tests.py --wall-seconds 300 --rss-mib 4096 --cpu-seconds 300 --address-space-mib 8192 -- cargo test --offline -p grimace-south-star2 native_work_count_probe_fixed_interior_prefix -- --ignored --nocapture
```

Mixed exact search remains a source-level risk, not an observed blowup on
these paths. It branches over unresolved core variables, propagates each
value, and stops only when every projected domain value has a witness
(`native.rs:465-585`). A late witness could require many earlier assignments;
the saturation check does not give a polynomial bound. The fast-exit
conditions at `native.rs:500-505` kept these measured cases out of DFS. The
next discriminating step is to derive a bound for reachable mixed cores under
writer-generated constraints, or construct a connected asymmetric labeled
graph whose fixed native prefix reaches a growing mixed core and count nodes,
leaves, and saturation point under a hard cap. A generic
solver model or whole-support timing would not establish that reachability.

## 2026-10-12 selected directional Forest segment qualification

Generic AnyOrder and Compact now select Forest topology for inputs with
configured directional relations. Their existing carrier-mark, side-pattern,
and ring-placement constraints remain active across dot segments. The traversal
progress owner determines when all atoms and bonds of a chemical component are
represented; the shared frame-completion boundary checks directional side and
carrier resolution at that point, including return from a leading-dot branch.
The RDKit writer regime retains its Tree topology and anchoring rules.

A selected connected F-C=C-Cl path places one directional carrier at a ring
endpoint across a dot; both generic languages accept the matching E/Z spelling
and reject the opposite marker. RDKit parser-based semantic checks distinguish
those isomers, not RDKit writer support. A native leading-dot branch witness
completes a configured E/Z component before returning to a disconnected
spectator. Full Forest directional support and leading-dot stereo semantic
qualification remain open.

Integrated checkout qualification: bounded offline South Star 2 Rust tests passed
296 with one ignored; root Rust tests passed 86. The six selected public Python
modules passed 77 tests against the freshly built extension copied to the
CPython 3.12 ABI import path (SHA-256
`c4742909421afe558cc83244c3640b52fa170a5fabfac3702c50a8e6b7792588`).
All three runners reported verified process-tree cleanup. The public selection
includes configured E/Z cross-dot ring parse-back, compact and generic paths,
mixed alkene/tetrahedral semantics, and unspecified-stereo controls. These are
selected semantic and regression results, not full maximal-language support,
all-prefix liveness, or RDKit writer parity for the new dot spellings.

## 2026-10-13 mixed aromatic representation slice

The general language now selects lowercase or uppercase spelling per aromatic
atom and aromatic, explicit single, or explicit double surface spelling per
source-aromatic edge. One exclusive matching factor checks both endpoint cases
and a latent integer 1/2 assignment against the existing immutable atom double
demands. A bare lowercase token fixes each incident source-aromatic edge to
surface aromatic; bracket lowercase can participate in mixed spelling. The
compact language compiles the old whole-component alternatives into the same
factor: all lowercase with aromatic edges, or all uppercase with integer edges.

Selected benzene, pyrrole, charged pyridinium, and naphthalene mixed bracket paths pass
native support tests; the selected positive strings parse back to the intended
RDKit molecules under 2026.03.4. The factor's per-value projection agrees with
a bounded finite assignment witness. This is a supported semantic slice, not
an exact RDKit writer-support claim or completion of the name `MAXIMAL`.
Mixed bare atom eligibility, all legal single-bond omissions, and broader
SMILES grammar coverage remain open.

The previously checked-in wildcard-ring witness `*1ccccc1` remains in
`GENERAL`: its sanitized RDKit parse preserves the intended aromatic graph,
even though raw bond-token inference at the wildcard is single. `COMPACT`
continues to spell that aromatic wildcard incidence explicitly. Wildcard has
one atom grammar branch while its unspelled representation case remains
unresolved; subsequent bond constraints choose the supported continuation.

After integration into the root checkout at `b083925e`, bounded offline root
Rust passed 86 tests, and the six selected public Python modules passed 79
against RDKit 2026.03.4 and the local compiled extension with SHA-256
`a4f55c5522efaa1ecbb7075de7f9b0d5b7eca8928a15a304b9f4dead97f2f9bf`.
Both runners verified process-tree cleanup. Captured logs are
`/tmp/grimace-mixed-representation-root-rust.log` and
`/tmp/grimace-mixed-representation-root-python.log`. The isolated full South
Star 2 native suite passed 299 tests with one ignored. These checks qualify
selected supported paths and preserve the separate RDKit writer-parity contract;
they do not establish complete fact-preserving SMILES support.

## 2026-10-13 deferred prefix successor preparation

Prefix choice discovery now records each distinct token text, its labeled
successor count, and acceptance while discarding unselected successors. Each
offered choice shares a materialized parent frontier and constructs only the
selected successor frontier when traversed. This preserves every labeled
native state and every ring-label alternative; choice discovery still checks
all native choices and propagates their failures. Direct `select_text` also
scans all source states before reporting an invalid token. An independent
small eager reference checks token order, multiplicity, acceptance, and
continuations for sibling selections.

On the same selected two-benzene GENERAL path, the f78c7bbc release extension
took 3.88 s with 478.77 MiB peak process-tree RSS; the baseline plus deferred
prefix extension took 5.63 s with 122.55 MiB. A separate isolated mixed-bare
candidate exceeded 2 GiB while eagerly retaining more than two million
successors at one prefix. Combined with deferred prefixes, that candidate
completed the path in 22.52 s with 126.72 MiB. These bounded measurements
include runner overhead and sampled RSS; the mixed candidate has a larger
language, so its time is not a measure of the deferral cost alone.

## 2026-10-13 mixed bare aromatic carbon candidate

The isolated candidate removes the bare-lowercase incident-edge restriction
and the compact whole-component case restriction. Its bare neutral aromatic
carbon interpretation is explicit: with normal total valence four and two or
three graph neighbors, implicit H is `3 - graph degree`; an independent latent
single/double assignment must still satisfy every prepared atom's integer
valence. This admits selected mixed bare benzene paths such as `c1=cc=cc=c1`.
The rule is a project interpretation of underspecified aromatic hydrogen
semantics, not a claim that OpenSMILES uniquely prescribes it. RDKit 2026.03.4
parses that selected spelling differently and is not its H authority. A
temporary test witness reads only raw six-carbon ring structure from RDKit,
enumerates at most 64 latent edge assignments, and applies the stated H and
valence rule independently. Its narrow scope should fold into a shared atom
fact, bond event, constraint, and rendering interpretation; it is no general
SMILES parser or exact-support proof. The `MAXIMAL` language contract remains
incomplete, including legal bond omissions and other aromatic semantic cases.
The native test includes a prepared carbon with one explicit graph-H neighbor
and checks that its non-graph H count must be zero before bare `c` is eligible.
RDKit input preparation was separately checked with `removeHs=False` to
confirm the explicit graph-H neighbor; both `[H]c1ccccc1` and
`c1([H])ccccc1` were accepted by the unchanged 9a7b9c0b public decoder.
The candidate public test accepts `[H]c1=cc=cc=c1` in `GENERAL` and its
smallest-label counterpart in `COMPACT`. The synthetic native graph fixture
tests atom eligibility only; it does not reproduce every Python preparation
detail needed for those full paths.

## 2026-10-13 selected leading-dot tetrahedral qualification

A bounded independent event witness interprets a four-atom C/F/Cl/Br star
with one H-bearing tetrahedral carbon. OpenSMILES v1.0 §§3.7.1 and 3.8.2
permit the dot-started branch and place an incoming ligand before bracket H;
ring-bond occurrence order reserves the ligand slot before closure. The
fixture checks `F1(.[C@@H]1(Cl)Br)` and `[C@@H]1(.F1)(Cl)Br` against
`F[C@H](Cl)Br`, their compact label-zero forms, and opposite markers with
both original and atom-renumbered source. The helper constructs edges and
signed ligand-order parity from tokens without asking RDKit to parse emitted
strings or consulting production parity internals. This temporary witness
should fold into the shared fact/event/constraint/solver/renderer model.
It qualifies these selected paths only; full Forest stereo, exact RDKit writer
support, and complete `MAXIMAL` coverage remain open.

## 2026-10-13 finite OpenSMILES completeness audit at `0a8468a6`

Scope: this audits `GENERAL` against the every-and-only fact-preserving target
in `docs/api/smiles-language.md`, for inputs actually admitted by preparation.
`COMPACT` is a narrower spelling policy; RDKit writer parity is a separate
contract. This ledger covers grammar alternatives and their semantic owners,
not all candidate strings. OpenSMILES v1.0 grammar and §§3.1–3.8 are the
syntax reference: <https://raw.githubusercontent.com/opensmiles/OpenSMILES/master/opensmiles.asciidoc>.
An RDKit parse is evidence for selected cases, not the language authority.

| Obligation | Current owner and narrow closure | Remaining precise obligation |
| --- | --- | --- |
| Admission and identity | `python/grimace/_south_star2/preparation.py` snapshots accepted atoms, bonds, hydrogen, aromatic and stereo facts or rejects unsupported forms. `docs/api/smiles-language.md` defines graph-isomorphism identity; atom IDs are witness labels. A source quadruple bond, for example, is outside admission, not a missing output. | Prove every accepted preparation preserves each identity fact, including graph versus non-graph H and unspecified stereo. Admission must not be inferred from one successful string. |
| Atom spelling and numeric fields | `rust/south_star2/src/atom_spelling.rs` owns bare/bracket, isotope, H, charge, class and tetra marker phases. `Phase::Number` emits exactly `0*` followed by the fixed decimal value, including value zero, in `GENERAL` (lines 685–735). This closes lexical zero padding for represented isotope/class values under the project's explicit numeric interpretation; charge and H widths are finite grammar cases. `Phase::AfterOpen` requires the isotope field for native `Some(value)`, including `Some(0)`; public preparation normalizes RDKit isotope zero to absence. | Partition admitted element/valence/aromatic contexts to check bare eligibility and other bracket omissions. |
| Ordinary single-bond spelling | OpenSMILES §3.5 explicitly gives `C-C` and `CC` the same meaning. Preparation maps an ordinary RDKit single bond to empty text (preparation.py lines 314–322). `nonstereo_writer.rs` fixes that bond to elided representation at lines 595–607, and `chemical_bond_specs` returns only its prepared text at lines 3140–3141. | **Confirmed coverage counterexample:** `GENERAL` accepts `CC` but rejects `C-C` for the same admitted ethane facts. Extend the bond representation decision to offer both explicit `-` and omission wherever the standard interpretation agrees; inspect traversal, branch and both ring endpoints, rather than fixing only inline bonds. Ring behavior is source-predicted here, not a live ring witness. Then show explicit/omitted alternatives do not change aromatic or directional meaning. |
| Other bond spelling | `nonstereo_writer.rs` owns `=`, `#`, `:`, directional marks, aromatic edge modes and ring endpoint plans. Preparation excludes unsupported quadruple and dative source bonds. Existing matching constraints relate aromatic surface forms to latent integer bond orders. | Partition every admitted bond by fixed single/double/triple, aromatic matching order, carrier role, and endpoint atom spelling; establish that `chemical_bond_specs` and `ring_emission_specs` offer every and only semantically equivalent explicit/omitted symbol at every syntactic position. The ordinary-single counterexample disproves closure of this partition today. |
| Roots, traversal, branches and dots | `traversal.rs` owns visited atoms/bonds, residual attachments, frame stack, branch return, new segment and dot branch transitions (lines 831–1044); `writer_state.rs::structural_frontier` publishes candidates. Selected forest and leading-dot paths have semantic witnesses. | Check the finite grammar constructors separately: initial/root atom, inline continuation with optional bond, `(chain)`, `(bond chain)`, `(dot chain)`, top-level dot, and branch return. For each constructor prove forward transition availability and reverse grammatical/fact-preserving interpretation. The shared induction step must show residual partition and frame completion preserve every legal sibling order and cross-dot ring obligation; selected paths do not establish that invariant. |
| Ring events and labels | `nonstereo_writer.rs::available_ring_label_options` and `ring_label_spellings` cover the OpenSMILES numeric label space `0`–`99`, including `%00`–`%09`; `RingLabels::release` frees a matched slot for reuse and `COMPACT` chooses the smallest free label. `traversal.rs` pairs each ring event to one prepared bond. | Check ring open, ring close, and later label reuse as separate finite event cases, crossed with bond text absent/first-end/second-end/both-end. For each case prove one prepared edge and one consistent chemical order; the induction invariant must exclude unmatched or contradictory pairs at acceptance. Label enumeration alone does not close event coverage. |
| Aromatic interpretation | The project specifies bare neutral aromatic carbon H as `3 - graph_degree` subject to feasible integer valence. `nonstereo_writer.rs::prepare_aromatic_matching` and atom/bond modes share one latent edge assignment; selected mixed uppercase/lowercase paths pass. | For each admitted aromatic atom and edge context, establish equivalence between the standard/project interpretation and the matching constraint, including mixed bare/bracket forms, omitted bonds, charges, heteroatoms and exocyclic bonds. The exact case partition is by emitted endpoint spelling and edge order/symbol; selected benzene and heteroatom examples do not discharge it. |
| Tetrahedral and directional stereo | Prepared tetrahedral layouts and directional side/carrier factors own marker and slash placement; selected ring-segment, leading-dot tetrahedral and configured E/Z paths have independent semantic witnesses. | For every admitted configured center/double bond and every grammar-valid Forest event order, prove ligand-order parity and carrier constraints yield exactly the preserving markers. Also prove unspecified stereo never becomes configured. Ring and dot crossings require explicit cases because they change event order without changing chemical identity. |
| Prefix soundness and completion | `prefix.rs` unions same-text labeled walker successors and defers materializing unselected ones; under deterministic replay of each labeled walker transition, this aggregation preserves their union. A small eager reference checks that behavior on selected examples. Terminal acceptance checks pending spelling, clean ring labels, and complete traversal scope/graph (`nonstereo_writer.rs::is_accepted`, lines 2113–2119); solver feasibility is a transition invariant. | Prove each offered walker choice has an accepting completion, and that every grammar-valid next event is offered by its owner before text aggregation. The confirmed `C-C` miss is a missing walker spelling choice, independent of prefix aggregation. |

The concrete bond probe used RDKit 2026.03.4 and the local `_core` extension
SHA-256 `c72671b59931760ab98b47d2f0272664cc199fce8feb98d11c0bb807915e5c49`.
Both RDKit sources `CC` and `C-C` accepted output `CC` and rejected output
`C-C` through `decoding.start(..., language=SmilesLanguage.GENERAL)` with a
1,000-prefix-node cap. The 60-second, 1,024-MiB bounded runner reported
`verified_no_live_members`; log: `/tmp/grimace-completeness-audit-bond.log`.
The same bounded runner separately verified that source `[13CH4]` has RDKit
isotope fact 13 and `GENERAL` accepts both `[13CH4]` and `[CH4]`; log:
`/tmp/grimace-completeness-audit-isotope.log`. Omission of the isotope field
changes the OpenSMILES isotope interpretation, independent of RDKit parse-back.
The selected public-path checks can be repeated with the existing bounded
`tests.south_star2.test_compact_language._walk_language` helper and a 1,000-node
cap; `None` means no accepted path:

```python
from rdkit import Chem
from grimace.decoding import SmilesLanguage
from tests.south_star2.test_compact_language import _walk_language

general = SmilesLanguage.GENERAL
ethane = Chem.MolFromSmiles("CC")
carbon_13 = Chem.MolFromSmiles("[13CH4]")
assert _walk_language(ethane, "CC", general, max_nodes=1000)
assert _walk_language(ethane, "C-C", general, max_nodes=1000) is None
assert carbon_13.GetAtomWithIdx(0).GetIsotope() == 13
assert _walk_language(carbon_13, "[13CH4]", general, max_nodes=1000)
assert _walk_language(carbon_13, "[CH4]", general, max_nodes=1000)
```

That extension is the qualified `d9b3354f` build; `git diff
d9b3354f..0a8468a6 -- python rust Cargo.toml Cargo.lock pyproject.toml` is empty,
so its executable implementation matches this audit's HEAD. The immediate
minimal repairs are the mandatory nonzero-isotope field and the ordinary
single-bond spelling choice with its ring/branch placement cases, each followed
by a focused semantic regression. This audit does not implement either repair
or claim a universal completeness proof.

`tests.south_star2.test_standard_smiles_semantics.GeneralLanguageNormativeRegressionTests`
now records both selected obligations in default discovery: `C-C` must be
accepted for admitted ethane, and isotope-free `[CH4]` must be rejected for
admitted `[13CH4]`. The corresponding `CC` and `[13CH4]` controls pass. The
ordinary-single-bond assertion remains red until its runtime repair; the
isotope assertion passes with the mandatory field repair. They specify
OpenSMILES/fact identity rather than RDKit writer support. The
60-second bounded focused run failed at exactly those two assertions and
reported `verified_no_live_members`; log: `/tmp/grimace-normative-red-focused.log`.

After the two repairs, both normative assertions pass. In `GENERAL`, an
ordinary prepared single bond offers omission or `-` on traversal, branch, and
either ring endpoint; `COMPACT` and the RDKit writer regime retain their prior
bond spelling policy. Native ring witnesses cover all four omitted/explicit
endpoint combinations, and the independent tree reference covers both
spellings for each ordinary single bond. Large traversal/stereo conformance
fixtures keep their exact assertions on a declared omitted-single spelling
sublanguage; selected `GENERAL` explicit spellings are checked independently.
Three expensive public parse-back fixtures use the finite `COMPACT`
traversal/stereo lane, with explicit `GENERAL` aliases covered by the normative
regression.

The second repair's release candidate `_core` has SHA-256
`bf9ea211c8fe631b0dd5d35943a02e3b928b985bdfcd732e2b51561b4b379132`.
Seven relevant public modules passed 86 tests in the bounded runner
(`/tmp/grimace-single-public-final.log`). The South Star 2 native core passed
302 tests, with one pre-existing ignored test, when the long-running
`prefix::support_growth_tests::bounded_support_growth` diagnostic was excluded
(`/tmp/grimace-single-native-core-final.log`). That diagnostic separately
completed CO, CCO, CC(C)O, and C1CO1 with one and four ring labels, then reached
the source-declared incomplete C1CO1/all-110-label case and exceeded the
300-second wall bound without an assertion (`/tmp/grimace-single-support-growth-standalone.log`).
Both runs reported verified cleanup. Root Rust tests remain unverified: two
bounded-runner attempts stopped during compilation because process accounting
could not reconcile transient `/proc` or namespace state, each with verified
cleanup (`/tmp/grimace-single-root-rust-final.log`,
`/tmp/grimace-single-root-rust-final-retry.log`).

## 2026-10-13 MAXIMAL contract naming

The public `SmilesLanguage.MAXIMAL` (`"maximal"`) is the normative every-and-only
OpenSMILES language for admitted molecular facts described in
`docs/api/smiles-language.md`; it is the default. This name denotes the target
contract, not completion. `COMPACT` remains a distinct restricted spelling
policy, and RDKit writer parity remains a separate contract. The earlier
`GENERAL` terminology in this dated roadmap records historical snapshots.

The current MAXIMAL implementation has known defects against that contract
and incomplete qualification of traversal, aromatic, stereo, and lexical
coverage. The prior ordinary-single-bond and isotope counterexamples were
repaired; their passing checks are focused evidence, not a completeness proof.
Audit against the normative contract and record further counterexamples as
defects rather than renaming or narrowing the policy.

## Bounded bracket-production closure slice

For a fixed admitted atom fact and caller-selected aromatic/tetrahedral context,
`atom_spelling.rs` follows the OpenSMILES bracket field order: optional isotope,
symbol, optional tetrahedral marker, H count, charge, and class. Native tests
compare the complete finite H/charge suffix set from `AtomSpellingState` with
an independently constructed grammar and numeric interpretation for H 0–9,
charge −99–99, and both spelling policies. A fixed caller marker is checked for
its two `MAXIMAL` tetrahedral forms (`@`/`@TH1` or `@@`/`@TH2`) and one
`COMPACT` form. Separate cases check field order, hydrogen-atom exclusion,
isotope zero versus absence, and class-zero behavior. The isotope and
class `NUMBER` states retain a zero-padding loop; induction on the number of
prefix zeros covers every finite padding length for a fixed represented value.
Every chosen field phase has a path to the next field, so this local spelling
argument does not require enumerating unbounded suffixes.

This establishes only a local production argument. Bare-atom eligibility in
aromatic contexts, caller stereo facts, admission, and global walker completion
remain separate obligations. The element table deliberately includes four
post-2016 symbols; strict OpenSMILES v1.0 lexical treatment of those symbols
remains an explicit contract decision.

## Structural grammar constructors and local phase invariant

OpenSMILES v1.0 grammar (§2) gives `branched_atom ::= atom ringbond* branch*`;
`chain` continues with an atom, bond plus atom, or dot plus atom. Each branch
contains a nonempty chain, optionally preceded by a bond or dot. The current
event correspondence is:

| Grammar constructor | Event owner and focused evidence |
| --- | --- |
| Initial atom or a dot-segment root | `WriterState::structural_frontier` offers a root; `TraversalState::begin_segment` visits it. The native root/segment test distinguishes branch return from segment finish. |
| Ring occurrence before a branch | The frontier offers ring open/close while `active_can_emit_ring`; `WriterFrame::commit_ring` records a prefix of local bond occurrences. The native triangle test covers opening, traversal, and closure. |
| `(chain)` and `(bond chain)` | `BranchChild` commits an incident bond, enters a nonempty child frame, and `CloseBranch` restores its parent. Bond spelling belongs to `nonstereo_writer.rs`; native star and parent-sibling tests cover return and remaining attachments. |
| `(.chain)` | `DotBranchRoot` begins a new segment with a suspended parent; the public compact graph witness covers a selected leading-dot branch. |
| Inline atom and bond plus atom | `InlineChild` consumes the last active attachment; bond spelling is selected before the new atom. The native star and triangle tests cover this transition. |
| Dot plus atom in a chain | `DotSegmentRoot` begins another segment; selected public dot tests show open ring obligations crossing segments. |
| Matched ring label and later reuse | `RingLabels` allocates on first endpoint and releases on second; native label tests cover free-slot reuse and the public compact test covers a connected example. |

Locally, a frame's ring occurrences occupy a prefix of its emitted bonds.
Starting a branch closes that frame's ring phase, including after a branch
return. The frontier excludes ring candidates after this boundary and forces
waiting ring closures before it; the ring-plus-substituent native test checks
the crossing while a later atom may still close an earlier label. The same
structural grammar governs `MAXIMAL` and `COMPACT`; each profile restricts
whole event or spelling productions at their owners. RDKit writer scheduling
is a separate policy. These local phase facts do not prove that every offered
prefix has a completed graph, that every grammatical event order is offered,
or that all bond and stereo interpretations preserve the admitted facts.

## Root ownership in the Forest topology factor: design before implementation

The current Forest factor constrains physical Traversal bonds to be acyclic.
It has no variable for a textual segment root. A role assignment can therefore
join two already emitted roots into one Traversal component while remaining a
valid plain forest assignment. This is a missing model law, independent of the
writer frontier. On the four-atom path `A-U-V-B`, starting a segment at `A`
and a dot branch at `B` makes such an all-Traversal role assignment possible
in the current factor. A bounded public `FCCCl` probe found productive ring
continuations after the analogous prefix `F(.Cl`; this is not evidence of a
currently published dead choice. A future profile that restricts all three
path bonds to Traversal would remove that escape.

For each atom in a Forest-prepared component, preparation should own one
binary root-role variable: selected when the atom starts a segment and
unselected when it is entered as a child. The component's existing topology
factor should include those variables. Interpret a selected root as a virtual
edge from that atom to one component-local super-root. A valid assignment is
one whose physical Traversal edges plus selected virtual edges are acyclic.
This says precisely that each physical Traversal component has at most one
selected root. At complete traversal, every atom has its root role committed
and every physical Traversal component has exactly one root because its first
visited atom started a segment. The virtual edges are solver facts only; they
do not create SMILES bonds or alter chemical components.

`WriterState` should restrict the existing prepared root variable on each
`Root`, `DotBranchRoot`, and `DotSegmentRoot` transition, and restrict it to
unselected on each child-entry transition. No root cache belongs in
`GraphProgress`. The native Forest projector should union forced virtual
edges with forced Traversal edges and remove an optional edge's selected role
when that edge would close a cycle. The independent finite-assignment checker
should apply the same stated graph law by a separate straightforward component
count or union procedure. Tree topology retains its physical spanning-tree
connectivity requirement and needs no virtual super-root path.

First pin the model gap with a native state test: start the two roots on the
four-atom path, then use the existing test-only bond-domain restriction to
require all physical bonds to be Traversal. The new factor must report a
contradiction; the old plain Forest factor accepts the assignment. Also check
all small complete root/role assignments against the graph law and retain
productive ring continuations for the public case. This strengthens the
structural model without adding a profile, changing emitted text, or claiming
global prefix-completion proof. Ring-label scheduling, lexical availability,
and semantic factor activation remain separate completion obligations.

The rooted-Forest model slice was checked in an isolated checkout from
`ed525b06`. Focused native tests cover two emitted roots with a forced
all-Traversal path, reciprocal root/bond role pruning, empty-root registration,
mixed-factor projection, and all 64 complete role/root assignments of a
three-atom triangle against an independently computed component/root-count
law. Existing tetrahedral root-factor activation and selectable child walks
passed. The bounded offline release wheel build reported verified cleanup;
the seven selected public Python modules passed 86 tests against that rebuilt
extension. For admitted `FCCCl`, both `F(.Cl0)CC0` and `F(.ClC0)C0` remain
accepted ring continuations. Logs are `/tmp/grimace-rooted-forest-focused.log`,
`/tmp/grimace-rooted-forest-build.log`, `/tmp/grimace-rooted-forest-public.log`,
and `/tmp/grimace-rooted-forest-fcccl.log`. The earlier full native run reached
the support-growth test and exited under its inherited CPU limit after about
211 seconds; its runner verified no live descendants. That interrupted run is
not a full-suite pass. The model law does not establish global prefix
completion, ring-label resource sufficiency, or latent future-factor activation.

## Selected structural extension witnesses

For an admitted prepared graph, a complete structural serialization visits each
atom exactly once. Each bond is represented exactly once, either on entry to a
child or by two ring occurrences at that bond's named endpoints. An initial atom
or a dot continuation starts a segment; entering an ordinary branch child does
not. Every branch starts with a nonempty chain and returns to its suspended
parent, while an inline child advances the active chain. At each atom, all ring
occurrences precede branches and inline continuation. Every opened ring is
paired at its other endpoint, every branch scope is closed, and every physical
Traversal component has exactly one segment root. Chemical components may
contain several such Traversal components joined by ring edges. These
conditions describe structural events only: atom and bond spelling, chemical
validity, stereo, RDKit writer policy, and finite ring-label allocation require
separate arguments.

The prefix counterpart records which atoms have been visited and, for each
bond, whether it is unrepresented, traversed, opened at a named endpoint, or
closed at both named endpoints. It also records the active frame, suspended
parents, and permanently finished segments. A branch start seals its parent's
ring phase: every still-unrepresented incidence at that parent must be a later
Traversal entry. A ring may nevertheless close after return at a later
entered atom. A segment boundary may likewise leave a named ring opening for
a later segment. These are structural obligations, not a claim that a local
frontier choice always has a globally completing suffix.

The native selected schedules now check candidate publication before each
transition, then inspect raw progress and complete the prescribed suffix. The
six-atom residual cycle checks two parent attachments in one chemical residual
component and a ring opened in a branch, followed by return, remaining child
entry, and closure. A three-atom dot branch checks that its ring closes at a
later entered atom after parent restoration, with no closure offered at the
sealed parent. Another three-atom schedule crosses a completed segment boundary
with an open ring and closes it in the next segment. Existing triangle and star
tests cover ring-before-entry and ordinary branch return. These finite
witnesses establish those specific extension paths and state observations.
They do not establish arbitrary valid event availability, the general
residual-partition sufficiency step, or universal prefix liveness. A full
correspondence proof still needs to show that every valid structural schedule
projects to an offered candidate at each prefix and that every offered
structural state retains some completing schedule under the prepared factor
domains.

The next named-endpoint check makes one necessary condition explicit at stable
constructor boundaries. For an open ring, its named first endpoint must be an actual
visited endpoint of its bond. If its other endpoint has already been visited,
that endpoint must be the active frame and its ring phase must still be open.
A suspended or finished endpoint cannot acquire a later ring occurrence. The
native test reconstructs this fact from the prepared bond's endpoints, visited
atoms, bond progress, and the observed active frame; it also
rejects an unrepresented bond whose endpoints are both visited. Malformed
observations from a valid closing state remove the active frame or seal its
ring phase while retaining represented bond progress; each isolates the
missing ring condition. This is a prefix invariant, not
an observed public failure or a completion theorem.

For a constructive structural extension argument, a total role assignment
must partition *all* bonds into Traversal and Ring, and assign each atom a
root role, compatibly with all current bond and root restrictions and
applicable structural factors. Represented Traversal edges on visited atoms
must form trees, each with exactly one emitted segment root. A further frame
premise must account for the active atom, every suspended branch parent,
permanently finished atoms, and the ordinary branch return or dot-segment
boundary; the current named-endpoint check does not establish that premise.

Given those premises, acyclicity and the rooted-Forest law give at most one
visited Traversal attachment into a rootless unvisited Traversal component.
To turn that graph fact into a child-entry step, first emit every outstanding
Ring incidence at the active atom. A suspended frame must have only Traversal
incidences remaining, and a finished atom must have none. If the proposed
child then had a second unrepresented edge to a visited atom, that edge would
also have to be Traversal, contradicting the one-attachment graph fact. A
component with no visited Traversal attachment instead needs a separate
root-domain and frontier availability argument. The graph fact, preservation
of its premises through transitions, and publication of each abstract event
as a candidate are distinct obligations; the latter two remain unproved.
Ring-label capacity, chemistry, profile restrictions, and later factor
activation remain separate obligations.
