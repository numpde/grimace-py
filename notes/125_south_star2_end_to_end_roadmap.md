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

Finite local extension qualification at `ffdfdbc2`: for a three-bond center
with one implicit H and a four-bond center, all 112 root/entry and fixed-role
combinations (68 legal, 44 illegal entry-ring) preserve exactly the lexical ligand orders formed by entry first,
then implicit H when present, then ring occurrences, then traversal children.
Both marker orientations give 224 prepared context/role cases and 448 parity
transitions, checked against independent
permutation inversion counts. The actual prepared Generic AnyOrder Forest
latent factor was activated for every fixed-role case alongside the initially
active Forest topology factor; its projected order
domain matched the independent local set or contradicted an illegal entry-ring
or incompatible marker. This is a local extension law under fixed incident
roles, not a proof that all jointly constrained molecules or public writer
paths complete. The lexical order follows [OpenSMILES §3.8.2](https://opensmiles.org/opensmiles.html);
the parity-to-marker mapping remains a prepared fact rather than an RDKit
writer-support claim.

The native writer-entry bridge fixes the same 112 local contexts under both
marker orientations and probes both parities through actual root or
leaf-to-center atom choices. On the prescribed elided child route, all 136
legal prepared cases yield exactly the independently expected marker and
remaining order domain; 88 entry-ring cases cannot enter through that bond.
The chosen factor alone activates for the center, its root decision matches
root or child entry, and the selected traversal/ring role partition is
preserved. Additional bounded witnesses cover elided and explicit branch
entries, explicit inline entry, and one adjacent-center child transition with
both center factors active and fixed incident roles. These observations qualify the
atom-entry seam for those routes. They do not enumerate full suffix support or
establish RDKit writer string parity.
Completion while honoring already selected local orders, and composition with
finite labels and other semantic factors, remain separate obligations.

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

## Residual attachment definition and bounded qualification

For a visited frame atom, a residual attachment contains exactly its
unrepresented incidences to unvisited atoms, grouped by connected component
of the unvisited induced graph. `ResidualPartition::remove_atom` deletes the
newly visited atom. With zero residual neighbours it removes an isolated
component; with one it retains the connected remainder; with several it
searches each remaining neighbour's reachable atoms and assigns one component
per reachable set. Assuming the pre-deletion partition is correct, these cases
preserve the unvisited connected components. `attachments` then scans every
incident edge of the frame atom and groups precisely the edges whose bond is
unrepresented and other atom unvisited. Thus an empty attachment list means
there is no such incidence at that atom, provided the partition and bond
progress agree with the visited set.

A native test checks this definition on all 43 connected labelled simple
graphs with two through four atoms, all 601 nonempty visited subsets, and all
1,268 choices of a visited frame atom. It compares incremental groups with
the existing independent BFS reconstruction and independently flattens the
groups against the graph's eligible incidences. This uses canonical deletion
order and unrepresented bond progress; the earlier selected five-atom shape
test covers all deletion permutations for those shapes. Existing constructor
tests additionally check that inline advance, branch-child completion, and a
dot segment inside an outer branch leave discarded atoms with no eligible
incidences. These finite checks support the partition/attachment premise;
they do not establish every writer prefix or universal candidate completion.

The finite rooted-Forest attachment check enumerates every labeled simple
graph on two through four atoms, every acyclic Traversal/Ring assignment, and
every visited/root subset consistent with one emitted root per visited
Traversal tree and at most one emitted root per full Traversal tree. Its
independent graph partition finds 10,900 compatible rows. Every unvisited
Traversal component has at most one Traversal attachment to visited atoms;
8,450 attached and 8,456 unattached residual components occur. This checks
the stated graph lemma at the finite bound, without assigning roots to future
trees or constructing writer successor paths; it does not prove arbitrary
graph sizes. The native-backed writer-state root-factor test additionally
observes that a root role remains selectable without a forced Traversal path
to the emitted root and becomes unselectable when such a path is forced.
Selected live dot-branch and cross-segment transitions check `can_enter_atom`,
child publication, and absence of unrepresented incidences from a permanently
finished atom to an unvisited atom. These root-availability observations use
ordinary preparation, no explicit restrictions on unvisited root variables,
and the current rooted-Forest factor; once all scopes finish, an unvisited
component without a forced Traversal path to an emitted root retains a root
choice. Additional profile root restrictions can remove that choice. The
attachment equality follows conditionally from the partition induction above.
The constructor induction below addresses preservation of its partition and
bond-progress premises through residual splits, inline parent discard, and a
dot segment inside an outer branch.

## Conditional structural correspondence for the topology-only Forest

The correspondence argument is restricted to the Forest writer in `AnyOrder` with ordinary
prepared root variables, unconstrained Traversal/Ring role choices except the
rooted-Forest topology factor, simple graphs, unlimited ring labels, and stable
boundaries after child entry. Atom and bond spellings, stereo, chemistry,
representation factors, RDKit scheduling, and the public finite label supply
are outside the claim. Its state boundaries are after a completed child entry,
not the intermediate state after a child candidate commits its bond and before
its atom is entered.

At such a boundary the induction invariant is: every bond has its named
unrepresented, Traversal, Ring-open, or Ring-closed progress; no unrepresented
bond has two visited endpoints. Every unresolved incidence at a visited atom
belongs to the active or a suspended frame; a permanently finished atom has
none. Every suspended frame is sealed, so its remaining incidences are forced
Traversal. If a Ring-open bond's second endpoint is visited, that endpoint is
active in its open ring phase. The residual partition is exactly the connected
components of unvisited atoms, and the **active** attachment list is exactly
its unrepresented incidences to unvisited atoms. A suspended frame's cached
attachments may be stale while a child runs; restoration recomputes them.
Represented Traversal edges on visited atoms are rooted trees with one emitted
root each, and the live return stack records their nested branch scopes.
Visited atoms have their ordinary root variables committed to one for segment
entry and zero for child entry. For an unvisited atom, root one is removed only
by a forced Traversal path to an emitted root; that path persists under later
restrictions. This root-domain statement uses only ordinary preparation,
entry restrictions, and the rooted-Forest projector.

The empty prefix supplies the base case: no visited atoms, no represented
bonds or frames, both root values initially available, and prepared residual
components equal the graph's components. For the empty graph, these claims
are a vacuous structural-machine base; this does not assert that an empty
SMILES is admitted. The
transition step is the following finite case split over **successful**
structural commits; child commits and their atom entries form one macrostep.
Raw frontier proposals alone are not induction transitions.

| Constructor | Preservation and availability step |
| --- | --- |
| Initial `Root` | `can_enter_atom(root, None)` checks every earlier visited incidence (none at the base). The atomic root-one restriction and Forest projector accept a compatible root; `begin_segment` removes that atom from the residual partition and creates its frame. |
| `RingOpen` | The active incidence is unrepresented and leads to an unvisited atom. Restricting it to Ring and recording its named first endpoint removes that incidence from the active attachment list while the ring phase remains open. |
| `RingClose` | Its named first endpoint and active other endpoint are checked; the active frame is still in its ring phase. Closing the second occurrence changes no residual component. |
| `BranchChild` plus entry | For any successful branch, `can_enter_atom(child, bond)` excludes another unrepresented edge from the child to a visited atom, and waiting Ring closures block departure from the parent. The commit atomically selects Traversal and child-root zero and forces remaining parent incidences to Traversal. It then seals/suspends the parent; entry removes the child from the partition and creates its frame, and the return stack gains that parent. The unique-attachment lemma is used separately to choose an available child in the forward extension. |
| `InlineChild` plus entry | The sole remaining active incidence is Traversal. Waiting Ring closures block departure; consuming that incidence empties the discarded parent's work. The child receives root zero and a new active frame; any outer return stack is unchanged. |
| `DotBranchRoot` | Ring closures at the active atom must be discharged first. The event seals its remaining incidences as Traversal and suspends it, while the root-one restriction and `can_enter_atom(root, None)` admit a fresh compatible root. |
| `DotSegmentRoot` | The active frame has no attachments or waiting closure. Discarding only that frame leaves any outer return stack intact; root selection and residual removal establish the new segment frame. |
| `CloseBranch` | The exhausted child frame is discarded and its suspended parent restored. `ResidualPartition::attachments` recomputes the parent's exact current list; its seal remains in force. No waiting Ring closure can newly appear at that parent because a visited second endpoint would have had to be active before suspension. |
| `FinishSegment` | An exhausted top-level frame has no outer parent. The candidate is suppressed while a compatible `DotSegmentRoot` is available, so a forward extension selects that root before discarding the frame; final finish occurs only when no atom remains. |

The partition induction is local to `ResidualPartition::remove_atom`: deleting
an atom either removes an isolated component, retains one connected
remainder, or splits it by reachability from its remaining neighbors.
`attachments` scans all incident unrepresented edges to unvisited atoms, so
the active list equality follows after each entry and restoration. Ring
events only alter bond progress and the active list, and frame completion
uses the recomputation just described. This addresses the earlier
transition-preservation gap for this restricted structural machine; the
finite graph tests check the implementation at its stated bound rather than
serving as the induction proof.

For a forward extension, fix a total structural witness: preserve each
represented bond and selected root, take every currently forced Traversal
bond as Traversal, and take every other unresolved bond as Ring. Give no
additional root to a Traversal component already containing an emitted root.
Give each rootless Traversal component one future root; its ordinary prepared
root domain still permits this choice when no forced Traversal path joins it
to an emitted root. Under this restricted preparation, the rooted-Forest
projector excludes a second emitted root in a forced Traversal component.
Acyclicity and root uniqueness give at most one visited Traversal attachment to each unvisited
Traversal component. At the active atom, close any waiting Ring occurrence
and open every remaining Ring incidence before leaving its ring phase. Each
Ring bond is opened at its first visited endpoint and closed when the other
endpoint becomes active. Its earlier occurrence makes a later atom entry safe:
any additional unrepresented edge to a visited atom would have to be
Traversal, contradicting the unique-attachment fact. The exact residual
attachment list then offers the forced Traversal children. Take a branch
while another child remains, leaving those other incidences forced Traversal;
take inline only for the last child. An empty branch frame returns to its
suspended parent. When a top-level frame is exhausted and unvisited atoms
remain, select a root in a component without a visited Traversal attachment
and take `DotSegmentRoot` before discarding that frame. `FinishSegment` is
offered only after no unvisited atom remains. For this fixed complete Ring
assignment, count both endpoint emissions of an unopened Ring bond and the
remaining endpoint of an open Ring bond. Each Ring event decreases that count.
Atom entries decrease unvisited atoms, including entries that add a frame;
frame completion decreases the number of pending frames, including the active
frame. Thus the lexicographic measure of unvisited atoms, remaining Ring
endpoint emissions, and pending frames terminates the construction.

For the reverse direction, take the total roles and roots from a target
serialization itself. Each emitted ring occurrence belongs to its Ring bond
and each child entry to its Traversal bond. A branch start commits every
remaining parent incidence to Traversal. The target assignment satisfies
that restriction because the grammar forbids a later Ring occurrence at the
sealed parent, leaving those remaining incidences as future child entries.
Each root or child entry also commits that atom's root role exactly as in the
target schedule, so its assignment satisfies those root restrictions and
bond-role domains at every prefix. A target cannot write a later Ring occurrence
at a sealed parent. Its inline child discards that parent only after all
other incidences there are represented. A new root whose Ring edge reaches an
earlier visited atom requires that earlier endpoint to have opened the ring.
These grammar facts and the same attachment partition make each target event
one of the frontier constructors: initial root, Ring open/close, branch or
inline child, dot branch/segment root, branch return, or final segment finish.
The current source owns these at `WriterState::structural_frontier`,
`TraversalState::can_enter_atom`, `can_close_parent_ring_phase`, and
`ResidualPartition::attachments`. The focused live tests check the otherwise
easy-to-miss completion precedence: an available dot-segment root suppresses
`FinishSegment`, whereas an outer branch returns with `CloseBranch`.

The result is correspondence for the restricted structural machine's stable,
successful constructor transitions: such states have a completing structural
schedule, and each valid target schedule has the corresponding constructor
sequence. The case split supplies the arbitrary-size transition induction
under the stated ownership and root-domain conditions; the bounded tests
check implementations of those conditions. It does not establish that every
raw proposed candidate succeeds, that every public token prefix is productive,
or that finite ring labels, chemistry, spellings, stereo, representation
factors, profile restrictions, and RDKit scheduling preserve this result.

## 2026-10-13 tetrahedral mixed ring event order

The mixed ring-order witness fixes A-B/B-C as traversal and A-C/C-D as ring,
opens A-C at A, and enters the hydrogen-bearing center C from B. At C the
already opened A-C can close while C-D can open. The two marker choices select
opposite independently calculated orders, and actual writer choices emit only
the selected first event, then the other event. After each committed local
path, the observed order and role-pattern domains are exact singletons, the
selected layout factor is active, incident roles remain fixed, and an actual
choice can leave the completed frame. This is a bounded event-order check,
not a full suffix-support result.

For a compatible satisfying assignment in the tetrahedral-only Generic
AnyOrder Forest lane, first complete the root assignment for each chosen
traversal-forest component: preserve emitted roots and any already selected
future root; otherwise select one root in a component without an emitted
root, and set its other root variables to zero. The ordinary root-provenance
constraints allow this choice and do not couple root variables to tetrahedral
order. Then freeze the shared traversal/ring roles, bond representation
plans, roots, and complete orders for **all** activated centers, including
suspended and completed ones. Assume the ordinary bond-production owners can
realize those chosen representation plans and an unlimited abstract supply of
ring labels.

Extend only fresh center-local variables, without changing existing selected
values, and select each new marker from its assigned order. Entry and virtual
H occupy their established ligand positions; ring occurrences follow in
assigned order, whether the next event closes an already opened bond or
opens a new one. Traversal children follow in the remaining assigned order,
branching until the final inline child. The writer's ring candidates and
parent-prefix restrictions preserve this shared witness. At completed atom
entry, ring-endpoint, and frame-retirement boundaries, the lexicographic
measure (unvisited atoms, remaining ring-endpoint emissions, live frames
including the active frame) decreases. Entry can add a frame while the first
coordinate decreases. Thus the prescribed structural construction is finite.

At local completion, every incident bond has been represented exactly once.
The four distinct ligands make the final prefix domain a singleton equal to
the assigned order. The surviving assignment makes that domain nonempty; the
active tetrahedral layout factor can then leave only its observed role
pattern, since every incident decision has its assigned traversal or ring
role. These are the exact checks in `validate_active_tetrahedral_completion`.
This is a conditional tetrahedral-only continuation from a compatible joint
witness at stable event boundaries. It does not establish that every public
transition supplies such a witness, nor cover aromatic and directional
composition, other writer profiles, finite-label exhaustion, or unfinished
atom and bond spellings.

## 2026-10-13 MAXIMAL ordinary aromatic single bond spelling

OpenSMILES §3.5.4 requires an explicit `-` for a nonaromatic single bond
whose two emitted endpoint atoms are aromatic. RDKit accepts an omitted
biphenyl bridge and normalizes it to a single bond, so that parse-back is
broader than the stated OpenSMILES spelling rule. This slice keeps the two
correctness questions separate.

The Generic MAXIMAL ordinary Elided single-bond production now consults the
already emitted source mode and restricts the future endpoint's aromatic
mode before publishing omission. When both modes are aromatic, only `-` is
produced. If either endpoint emits an aliphatic spelling, omission and `-`
remain available. The same prepared representation domain excludes Ring00
for the aromatic/aromatic pair while keeping Ring10, Ring01, and Ring11, so
one ring endpoint may omit its bond glyph when the other supplies `-`.
Focused native tests cover four endpoint-mode pairs, the complete ring
placement domain, and a real branch-carrier choice. They do not establish
general aromatic writer support or RDKit writer string parity.

Verification: the focused native test module passed 18 tests. A bounded
offline release build produced a CPython 3.12 wheel, and 16 public prefix
tests passed against its extracted package. A public MAXIMAL prefix witness
accepted `c1ccccc1-c1ccccc1` and rejected the omitted-bridge spelling
`c1ccccc1c1ccccc1`; the same witness passed against the refreshed checkout
extension. Each bounded run reported verified process cleanup.

## 2026-10-13 noncarbon tetrahedral admission

Preparation now admits an RDKit configured tetrahedral tag when its four
represented ligand slots are either four incident graph bonds with no non-graph
hydrogen or three incident graph bonds with exactly one non-graph hydrogen.
Element is not part of this gate. The incident-bond order and virtual-H slot
continue to define the prepared parity reference. Other atom, charge, valence,
and spelling checks remain at their existing boundaries. A configured
three-bond center with no hydrogen still has no fourth modeled ligand and is
rejected; supporting a lone pair would require an explicit fact and order
model rather than treating the vacancy as hydrogen.

Guided public paths for `[Si@H](F)(Cl)Br` and `[N@+](C)(F)(Cl)Br` now check
acceptance in MAXIMAL and COMPACT from original and renumbered input, rejection
of the opposite marker, and RDKit parse-back distinction for these examples.
These establish selected semantic witnesses, not complete tetrahedral language
coverage or RDKit random-writer support. That writer profile still has its
separate narrow C/O admission boundary.

## 2026-10-13 finite ring-label zero-budget transition

The native Forest writer now intersects the prepared Traversal domains of a
new active atom's remaining incidences with the same solver transition that
enters it when all visible ring labels are occupied. It does the same after a
ring opening consumes the final free label. This restriction applies only when
the active atom has no already opened ring to close; such a closure can free a
label before another local ring event. A child or dot branch seals its parent's
ring phase, so releasing a label elsewhere cannot enable a later ring opening
at that parent. The restriction preserves joint role and topology solving; it
does not choose roles one bond at a time or search for a completed suffix.

A native test with a one-label limit opens a ring in one two-carbon component,
then checks that an unproductive triangle branch is refused before branch
text. Its alternative dot path closes the held ring, returns to the triangle,
and reaches acceptance. A separate test checks that a last-label opening still
permits a required Traversal continuation. The 100-label diagnostic uses
actual labels `0..99` across 100 open two-carbon components, then reaches the
same forced triangle ring; it is ignored by default because its prescribed
native event path takes about 45 seconds. Run it from the repository root with:

`python3 scripts/run_bounded_tests.py --wall-seconds 90 --rss-mib 2048 --cpu-seconds 90 --address-space-mib 4096 -- env CARGO_NET_OFFLINE=true cargo test --manifest-path rust/south_star2/Cargo.toml public_ring_label_capacity_rejects_a_forced_101st_opening -- --ignored --nocapture`

These tests use prepared fixed `C` atom tokens and labeled native transitions.
They establish the zero-free-label local obligation for those selected paths,
not public character-prefix union behavior, independent parse-back, or a
general positive-capacity completion theorem.

Qualification used the isolated `06f958af` archive before integration. The
one-label RED test first observed the offered triangle branch ending in typed
label exhaustion; the 100-label RED test observed `next_label: 100` beyond
`maximum_label: 99`. After the zero-budget transition change, the one-label
recovery, last-slot Traversal, and ignored 100-label diagnostic passed. The
focused native ring-label selection passed 11 tests with the diagnostic
ignored. The bounded offline release build produced wheel SHA256
`6e0d90992bd0d18e9201c3e8297b030fabe9cd14a790b6678aba8b1adb247f18`;
its extracted package passed 16 public prefix tests. The ignored checkout
extension was backed up and replaced with the wheel's extension (SHA256
`9926635e982ebca3a293149461c3c4da13a7aa0e6753fda817a58dd4b78683d9`),
then the same 16 tests passed against the checkout. Each bounded runner
reported verified process cleanup.

## 2026-10-13 positive ring-label capacity design

A diamond graph with edges AB, AC, AD, BC, BD exposes a local capacity
contradiction: entering B as a branch child from A seals A's ring phase, so
BC and BD must both be ring edges. With one free label, the current writer
offers that branch and the new child has no semantic successor. This is a
native graph/event witness, not a claim about complete RDKit writer parity.

The intended owner is one latent ring-count factor prepared for each atom.
It counts Ring roles among all incident bonds and has Boolean budget bits.
On entry, when the bound is narrower than the atom degree, the writer fixes
the bits and activates the factor in the same solver transition as the
structural candidate. The bound is free labels plus twice the already waiting
local closures: those closures are themselves Ring roles and each can release
one label before a new opening. With at most 100 visible labels the bound is
at most 200, so eight bits suffice without restricting graph degree. Exact
joint projection must include the count factor, topology and active stereo
factors; local forced-role counts alone miss correlated obligations.

Alternatives considered: one factor per possible bound duplicates the model;
a scalar contextual solver transition changes the solver contract and exact
plan; testing completed suffixes would move capacity outside the live
constraint model. The possible regret is increased exact-search work on
high-degree ambiguous atoms, which needs a focused native work bound before
broad integration. The old zero-budget restriction is replaced by the new
factor at atom entry, leaving one owner for this local capacity obligation.

The positive-capacity implementation uses one latent per-atom factor with
Boolean budget bits. Its typed model boundary rejects nonbinary budget
domains and bit widths beyond the native integer width. Native propagation
handles immediate minimum-count contradictions; mixed exact projection joins
correlated bond roles with topology and other active semantic factors. The
exhaustive reference backend evaluates the same factor.

In the isolated `47448326` archive, the diamond root-branch test was RED
before the factor and GREEN afterward. A two-label test with a real held
ethane ring refused the diamond branch with one free slot, released that slot
through a real closure, restored the branch, and reached an accepted state
through both diamond openings and closures. Correlated-role and unfixed-budget
tests matched the exhaustive oracle. A 70-incident-bond factor test confirmed
there is no 63-bond degree cap. The focused native writer set passed 118 tests
with two ignored; model and count-factor tests passed. Finished bounded runs
reported verified process cleanup.

An ignored actual-namespace diagnostic with 99 real held ethane rings and one
free label passed: the diamond branch was pruned, then a real ethane closure
restored it. Its final bounded run took 44.26 seconds and verified process
cleanup. A prior pre-guard run passed in 82.43 seconds; the early degree guard
avoids constructing prospective traversal contexts when free slots already
cover every incident bond. The small-namespace accepted continuation remains
the detailed recovery witness. These are selected native event paths, not
complete public string-support equality or RDKit writer parity.

The integrated bounded offline release build produced CPython 3.12 wheel
SHA256 `2f53e54f996d9759b32b175e7bc3519dd9671c4c17ca2f13c70143743b065698`.
Its extracted package and the checkout refreshed from its extension each
passed all 16 public-prefix tests. The extracted and checkout extension
SHA256 was `38d04e269971fe578b092dd4efdda386b064518f20dd5cd78ad0d457043c825e`;
the prior checkout extension was backed up. The build and both public runs
reported verified process cleanup.

## End-state roadmap: shared productive witness

MAXIMAL should expose every and only standard-valid serialization of admitted
labelled molecular facts, with every published prefix extendible to an accepted
string. The walker owns traversal events and live labels. The incremental
constraint engine owns unresolved topology, spelling, stereo, and resource
relations jointly; the renderer publishes choices already compatible with
those relations. Writer profiles restrict productions. Neither runtime suffix
enumeration, output filtering, nor a continuation DAG supplies feasibility.

The proof target is one shared satisfying witness through each transition:
every valid complete string induces its facts, event sequence, and assignment
(completeness); every published choice preserves an assignment with a finite
accepted continuation (productivity); every accepted string realizes exactly
the admitted facts (soundness). Constructor coverage must show that every
admitted fact has the required semantic variables and rows, including valid
ring/stereo/resource combinations. Lexical lifting must show that each
compatible abstract event has exactly its permitted token spellings and that
token prefixes preserve the same witness. The current selected witnesses and
count bound alone establish none of these universal claims. Test principled
semantic coverage independently of exact RDKit writer support; keep pinned
RDKit parity as a separate profile obligation. Output-proportional cost remains
unproved and needs measurement or a bound after correctness.

The next concrete milestone is exact local ring-phase feasibility at atom
entry, composed with the existing tetrahedral order authority. For each
assigned order, enumerate the remaining ring incidences in that order. An
incidence whose bond is already open closes and releases one label; any other
Ring incidence opens and consumes one. Every prefix of this local sequence
must fit the actual free-label count. The unordered necessary bound
`total Ring <= free + 2 * waiting closures` is insufficient: even when the
bound is at least degree, an opening before a needed closure can fail. Keep
one ring-resource owner, sharing the established order variable and layout
rows; do not clamp quotas, cap admitted degree, or add an interpreter.

Before implementation, write finite independent relation tests over role
patterns, tetra orders, waiting masks, and free-label budgets, comparing the
relation's projected domains with explicit prefix-capacity arithmetic. Include
degree four, two waiting closures, one free label, where count is vacuous but
order binds. First demonstrate a red failure in the current model, review the
factor design and test, then change runtime code.

### Local ring-phase milestone in the isolated archive

The ordered resource relation for a configured tetrahedral center now shares
its prepared order variable and physical bond mapping with the existing layout
factor. At entry, the waiting mask and actual free-label bits are fixed and
both factors activate in the same solver transition before text publication.
Other atoms retain the unordered ring-count relation. Native and exhaustive
projection agree on selected joint topology/layout/resource cases and on
unfixed waiting/free domains; a finite reference matrix checks three- and
four-incidence role, order, waiting, and budget partitions. A reachable
one-label oxirane-like path previously offered both tetrahedral markers, one
immediately exhausting labels; it now offers only the marker with an accepted
close-first continuation. This closes that local ring-phase obligation for
the current tetrahedral model. Global continuation, constructor and lexical
composition, principled semantic coverage, exact RDKit writer parity, and
output-proportional cost remain unproved.

Focused bounded native tests passed in the archive. A broad native attempt was
interrupted by the CPU limit and is not a suite pass; its process-tree cleanup
was verified. Independent focused review found no remaining actionable issue.
The integrated bounded offline release build produced wheel SHA-256
`407246d09b5fb78cf6f1ebb1bbeb48bf4a98a59e30bfd4d41eddc12d794dd0fa`.
Its extracted extension SHA-256 is
`2f4d56596792c6404bdf3350512dbb12b2b0e14e95170f3b6094746862fef9b9`.
The extracted wheel and refreshed checkout extension each passed all 16 public
prefix tests. The previous ignored checkout extension was backed up at
`/tmp/grimace-ring-phase-checkout-core-before.so` before replacement. The
build and both public runs reported verified process-tree cleanup. Focused
tetrahedral, finite relation, joint native/exhaustive, and existing ordinary
capacity tests passed under bounded runs.

### Future two-center ring schedule obligation

A finite diamond exposes a boundary beyond local entry factors. Let A-B, A-C,
A-D be traversal edges, and B-C, B-D be ring edges. A and B each have three
bond ligands and virtual H. With one live ring label, a depth-first order that
starts at C can finish only as C, B, D: C opens B-C, B closes B-C before opening
B-D, and D closes B-D. The symmetric D, B, C order is also feasible. Starting
at B needs two openings there; C, D, B and D, C, B need two simultaneously
open labels before B. The relation enumerates all 24 A orders and 24 B orders,
checks both tetra layouts, and updates the two ring-edge identities after each
event. Its finite test retains exactly the two feasible order pairs, in opposite
A marker parities. Run the selfcheck with
`python3 scripts/run_bounded_tests.py --wall-seconds 90 --rss-mib 2048 --cpu-seconds 90 --address-space-mib 4096 -- env CARGO_NET_OFFLINE=true cargo test -p grimace-south-star2 --lib finite_two_center_diamond_schedule_requires_close_before_next_open`.

An opt-in native regression uses this four-atom prepared graph with C and O
terminal spellings, fixes the five edge roles as above, limits the test label
namespace to one, and roots at A through normal writer transitions. For A's
`[C@H]` marker, the writer currently publishes the `(` branch toward O even
though that marker fixes the D, C, B order; the independent complete schedule
relation finds no continuation for that branch. The other parity has the
compatible C, B, D schedule, so the reference detects overpruning as well as
overpublication. The regression is ignored while red; run it explicitly with
`python3 scripts/run_bounded_tests.py --wall-seconds 90 --rss-mib 2048 --cpu-seconds 90 --address-space-mib 4096 -- env CARGO_NET_OFFLINE=true cargo test -p grimace-south-star2 --lib tetrahedral_diamond_first_child_respects_future_ring_schedule -- --ignored`.
The finite reference selfcheck passed its focused bounded run. The opt-in
regression failed its stated assertion (`offered=true`, `viable=false` for
`[C@H]` and first A-D), not a time or resource limit; both runs reported
`verified_no_live_members` cleanup.
This is a conditional internal productivity defect for the forced roles and
one-label test namespace. It does not establish a default 100-label public
counterexample, all-molecule incompleteness, or exact RDKit parity.

The next owner needs one shared future event-schedule and peak-live-label
relation over the same topology and tetra orders. An individual center's ring
budget cannot express which ring identities cross child subtrees or when a
future center releases a held label. A scalar peak summary per child is
insufficient if ring edges cross those subtrees; any composable summary must
retain the boundary ring identities and their live state. Derive the fixed
traversal-tree schedule relation first, prove its composition at child and ring
events, then connect it to the incremental solver. Do not use runtime suffix
enumeration or a molecule-specific branch guard. Lexical prefix lifting,
unfixed topology, and full admitted-graph completeness remain separate proof
obligations.

### Fixed-tree future ring schedule slice

The native writer now activates one latent event-schedule factor when a candidate
fixes every bond role to one spanning traversal tree, has one selected root, and
requires more ring identities than the regime's total label namespace. A
non-tetrahedral atom must have at most one traversal child and one ring
incidence; existing tetrahedral order variables and entry layout factors own
all other local order constraints. The factor visits the fixed rooted tree in
those orders, opens or closes each ring identity at its endpoint occurrence,
and rejects any event cut whose live-ring count exceeds the namespace. Root and
capacity bits are derived from the unpublished writer candidate, then activated
with all selected entry layouts before the candidate token is offered. Native
exact-component search joins this relation to the same order variables and
projects support; the factor itself checks only complete assignments. This is
finite shared constraint search, with potentially exponential order choices;
it does not run a writer suffix walk or construct a support DAG.

The formerly ignored one-label two-center diamond regression now passes. An
unseeded writer test reaches the fixed-role guard when the first branch seals
the root ring phase; it checks factor activation and the surviving sibling
order against the independent 24-by-24 event reference. A separate test checks
the composed layout and schedule relation against that reference for every
order pair, and a native/exhaustive test compares projected domains. The
disconnected dot-branch test confirms that case remains outside this factor.
These were bounded offline runs with verified process-tree cleanup; logs are
`/tmp/grimace-fixed-schedule-focused.log`,
`/tmp/grimace-fixed-schedule-regression.log`,
`/tmp/grimace-fixed-schedule-dot-guard.log`, and
`/tmp/grimace-fixed-schedule-check.log`.

The reduced one-label test namespace causes test builds to prepare the latent
factor on this small graph; the default production namespace has 100 generic
labels or 99 RDKit labels, so this fixture does not prove a default public
counterexample. Unfixed roles, a selected forest with multiple roots, and
non-tetrahedral atoms with more than one child or ring incidence receive no
schedule pruning. A branch published before the roles become fixed may still
lack this future proof. Lexical prefix lifting, broader forest schedules, and
full admitted-graph completeness remain separate obligations.

The integrated bounded offline release build produced CPython 3.12 wheel
SHA-256 `2d349b16455b27845cdd0c3043116d3da4e2a9a255bee78f8575ae610aac03af`.
Its extracted extension SHA-256 is
`0001e078d9bedac2a2773de0dc8a9eba8fb6d124d3bc28df38ba35c0fce22af3`.
All 16 public prefix tests passed against both the extracted wheel and the
refreshed checkout extension. The previous checkout extension is preserved at
`/tmp/grimace-fixed-schedule-checkout-core-before.so`; the checkout extension
bytes match the verified wheel extension. Build and public-test runners all
reported `verified_no_live_members` cleanup. Evidence logs are
`/tmp/grimace-fixed-schedule-wheel-build.log`,
`/tmp/grimace-fixed-schedule-wheel-public.log`, and
`/tmp/grimace-fixed-schedule-checkout-public.log`.

### Ordinary-incidence fixed-tree schedule slice

The fixed-tree schedule now shares one Boolean precedence fact for each pair of
incident bonds at an ordinary atom. A latent local factor rejects directed
cycles, and the writer records an ordinary atom's entry incidence before its
other incidences, then retains selected-event precedence over every remaining
incidence even after a branch frame is popped. On fixed roles, the schedule
activates these facts with the tetrahedral layouts and projects the future
ring-label peak through the shared exact solver. There is no 64-value local
permutation cap: storage is O(sum of squared atom degrees), and the local
fixed-direction cycle check is O(degree squared). Exact search over the shared
orders can still be exponential.

The one-label subdivided-diamond conditional regression is now green: with
A-X-B as the long traversal arm, A-C/A-D as the others, and B-C/B-D as rings,
the ordinary root does not offer X first and retains both viable short-arm
starts. A finite independent 6-by-6 event relation establishes that expectation.
An unseeded writer test in the same one-label namespace reaches the original
four-atom diamond naturally: after A chooses the C branch, all roles fix, the
factor activates, B-before-D is projected, and the ordinary order facts persist
after the C branch pops.
An ordinary-only cycle graph prepares the test factor. Four-incidence testing
checks all 64 tournaments against 24 orders and a partial transitive projection
against the exhaustive oracle; degree five checks construction beyond 64
possible orders.

Bounded offline `cargo check -p grimace-south-star2`, focused `ordinary_`, and
focused `diamond_` Rust lanes passed with verified process-tree cleanup. Logs:
`/tmp/grimace-ordinary-schedule-check.log`,
`/tmp/grimace-ordinary-schedule-ordinary-focused.log`, and
`/tmp/grimace-ordinary-schedule-diamond-focused.log`.
Unfixed topology, selected forests, lexical prefix lifting, and full admitted-
graph completeness remain separate obligations. The one-label fixture does
not establish a default 100-label public counterexample.

After integration, the bounded `ordinary_` and `diamond_` native lanes passed
9 and 8 tests. A
bounded offline release build produced CPython 3.12 wheel SHA-256
`6540d7bbce7ef89cacfee7ee5c8c4a1d6b2cce9ee761230da3c76e8095238974`;
the extracted extension SHA-256 is
`c2442dd2f76310e1c0063b20aab096c230b66c042364c469157e562567ae7f07`.
All 16 public prefix tests passed against both the extracted wheel and the
refreshed checkout extension. The prior checkout extension is backed up at
`/tmp/grimace-ordinary-schedule-checkout-core-before.so`; the current checkout
extension matches the wheel bytes. Each bounded run verified process-tree
cleanup. Logs: `/tmp/grimace-ordinary-schedule-integrated-ordinary.log`,
`/tmp/grimace-ordinary-schedule-integrated-diamond.log`,
`/tmp/grimace-ordinary-schedule-wheel-build.log`,
`/tmp/grimace-ordinary-schedule-wheel-public.log`, and
`/tmp/grimace-ordinary-schedule-checkout-public.log`.
### General event schedule after the disconnected spectator witness

The one-label subdivided diamond remains impossible when an isolated O
spectator is added: A-X first makes B open both B-C and B-D before C or D can
close either, while C-first and D-first each have a complete schedule. X is N
in both focused fixtures, so B has distinct N, C, O, and H ligands. The
disconnected test is a conditional private writer regression, not default
public or RDKit-parity evidence.

Replace the connected fixed-tree schedule with one event relation over the
existing topology roles and ordinary/tetrahedral order variables. Each atom
has a visit rank, incoming syntax kind (top-level root, bonded branch, bonded
inline, dot branch, or dot continuation), and a conditional branch-end rank
encoded by monotone unary bits. Fixing bit `r` asserts that an open branch
contains reserved visit `r`; a closed branch fixes its entire end pattern.
The rank-ordered fold reconstructs source frames and chemically incoming
bonds; no independent parent or root authority is needed. It checks traversal
orientation, ring-before-child local order, conditional tetrahedral layout,
and peak live ring identities. Forest roots must agree with the existing
topology root variables; segment-mode resource preparation uses the safe edge
count bound rather than connected-tree cycle rank.

The writer binds rank and syntax kind when it reserves an atom, including
before `(` or dot text, raises every open branch end to include the new rank,
and fixes a branch's end in the same checked transition that publishes `)`.
Its return frame carries the branch root's identity. Native search groups ambiguous bond roles before earliest-rank and
branch-end choices, then checks a forced source prefix, closed branch
intervals, local incidence order, and live ring cuts before the complete
event predicate. The writer's candidate transition needs one joint witness
across the active component and keeps conservative bounds for future facts;
every structural event candidate is checked again after its own facts are bound;
pending atom spelling inherits that checked state. The
solver names the conservative accessor `domain_bound`; an explicit exact
`transitioned` call projects all active event variables, including when the
source is a feasible writer state and the transition adds no new fact. The selected
tetrahedral entry uses the same immutable layout row allocation as its
prepared layout factor, so the event relation does not reconstruct future
layout rules. The search does not walk writer suffixes or materialize output.
Its cost can still be exponential. The connected/disconnected regressions,
held-label dot rescue, unresolved-role projection, early-return rejection,
and exhaustive three-atom dot grammar comparison are required proof points.
A fixed-forest guard or an independent parent selector would preserve the
wrong ownership boundary.

The prepared schedule is conditional on the actual visible ring-label
namespace: the normal generic MAXIMAL and COMPACT writer uses indices 0..99,
and the RDKit policy uses 1..99. A focused test constructor supplies a smaller
namespace before preparation and initializes its runtime labels to the same
maximum. Forest preparation uses `E` as a safe possible-ring upper bound;
the same path skips the event factor when that bound fits the namespace. This
avoids activating a 12-atom event relation in ordinary aromatic-mode tests
whose normal namespace has ample labels. The disconnected two-edge plus
isolated-atom test prepares a factor with a one-label namespace; the same
graph skips it with the normal namespace.

Focused offline native qualification: production `cargo check`,
`event_schedule` 5/5, `diamond_` 8/8, `ordinary_` 9/9,
`tetrahedral_` 32/32, and the preparation-boundary test pass under the
90-second bounded runner with verified process-tree cleanup. A four-atom independent syntax-tree assignment checks
feasible-to-exact no-op and redundant transitions against the exhaustive
solver for all event and physical variables; an exhaustive three-atom dot
grammar comparison covers event encodings. The visible branch-close test
checks that the returned `)` state fixes the branch end bits. Initial six-atom
spectator search diagnostics before root/kind propagation counted 16,363
mixed branches for the isolated O root and 114 for A; after propagation, the
eight diamond regressions together run in under a second. These are local
diagnostics, not asymptotic bounds or full public-support proof. The event
relation is prepared only when the ring bound requires it; complete generic
MAXIMAL support and default RDKit parity remain separate obligations.

Integrated conditional-entry qualification at `d03e59a3`: the 13 reviewed
candidate paths were copied byte-for-byte from
`/tmp/grimace-conditional-entry-d03e59a3`. Bounded offline native filters
passed: `event_schedule` (5), `diamond_` (8), `ordinary_` (9),
`tetrahedral_` (32), and the exact
`event_preparation_uses_the_actual_writer_label_namespace` (1). The earlier
short namespace filter matched zero tests; the exact rerun passed. The offline
release wheel SHA-256 is
`abe73854e38d26db5326dc4e46b7dcce1dee5164036a21502728174c4748ecec`;
its extracted extension SHA-256 is
`b816afcdd7404b54f88d592f68f5df4582619b141e2cfd6ab94c26a27a496661`.
The extracted wheel and refreshed checkout extension each passed all 16 public
prefix tests. The previous checkout extension is backed up at
`/tmp/grimace-conditional-entry-integrated-d03e59a3/checkout-core-before.so`;
the checkout bytes match the wheel extension. The maintained
`tests.run_south_star_semantics -q` attempt hit its 120-second wall limit
(exit 124), so this suite is **not qualified** by this run. Every bounded run,
including that timeout, reported `verified_no_live_members`. Persistent logs
and the wheel are under `/tmp/grimace-conditional-entry-integrated-d03e59a3/`
(`native-*.log`, `wheel-build.log`, `wheel-public.log`,
`checkout-public.log`, and `south-star-semantics.log`). This local event-schedule
slice still does not prove lexical prefix lifting, broader forest schedules,
full admitted-graph completeness, or exact RDKit writer parity.

Follow-up qualification isolated the maintained semantics runner: it aggregates
legacy South Star 1 tests, so its 120-second timeout above remains unresolved
for that full suite. A focused South Star 2 semantic selection
(`test_standard_smiles_semantics`, `test_compact_language`,
`test_leading_dot_tetra_semantics`, and `test_kekule_semantics`) passed 37 tests
in 49.030 seconds with verified cleanup; log:
`/tmp/grimace-conditional-entry-integrated-d03e59a3/south-star2-focused-semantics.log`.
A verbose 90-second legacy run reached the online ring decoder test; that test
passed alone in 12.944 seconds. The legacy fact-relabeling test reproduced two
implicit-H mismatch errors with both the current and backed-up pre-change
extension. Logs in the same directory are
`south-star-semantics-verbose-diagnosis.log`, `online-ring-current.log`,
`fact-relabel-current.log`, and `fact-relabel-old-abi.log`. These observations
show no new failure in the focused South Star 2 selection, but do not qualify
the full maintained semantics suite.

## Bounded per-state work diagnostic at `f4e79c82`

An ignored native diagnostic in an isolated archive measured one complete
`NonStereoWriterState<NativeSolverState>::choices()` call after selecting a
degree-one carbon root. Each added remote gadget has three labelled carbon
atoms and one triangle; atom degree remains at most three. The one-label test
namespace activates the Forest event schedule at both sizes. The root's local
graph incidence count stays one, while the number of permitted remote dot
choices can grow; the actual returned transition count is therefore reported.
The helper that selected the root was measured separately and considered more
than one possible root. Graph construction and the later model-counting loop
were outside the preparation timer.

| Remote triangles | Prepared atoms / edges | Variables / factor references | Unary end bits | Root selection: paged reads / search nodes | One post-root `choices()`: raw outputs / distinct observed `(text, state)` projections | Post-root search nodes / factor revisions / paged reads |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 4 / 4 | 55 / 71 | 16 | 50,763 / 92 | 9 / 9 | 147 / 938 / 145,077 |
| 2 | 7 / 8 | 124 / 155 | 49 | 1,732,715 / 420 | 11 / 11 | 3,072 / 52,126 / 25,331,088 |

During those same post-root calls, sibling restrictions proposed rose from
524 to 48,605; 236 and 3,420 successfully propagated and were pushed, while
133 and 3,052 child states were actually popped. The search eagerly propagates
siblings before it explores them, so even a first feasible witness can incur
work on alternatives that are never visited. Structural candidate attempts
were 8 and 11, including two contradictions at each size. The source state
recorded 538 and 65,166 copy-on-write domain value-page copies during the
post-root calls; index-node copies were 0 and 48,625. The counters named
“paged reads” include every `PagedStore::get` for domains, active factors, and
factor adjacency, not just variable inspections. Factor revisions count
propagation-queue pops, search nodes count DFS pops, and the sibling counts
distinguish attempted restrictions, successful propagated pushes, and actual
pops. The reported model payload lower bounds, 724 and 1,612 bytes, include
only variable-domain and factor-variable-reference payloads; they are not
total owned allocation sizes.

An acyclic 102-carbon chain with 101 edges and the normal Generic MAXIMAL
namespace of 100 labels prepared the EventSchedule in 34 ms, producing 11,732
variables (10,404 unary end bits) and 12,137 factor references. This follows
the Forest safe bound `possible_ring_count = E`, even though the graph has no
cycle. The diagnostic stopped after preparation for this case: no default-size
root selection or transition cost was measured. The k-gadget family enters
the same default-namespace preparation path at k=26 (`E=104`), but the small
case timings cannot be extrapolated to that size.

The complete bounded k=1,2 run and the separate 102-atom preparation run both
passed and reported `verified_no_live_members`. Their durable logs are
`/tmp/grimace-work-bound-f4e79c82/work-k1-k2.log` and
`/tmp/grimace-work-bound-f4e79c82/work-default-chain-prep.log`. The named
commands, run from the isolated archive, were:

```sh
python3 scripts/run_bounded_tests.py --wall-seconds 45 --rss-mib 2048 --cpu-seconds 45 --address-space-mib 4096 -- env CARGO_NET_OFFLINE=true GRIMACE_WORK_SIZES=1,2 cargo test -p grimace-south-star2 --lib work_diagnostic_remote_triangle_chain -- --ignored --nocapture
python3 scripts/run_bounded_tests.py --wall-seconds 45 --rss-mib 2048 --cpu-seconds 45 --address-space-mib 4096 -- env CARGO_NET_OFFLINE=true GRIMACE_WORK_MODE=chain_prep GRIMACE_WORK_SIZES=102 GRIMACE_WORK_MAX_LABEL=99 cargo test -p grimace-south-star2 --lib work_diagnostic_remote_triangle_chain -- --ignored --nocapture
```

All 9 and 11 returned choices had different `(text, observe_raw())`
projections, so these measurements contain no exact duplicate full states;
the projection omits some solver facts and cannot establish full-state identity
when two projections match in another case. The diagnostic does not prove
productivity or completeness of the returned transitions, nor an unavoidable
asymptotic lower bound. The measured implementation cost comes from repeated
global rank/end revision and search, amplified by eager sibling propagation.
A narrow next mechanism correction is lazy admission of sibling restrictions,
with the same per-state counters and independent semantic checks. Even if that
reduces unvisited-sibling work, the global event fact volume and repeated scans
remain to be measured against the governing affected-facts-and-outputs bound;
no such bound is established here.

### Deferred exact-search siblings (2026-10-14)

The mixed finite-domain search now queues a parent state and each ordered child
restriction, propagating a child only when DFS explores it. This keeps the
previous branch order, first-witness stop, and exact projection union. The
diagnostic's `sibling_proposals` still counts attempted restrictions;
`sibling_offered` counts queued alternatives, including those never attempted.
The k=1,2 probe offered the same 524 and 48,605 alternatives as before, but
attempted 293 and 46,942, respectively. Successfully propagated children
fell from 236 to 133 and 3,420 to 3,052; explored children remained 133 and
3,052. Paged reads fell from 145,077 to 90,087 and 25,331,088 to 24,582,951.
The observed choice counts, rejection counts, distinct `(text, observe_raw())`
projections, and search-node counts were unchanged. Elapsed choice time was
24 ms versus 17 ms at k=1 and 2,946 ms versus 2,486 ms at k=2 in these runs;
those timings do not establish a speedup.

The post-change bounded run used the same k=1,2 command above from the
isolated archive at `/tmp/grimace-lazy-children-eb108c05`, with its log at
`/tmp/grimace-lazy-children-k1-k2.log`; it reported
`verified_no_live_members`. A direct ignored work-count test confirms that a
first witness leaves its queued sibling unpropagated. Focused native, event,
and diamond correctness tests passed; a compact active ring-capacity model
also matched exhaustive projection across both role assignments, and a
locally consistent pre-exact search state exercised a contradictory first
child, a later witness, and full contradiction. These observations measure a
bounded mechanism improvement, not the governing end-to-end work bound.
Global EventSchedule fact preparation and repeated rank/end revision remain
open.

## 2026-10-15 proposed incremental productive-transition refactor

This is the current design target, not a result established by the EventSchedule
implementation or the bounded diagnostics above. Replace whole-future traversal
planning in next-event admission while retaining exact, jointly solved chemical
constraints. A finished MAXIMAL walker must offer all and only productive
labelled `(text, successor)` transitions for standard-valid serializations of
its admitted molecular facts. Productive means that a finite, profile-valid
accepted continuation exists. Different successors with the same text remain
different transitions. An accepted string must realize the intended labelled
graph and stereo assignment. RDKit writer parity is a separate profile claim.

The finished runtime has one authority for actual traversal and one shared
semantic consistency authority. It has no all-atom future visit ranks, entry
kinds, or branch-end bits; no complete hypothetical event schedule relation or
search; and no global schedule fallback, graph-specific fast path, support
subset, suffix or reachable-state enumeration, proof/count DAG, or retained
continuation plan. The existing Event model may remain privately during the
migration and later as test-only differential evidence. These are falsifiable
end invariants: search the production construction and transition path for the
displaced representations and compare each returned labelled transition with
an independent finite productive grammar on bounded instances.

| Owner | Facts it may retain | Consequence |
| --- | --- | --- |
| Prepared molecule and profile | Immutable labelled graph, chemistry, incidence adjacency, semantic relation definitions, and rules for permitted structural and lexical events | An ordinary graph or profile change enters through these definitions, without a second traversal inventory. |
| Live walker | Actually visited atoms and written bonds; active atom and emission phase; ordered return stack; still-writable incidences; pending actual event; held ring obligations with bond, endpoints, label, emitted endpoint spelling and direction; occupied labels | An event changes one authoritative history. Residual graph regions and boundary attachment opportunities are derived from these facts and immutable adjacency, not copied history or an opaque cache. |
| Shared semantic solver | Unresolved role, incidence order, tetrahedral parity, carrier direction, endpoint compatibility, and resource correlations over shared identities | Domains alone can lose correlations. The solver maintains the applicable joint relations and exactly resolves a genuinely affected semantic component when propagation is insufficient. |
| Renderer and client | Already admitted text commitments and their successor states | Rendering cannot add an unverified spelling; text grouping cannot merge distinct successors. |

The proposed compact state is a candidate representation, not a sufficient
state theorem. In particular, a residual region's boundary opportunities may
interact through role, order, direction, ring labels, and dot placement. A
summary that records only each opportunity's marginal domain can accept a
combination that no shared assignment realizes.

### Atomic event and admission contract

For each source-local legal event, fork the current compact state and form one
delta containing every actual structural, lexical, and semantic commitment.
Apply it atomically, derive the affected residual interfaces, and propagate and
if needed exactly solve their shared relations. A contradiction discards this
candidate; backend failure aborts the choice batch with an error. Before
publication, the still-missing completion criterion must prove that this
successor has a finite continuation. Only then render its committed text and
publish `(text, successor)`. Structural capacity and chemistry cannot be
certified independently when they share a role, order, endpoint, or carrier.
There is no opaque runtime `can_finish` call that conceals future-walk search.

| Actual event | Walker commitment | Joint relation update |
| --- | --- | --- |
| Enter root, bonded child, or dot segment | Visit the selected atom; record incoming incidence or actual dot; advance active atom and emission phase | Restrict incoming role, component/segment attachment, incidence order, and profile rules together. |
| Open or close a ring | Write the chosen bond endpoint and label; create or discharge the held bond/endpoints/spelling/direction obligation; update occupancy | Join role, endpoint compatibility, label capacity, local order, tetra parity, and directional carrier relations on the same identities. |
| Push or pop a branch | Push actual return frame or restore it; consume the visible parenthesis | Update still-writable incidences and boundary opportunities; test whether the return strands a required attachment. |
| Commit bond, atom, or stereo spelling | Advance the pending emission phase with the exact token spelling | Restrict representation, parity, carrier, and profile lexical rules before its text is offered. |

The event table describes ownership and effect ordering, not a completed
algorithm. Terminal acceptance requires finished emission, discharged stack
and ring obligations, all graph facts represented, and a jointly consistent
semantic assignment under the active profile.

### Missing extension lemma and first proof gate

The central missing lemma is an effective criterion over the *live interface*
and the residual relations derived from it: it must hold if and only if at
least one finite sequence of permitted actual events reaches acceptance while
realizing one common assignment satisfying all applicable semantic relations
and the active profile. Its formulation must specify the boundary facts, joint
correlations, event update rules, and why each admissible residual obligation
can be discharged in a
finite order. Prove both directions and show that every actual event updates
the criterion without enumerating or retaining a future traversal. A
constructive mathematical proof or test witness may contain a future order;
ordinary runtime admission may not. Calling a state Markov, finding a
satisfiable semantic domain, or showing equivalence to EventSchedule is not
the extension lemma. If the only exact criterion is the entire future
traversal under another name, this architecture has not been obtained.

First build an independent **test-only** finite labelled event grammar with a
fixed small structural vocabulary, lexical representatives, and label
budgets. Bound atom visits, bond-endpoint occurrences, and branch events so
the raw transition graph is finite; backward marking in that graph cannot
become production admission. The grammar applies local syntax and incidence
legality but includes structurally legal incomplete raw states that have no
accepting completion. Mark productive raw states backward from accepted states
in this finite
reference, and compare *each* proposed labelled `(text, successor)` transition
against productive successors, checking false acceptance and false rejection.
Do not compare only counts or text sets. A raw legal dead end is a useful
negative witness; an already illegal inline move is not a reachable
production state. The production API's reachable successors must be
productive, while the raw reference deliberately has a wider state space.

Anchor the matrix in ordinary, tetrahedral, ring, and branch cases: the
subdivided diamond with one label must reject the X-first move and retain its
C-first and D-first siblings; its ordinary root has a tetrahedral B. An
isolated spectator cannot rescue that move, while a separate case with a real
held label can restore a branch after returning through `)`. Nested returns,
label reuse, and interacting incidence orders must challenge independent
local tests. Existing anchors in
`rust/south_star2/src/nonstereo_writer_tests.rs` are
`ordinary_root_subdivided_diamond_first_branch_respects_future_ring_schedule`,
`disconnected_spectator_cannot_rescue_subdivided_diamond_first_branch`,
`releasing_a_real_held_label_restores_diamond_branch`, and
`tetrahedral_diamond_first_child_respects_future_ring_schedule`; the new
reference must assess their labelled transitions independently. Keep the
existing Event relation as another differential witness, not the sole authority.
This finite harness cannot prove unrestricted MAXIMAL chemistry or lexical
completeness. Later enumerations or probes use the bounded runner, with slow
ones behind a named diagnostic runner; exhausted budgets are unqualified
failures, never silently truncated evidence or grounds for automatic budget
increase.

### Authority replacement and cost gate

First inventory the live facts and event deltas, then formulate and try to
falsify a concrete compact completion criterion on the adversarial cases.
Prove the general structural extension case and its composition with the
shared semantic relations and profile lexical rules before treating the
private implementation as a replacement. The first implementation slice can
introduce the compact facts and independent finite harness; it is not the
completed refactor and must not quietly reduce productivity. Switch production
authority once exact labelled-transition comparison and the extension proof
justify it, then delete the displaced runtime schedule machinery. If the
lemma fails, stop this replacement and report the counterexample and required
new fact; do not preserve dual authorities as an end design. Retain finite
witnesses and regressions after the switch.

The deletion map is `nonstereo_writer.rs` preparation near line 831 (future
rank/kind/end allocation), its `event_schedule_transition` near line 3592 and
the second semantic transition, `event_schedule.rs` complete-event relation,
and `native.rs` schedule activation near line 731, `event_rank_branches` near
line 832, plus search,
search, and model plumbing. Remove ordinary order variables only where they
exist solely to reconstruct hypothetical schedules; keep real semantic order
constraints. Recheck current locations when editing.

The work target remains per non-determinized state and includes preparation,
fork/copy cost, every failed candidate, relation revision, and exact search,
as well as returned transitions. A global scan of genuinely affected residual
facts is not automatically disallowed, but its dependencies and cost must be
stated. No output-proportional bound follows from this proposal. Optimizing
EventSchedule leaves whole-future planning in authority; local consistency
alone can publish dead ends; compact region summaries remain unproved until
they preserve shared correlations. A graph algorithm over live obligations
may be the right criterion, provided its state and cost are honest. As an
evolution check, a profile that forbids dot and changes the ring-label
namespace must flow through the same event model and joint relations, with
early admission effects rather than an output filter.

Qualification of the frozen Rust candidate from baseline `eb108c05d177af58d502a27432df3a5494706d7d` used `/tmp/grimace-lazy-children-eb108c05`; the build-time tracked-source comparison found only this note, `native.rs`, `work_diagnostic.rs`, and `nonstereo_writer_tests.rs` changed, with `native.rs` SHA-256 `96b291bbd6581388f6226f566ef5d8673cbfb1351afb1f82e39f6f843bc54be3`. The bounded offline release wheel at `/tmp/grimace-lazy-qualification-eb108c05/dist/grimace_py-0.1.10-cp312-cp312-manylinux_2_34_x86_64.whl` has SHA-256 `ae3c090b9e1d421fef384231ab48de60622d5c15040fb4f1c8071c9a53973505`; its sole `_core` extension has SHA-256 `10b8fd9e896e9600c8819ba01162d6ce7053a0a5a500d8d14f25cce388aba8f9`. Import checks resolved `grimace` and `_core` to the extracted wheel, and all 16 public-prefix tests passed there. The prior ignored checkout extension (SHA-256 `b816afcdd7404b54f88d592f68f5df4582619b141e2cfd6ab94c26a27a496661`) was backed up at `/tmp/grimace-lazy-qualification-eb108c05/checkout-core-before.so`; the refreshed checkout extension matches the wheel bytes and resolved from the checkout package. The focused South Star 2 semantic selection (`test_standard_smiles_semantics`, `test_compact_language`, `test_leading_dot_tetra_semantics`, `test_kekule_semantics`) passed 37 tests. Every bounded build/import/test run reported `verified_no_live_members`; logs are `/tmp/grimace-lazy-qualification-eb108c05/{wheel-build,wheel-import,wheel-public-prefix,checkout-import,checkout-south-star2-semantics}.log`. This qualifies the represented public and semantic paths, without establishing a speedup or the broader work bound.

### Initial live-obligation proof gate (private candidate)

The current traversal store already records visited atoms and each bond's
unrepresented, traversed, open-ring, or closed-ring progress; active and saved
writer frames record actual branch returns. The non-stereo layer separately
holds occupied ring-label slots and pending lexical emission, while the shared
solver carries role, incidence-order, and stereo relations. These facts should
be the starting live inventory. `PreparedNonStereo` additionally allocates
`EventVisit` rank, entry-kind, and branch-end variables and an all-future event
factor. Those variables are the displaced authority, not live facts.

A test-only, fixed-role finite raw event grammar now separates legal local
moves from productive successors for small ordinary examples. It has explicit
visited/bond progress, return stack, label occupancy, and ring-before-child
phase, and reverse-marks accepted states. For the one-label subdivided diamond,
X-first is locally legal and reaches an incomplete raw dead end; C-first and
D-first are the exact productive labelled branch-and-atom structural successors.
A production `observe_raw` projection compares those `(macro text, structural
successor)` pairs at that root; a two-label variant restores X-first. Changing
X-B from traversal to ring and B-C from ring to traversal also restores
X-first with one label, by opening X-B, returning to A, entering C-B, closing
X-B, and reusing its label for B-D. A separate five-atom tree case asserts
that a productive state retains two ordered branch returns. This is a bounded
structural gate only. It does not model tetrahedral or directional relations,
dot rescue, all lexical representatives, or unrestricted MAXIMAL support, and
it is not the missing compact completion criterion.

An unseeded production witness reaches the actual A-X branch without
restricting future bond roles. There, X-B traversal is absent while X-B ring
opening is offered; the test compares normalized complete ring-token
structural projections with the union of the three compatible fixed-role raw
grammars. It separately retains production bond-domain vectors because the
same text and structural projection can have distinct future endpoint
spelling plans. An actual continuation returns to A, enters C-B, closes X-B,
opens B-D using the released slot, returns again to A, enters D, closes B-D,
and accepts. This still does not compare full semantic successor identities
against a model of all those relations; the structural projection is only the
explicitly scoped first gate.

In particular, a residual boundary must retain the identity of held ring
bonds and their joint incidence-order alternatives. At a center with ring
incidences BC then BD and one slot, arriving with BC held permits close-then-
open, while arriving with BD held requires opening BC before closing BD and
exceeds the slot. The same occupied-label count and residual connectivity do
not determine productivity. An exact candidate must jointly test this order
with role and stereo assignments; no replacement of the runtime event factor
is justified until the general extension lemma is established.

A second, delayed obstruction is witnessed on ordinary atoms
P,A,H,B,C,D with edges PH, PA, AH, AB, BC, BD, CD and one label. After
P opens PH and enters A, A-H first closes PH, returns to A, and permits
an accepted A-B-C-D continuation that reuses the freed label for CD.
A-B first would strand the required A-H attachment behind a B branch while
PH occupies the sole label; the B-C-D triangle needs a ring endpoint before
that branch can return. The current Event relation rejects A-B at A. In a
test-only clone that merely disables activation of this latent factor while
retaining its prepared variable namespace and all other solver relations,
A-B is offered, but the entered B state has two candidates semantically
rejected and reports `AllCandidatesSemanticallyRejected`. This differential
diagnostic shows a delayed local-only failure for that exact reachable prefix;
it is not an independent grammar proof of the six-atom case. It does not
establish a general compact completion criterion or justify disabling the
factor in production.

### 2026-10-15 actual-event transaction cleanup

From the `994027e0` proof baseline, the private writer now adds the observed
EventSchedule restrictions and activation to the same candidate solver
transition as the structural, lexical, and ring-resource commitments. The
schedule delta depends on the source walker and the source schedule activation
bit, so the second semantic transition was redundant. Repeated variable
restrictions are intersected by `WriterState`; a conflicting intersection is
a candidate contradiction. Backend failures still abort the choice batch.
The source state remains immutable.

The focused root regression checks one solver call and equality of every
semantic domain against the former two-transition path for a tetrahedral
diamond root. On the combined source, four finite live-obligation reference
tests and that regression passed, as did the existing complete-walk tests for
coupled tetrahedral, selectable tetrahedral/directional, and directional ring
fixtures. Every bounded offline run reported `verified_no_live_members`;
logs are under `/tmp/grimace-live-event-994027e0/`. These bounded witnesses
do not prove general domain equality or the compact completion criterion.
The legacy EventSchedule remains production admission authority until that
criterion and labelled-transition equivalence are established.

Artifact qualification used the frozen combined archive at
`/tmp/grimace-live-event-994027e0`, compared with tracked baseline
`994027e0fb553637acbf8af05ab1503fe6a3e24a`: only this note,
`nonstereo_writer.rs`, and `nonstereo_writer_tests.rs` differed. The two Rust
files had SHA-256 `909a62fd0e12443bbbb8a5d40e8161f16b097570539b96e37b35ff71b357cd5a`
and `0190a7ac8f1b30806a3b2317b9f61779858be182f240156a8bc9a3cab05cf572`.
The bounded offline release build produced CPython 3.12 wheel SHA-256
`34768e374fd5abf9b3a6cf034c16038643d7ce7b789521d2f5932647db3493a7`;
its sole `_core` extension had SHA-256
`9b29df18be5aca390418c66ca9de7344a9501f4e951362e6d63aabbf702e32e6`.
Imports resolved to the extracted wheel, where all 16 public-prefix tests
passed. The previous ignored checkout extension (SHA-256
`10b8fd9e896e9600c8819ba01162d6ce7053a0a5a500d8d14f25cce388aba8f9`)
was backed up as `/tmp/grimace-live-event-qualification-994027e0/checkout-core-before.so`;
the refreshed checkout extension matches the wheel bytes. Checkout imports
resolved to that extension, and the four focused South Star 2 semantic modules
passed 37 tests. Every bounded build and test run reported
`verified_no_live_members`; durable logs and the wheel are under
`/tmp/grimace-live-event-qualification-994027e0/`. These checks qualify the
represented public and semantic paths, not the unproved completion criterion,
general support equality, or latency.

### First-release necessary relation (candidate slice)

At a full, nonempty visible ring-label set, a child traversal can be rejected
when even an optimistic path cannot reach its first held-ring closure. The
preview uses the actual traversal operations to commit and enter the proposed
branch or inline child, then reads visited atoms, unrepresented bonds, held
ring targets, active frame, and saved returns. A visited held target or one
with no unrepresented edge to a visited atom may release (the latter grants a
direct dot). Otherwise, a held target with one such edge has a unique visited
owner; targets with more have no usable entry before a release. Each unvisited
component is summarized from all unrepresented edges and all visited-boundary
incidences, including discarded frames. In active-then-saved return order, a
frame may reach its own held target; otherwise it can drain only components
that are trees with one boundary incidence to that frame. Exhausting the
frames without release contradicts the candidate.

The relation is necessary only: it grants arbitrary dots and ignores role,
local order, stereo, and source phase limits. It runs only for branch and inline
child candidates before the existing Event transition; false becomes the
existing candidate contradiction. Other candidate classes and the later
lexical child-entry transaction remain as they were. The legacy Event relation
still decides all admitted futures, and no general completion theorem or
speedup is claimed. Each applicable attempt adds a copy-on-write traversal
preview, a graph scan, and ordered-set owner operations, including attempts
later rejected by another relation.
The full-slot guard checks occupied physical slots without generating label
spellings; an occupied label with a spare slot bypasses the graph preview.

The independent finite structural grammar marks the stronger P,A,H,B,C,D
example unproductive after A-B even though held H shares B's unvisited
component through D-H; H-first remains productive and releases P-H before
reusing the label. With Event activation disabled only in tests, the actual
writer rejects both delayed A-B branch witnesses through the live relation.
A separate reachable tree witness retains an inline child while its saved
frame owns the held target. These are finite structural and native admission
witnesses, not exact public support equality or a dot-complete grammar proof.
The raw reference omits dot events, so its reverse reachability cannot establish
a MAXIMAL restriction. A smaller fixed-role triangle contract uses carbon atoms
A, B, T: A-B and A-T are committed Traversal bonds and B-T is a Ring bond.
The fixed graph has the ordinary SMILES witness `C(C0)C0`: A saves a return
before entering B, B opens the ring to T, the writer returns to A, enters T
through A-T, and closes the ring. The actual writer accepts this event path;
the test does not assert that exact lexical spelling. The grammar-legal `CC0`
prefix has B active and no saved A; even a later
dot entry to T cannot write the committed A-T Traversal incidence, because a
dot creates a new segment and cannot reintroduce the same A atom. The actual
writer rejects inline B before publication. This establishes only that an
unresolved committed Traversal incidence must retain whether its visited
endpoint is usable as a parent. Necessity of the full saved-stack order, local
phase, and stereo or capacity congruence remains unproved. This contract does
not replace the Event relation or establish a bounded completion algorithm.

Owner-set equality alone appears insufficient for a stronger capacity
criterion. In a five-carbon labelled graph, commit A-T and B-U to Traversal
and B-T, X-T, T-U to Ring. The leading-dot branch prefixes `A(.B0(.X1` and
`B0(.A(.X1` can have the same visited atoms, active X, held labels, and
available owner set `{A, B}`, but reverse the saved nesting. The first order
must finish B-U before A-T; the second can reach T first. With two labels,
opening U-T before closing X-T needs a third slot, whereas closing X-T first
releases one. Candidate complete spellings are `C0(.C(.C1)C010)C0` for the
T-first order and `C(.C0(.C1)C2)C012` for the U-first order with a third label.
The existing C/O dot-branch grammar witness interprets both spellings
independently of the production decoder and RDKit's SMILES parser; its test
asserts the exact labelled bonds in visit order. The two-label versus
three-label scheduling argument is limited to these fixed roles and orders;
the test does not prove unrestricted capacity minimality or RDKit writer
parity. Whether a reduced ordered-owner obligation suffices alongside stereo,
local phase, and cut constraints remains open.
A direct traversal test separately confirms that a held target with no
unwritten visited-boundary entry may be visited by a dot branch and close its
ring, and a spare-label admission test confirms that the relation does not
restrict the same delayed branch before capacity is full.
A separate bounded full-crate qualification attempt reported 362 passing tests,
5 ignored tests, and one exhaustive growth diagnostic stopped at its
300-second CPU limit. The cause remains unresolved; it does not establish a
regression from this slice. Focused first-release and layout checks passed.

Final focused native qualification passed 13 tests in seven filters; every
bounded run reported `verified_no_live_members`. A bounded offline host build
produced the CPython wheel in 18.531 seconds. The initial sandbox build stopped
after 2.36 seconds on namespace accounting, with verified cleanup; the host
process view was then used for the successful build. The wheel SHA-256 is
`560b9486726a179c9ad4ba13230bb04b3f43f4ffbd315b01458055ab2f1a0de2`;
its `_core` SHA-256 is
`01f4b6c21b4d83752854d2009dfc337d68bc1c2b0eff9563257ccc0002b1e3cd`.
Isolated exact-artifact provenance passed. Against that artifact, all 16 public
prefix API tests and 37 tests in the standard SMILES, compact language,
leading-dot tetrahedral, and Kekule semantic modules passed. These runs also
reported verified cleanup. Logs and artifacts are under
`/tmp/grimace-first-release-qualification-9db9adf8/final/`. The exhaustive
growth diagnostic above remains unqualified, with no established cause.
Root checkout provenance then passed: the five Rust source hashes and tracked
Python wrappers matched the archive, and the ignored checkout `_core` was
backed up and replaced with bytes identical to the qualified wheel. A bounded
root import resolved the checkout package, core, and preparation code and
confirmed the core SHA-256 above; command and runner exited successfully with
`verified_no_live_members`.

### Entered-child admission transaction

A child candidate now has one actual traversal entry at structural admission. `WriterState` prepares it from the source walker by committing the child incidence, capturing the inline parent's finalized local order, and entering that branch or inline child. The first-release necessary relation and child layout read this same prepared traversal; accepted lexical alternatives share its copy-on-write state, and solver restrictions still transition atomically from the source semantics. The source supplies EventSchedule ranks and source-relative directional commitments. The admitted successor owns the entered traversal; pending bond and atom spelling only refines semantics and emits text.

Keeping the first-release preview as a cache while retaining delayed child entry was considered. It preserves two owners of the same physical event and risks divergence in layout or later lexical alternatives, so the prepared entry is consumed as the actual successor instead. Parent local tetrahedral and directional-carrier checks must use the captured inline frame after the solver transition. Component-wide directional completion stays at completed-emission/path boundaries: entering a final inline child can complete graph progress before its atom spelling has committed its local semantics. Branch traversal spelling must derive its source endpoint from the incoming bond opposite the already active child. Distinct lexical alternatives must retain distinct solver domains while sharing only immutable entered traversal.

Deletion gates are the committed-child entry functions, committed-child layout previews, obsolete pending-entry modes, and the helper that creates then discards a first-release preview. Root and dot-root entry retain their existing single entry owner. Focused checks cover branch and inline implicit/explicit traversal spellings, directional branch orientation, final inline atom of a directional component, inline parent tetrahedral order, saved-frame restoration, distinct spelling domains, full-label dead branch and rescue/spare cases, and source immutability. Preparing rejected candidates still incurs a copy-on-write traversal clone and, for inline candidates, a local-order capture even when a later solver relation rejects them; the first-release graph scan remains conditional on full occupied label slots. No performance claim or EventSchedule deletion follows from this boundary change.

Focused native verification passed the seven live-obligation reference cases, structural writer-state tests, inline-entry tests, selected branch/directional/tetrahedral complete walks, the small independent transition-oracle comparison, and the new final-inline and equal-parenthesis regressions. Filter runs overlap, so their counts are not an additive test total. Each final bounded receipt reported exit status 0 and `verified_no_live_members`; logs are under `/tmp/grimace-entered-child-*.log`. The earlier exhaustive support-growth diagnostic remains historically unqualified and was not rerun.

A bounded offline host build of the frozen source produced a CPython wheel (SHA-256 `9433ebf9f0f0b8ba35870ac8450121f2dd792b9e77eeee4b1841421d1a8eeadd`) with `_core` SHA-256 `cc1b2a672692b9e689f200dded49e1a32056ac05257448d84037c0176097b6f5`. Isolated import and artifact-identity checks passed. Against that exact artifact, all 16 public-prefix tests and 37 tests across the four South Star 2 semantic modules passed. The 186 tracked crate and Python-wrapper files were unchanged and byte-equal to root. The root ignored ABI was backed up and refreshed with the exact qualified bytes; a bounded root import and byte-identity check passed. Every artifact qualification receipt exited 0 with `verified_no_live_members`; logs and artifact are under `/tmp/grimace-entered-child-qualification-8aeb4dac/`. This qualifies those tested paths, not EventSchedule removal, a general completion theorem, or a work bound; no speed claim follows.

## Attached forced-Traversal forest: conditional semantic realization contract

The proposed `MAXIMAL` lemma concerns an actually valid live state with no held
ring labels. Every unrepresented bond is already forced to Traversal, and each
unvisited residual component is a tree attached by exactly one physical bond
to the actual active or saved frame. There are no unattached future roots.
The physical edge and attachment determine each entry's incoming bond and
source without selecting a future visit rank or branch end. The claim is
conditional on one common assignment of every applicable semantic relation,
with one realizable local lexical row for each entry and a finite completion
of any pending spelling. A row includes the induced entry, bond role and
representation, local atom spelling restrictions, and directional mark at the
physical source; alternatives form a disjunction. If a future proof establishes
row-to-renderer correspondence, remaining children can be emitted in a
row-permitted order, branching except for the final inline child and returning
through actual saved frames. The converse must project each accepting labelled
walk onto the same joint assignment. This is a proposed construction, not a
proved general extension theorem or a production admission rule.

One test-only chain contract in `forest_realization_contract_tests.rs` checks
a narrower case: six fixed, non-aromatic, non-tetrahedral atoms in a single
path, with two double bonds and three directional carrier bonds. The middle
carrier belongs to both directional relations. The relation phases are false
and true. With physical endpoint orientation from the prepared graph, the
independent mark rule is `m0 != m2 && m2 == m4` for marked carriers; a plain
carrier cannot supply either required side phase. The test checks all 27
three-valued mark tuples from each endpoint root, 54 tuples in total. Per
orientation, exactly two tuples satisfy this rule, yielding four accepted
row-guided traces; the other 25 have no complete compatible row
product or contradict the joint native relation. It folds each compatible
row product into one native solver transition, then fixes the same semantic
inventory and follows the induced inline entries through the actual writer.
At each step it checks the labelled bond and atom spelling, physical source,
visited count, incoming edge, role progress, pending phase, local grammar
successor, and final acceptance. Directional text is independently checked
against the physical source and mark; the same mark variable is used on both
relations. Atom lexical spellings can have more than one raw string, but this
fixture has one semantic atom class per entry, so one finite grammar path per
class suffices for this correspondence check.

This fixture's prepared model has no EventSchedule factor or future visit,
entry-kind, or branch-end variables. Preparation also allocates no ordinary
pair-order variables for it. The test inventories the factor families and
activation: topology and binary side-pattern/parity factors are active;
ring-count and directional ring-placement factors are latent. It asserts the fixed
atom alternatives add no semantic restrictions or factor activations. The
writer and native solver share the prepared relation definitions, so the
separately derived parity rule and per-step physical/grammar checks provide
an independent check of their use for this chain, not a separate general
solver proof. The older closed-ring-tail test from the partial archive is
excluded: its setup failed before the intended boundary and supplies no
normative production counterexample. Its diagnostic remains in the original
`/tmp/grimace-forest-contract-7b3784f8` archive.

General forest order, tetrahedral child order and parity, branch returns,
closed-ring commitments, all pending lexical phases, and all semantic row
classes beyond this fixed chain remain unproved. EventSchedule currently
prepares ordinary pair-order variables alongside its speculative future
variables; any truly semantic order dependence must be separated before that
allocation and the schedule can be retired. This test makes no deletion,
efficiency, or general completion claim.

The exact focused test passed with exit status 0 and
`verified_no_live_members` under the bounded runner; its retained log is
`/tmp/grimace-forest-common-final2.log`. This result does not close the
general forest proof or authorize EventSchedule retirement.

### Joint child-entry spelling admission

Branch `(` and explicit inline bond text now come from a child-entry transaction that joins the selected traversal bond emission with the child's initial atom grammar, incoming layout, parent prefix, and directional anchor before publication. Each admitted alternative carries its selected bond text and atom grammar through pending syntax; equal visible tokens may therefore have distinct labelled successors. The implicit child route uses the same admission. Later grammar characters still pass the existing writer filter and semantic consistency check. This closes the local realization split only; EventSchedule remains the completion authority, and no work bound or global support theorem follows from this slice.

### Shared tetrahedral incidence relation

`TetrahedralIncidence` now owns the local physical rule: incoming bond, virtual
hydrogen, committed bond prefix, remaining ring-before-traversal order, and
order parity. The latent joint layout factor derives its pattern/order rows
from that rule. Atom token, parent prefix, and local completion restrictions
use its prefix projection; EventSchedule's full and partial local checks use
the same relation when interpreting a hypothetical physical entry. Prepared
and schedule centers no longer copy root/entry row arrays. The factor retains
its rows as a solver projection so role pattern and order remain correlated.

Finite tests independently enumerate three-bond centers with virtual hydrogen
and four-bond centers across incoming bonds, role patterns, both parities, and
every physical prefix length. They check the factor and renderer contracts;
focused adjacent tetrahedral, shared carrier, and Event tests passed. This is
one local semantic authority only. Lexical writer-policy totality, general
forest productivity, and EventSchedule retirement remain open.

### A fixed-Ring residual width proof gate

One exact structural subproblem exposes why an immediate release check is
insufficient. Fix every bond to `Ring`, allow every atom to be a dot root, give
ordinary atoms unrestricted incidence order, and count labels only at complete
atom-block boundaries. Let `P` be visited atoms, `U` unvisited atoms, and `H`
the held physical ring-bond identities. For a valid prefix, every held bond has
one endpoint in `P` and one in `U`. Completing `v` closes its incident bonds in
`H` before opening its other incidences to `U \ {v}`; consequently
`H' = (H \ incident(v)) union E(v, U \ {v})`. The minimum peak within this
block is `max(|H|, |H'|)`. Each dot-separated atom permutation is realizable;
return events do not reduce occupancy between complete blocks. Thus a valid
prefix has a completion with capacity `K` exactly when its residual edge-cut
width, starting at `H`, is at most `K`.
Necessity follows because every continuation crosses its sequence of cuts;
sufficiency follows by choosing a minimum-width atom permutation, closing
before opening at each block, and reusing freed label slots in its dot visits.
For test-only finite reference, define
`c(S)` as held bonds whose target is outside visited suffix set `S` plus
edges between `S` and `U \ S`; then `d(empty)=|H|` and
`d(S)=max(c(S), min_{v in S} d(S \ {v}))`. This enumerates subsets to check
the theorem, and is not a runtime admission algorithm or work bound.

For labelled `K4`, every first atom leaves three held labels, so visiting any
second atom releases one; nevertheless every two-atom cut has four edges.
The bond identities may reuse literal label slots once closed; `K` counts
simultaneous occupancy, while the production namespace has 100 slots. Capacity
three admits no full order, while capacity four admits all 24 labelled orders.
A three-atom prefix has residual future width three, but no such prefix
is reachable at capacity three because its past crossed a four-edge cut. The
finite test checks each physical cut and update at every prefix, and compares
two fully fixed dot assignments with the existing Event relation. This only
establishes the stated structural case: general role choices, branch stacks,
semantic correlations, and a compact incremental exact criterion with bounded
choice work remain unresolved. The finished target is still an actual-event
walker plus one joint live semantic constraint authority, without future visit
ranks, entry kinds, branch-end search, continuation assets, a fallback, or a
runtime fast path for this case.

### Private cut-load word algebra prototype (2026-10-15)

The test-only `cut_load_algebra` module represents a bounded nonnegative load
word at one fixed boundary context. It normalizes by deleting equal neighbors
and interiors between endpoint loads, compares words by coordinatewise
domination after monotone repetitions, and joins two words with horizontal or
vertical grid interleavings. Each grid cell keeps only nondominated typical
prefixes. Profiles may be compared only when they have the same boundary facts
and future choices; the module does not represent those facts. The result is a
quotient under mutual domination, not an inventory of exact attainable traces.
The normalization, comparison, and merge congruence use Bodlaender, Jaffke,
and Telle, [Lemma 2.10(i-v)](https://link.springer.com/article/10.1007/s00224-021-10030-3).

For capacity `K`, typical words have length at most `2K+1`; a coarse finite
inventory bound is `sum_{l=1}^{2K+1}(K+1)^l`. The DP has `|a|*|b|` cells,
but its per-cell profile inventory and domination comparisons remain part of
the work. This bound is deliberately not called practical at the writer's
100-label namespace, nor output proportional. Focused tests compare small
raw interleaving peaks under external contexts, association up to domination,
capacity shifts, and the valley counterexample `[2,0,2]` versus `[2]`.
On its own, this one-word module proves no graph decomposition, LIFO branch or
shared chemistry model, general Event replacement, or runtime admission claim.

### Private boundary-cut relation prototype (2026-10-15)

The test-only `boundary_cut_relation` module lifts bounded load words to a
supplied decomposition of a loop-free physical multigraph. A piece is
`P=(V,E,B)`: owned vertices, owned physical edge IDs, and boundary vertices
`B` contained in `V`. The graph owns edge endpoints; distinct IDs preserve
parallel edges. The piece owns isolated vertices explicitly because zero cut
loads cannot count visits. A raw row comes from one permutation of `V`, grouped
by its boundary order `pi`, with one cut-load word in each of the `|B|+1` gaps.
Each gap includes its endpoint cuts. For capacity `K`, let `R_K(P)` be these
raw rows whose peak is at most `K`. Write `x <= y` when monotone repetitions
align their entries with every entry of `x` at most the corresponding entry of
`y`; compare tuples componentwise only for the same `pi`.

The stored antichain `S(P)` has this invariant: every `r` in `R_K(P)` has some
`s` in `S(P)` with `s <= r`, and every `s` in `S(P)` has some `r` in `R_K(P)`
with `r <= s`. Thus `S` and the raw relation mutually cover each other for
existence under any nonnegative outside load, while `S` is not an exact trace
inventory. Pruning a row only against another *whole tuple* with the same
`pi` preserves both directions by transitivity. Independent gap unions would
mix incompatible choices from different child permutations.

The operator induction starts with the empty piece's single zero gap.
Introducing an isolated boundary vertex inserts it at one cut of one child
gap and duplicates that cut load as the two new gap endpoints. If `a <= b`
has aligned repetitions, any split of `b` admits an aligned split of `a`;
truncating the alignment on both sides gives two componentwise dominating
subwords. The reverse coverage uses the same alignment with roles exchanged
where the child rows mutually cover. Crucially, choose one *whole child row*
before splitting; the other gaps stay correlated. Introducing an owned edge
between boundary endpoints adds one to exactly the gaps strictly between their
positions, preserving the preorder. Forgetting a boundary vertex concatenates
the adjacent gaps, retaining both its possibly different pre-visit and
post-visit cut values. The old boundary orders partition all physical orders,
and word-concatenation monotonicity preserves coverage.

At a join, the two pieces have identical `B`, disjoint owned edge IDs, and
intersecting vertex sets exactly `B`; invalid ownership fails at the operator.
Every parent permutation restricts to one permutation per child with the same
`pi`, and every pair of such child permutations combines by independently
interleaving their internal vertices inside each gap. Edge ownership makes
the cut loads add. In *each* coverage direction, select one whole raw witness
per child first, then apply merge monotonicity gap by gap and take the product
inside that fixed row pair. The word merge covers grid interleavings up to
domination: a diagonal step from repeated extensions can be refined into two
single-piece steps, placing a decreasing step first so the intermediate sum
does not exceed the larger endpoint sum; stationary repeats disappear in the
typical quotient. This is the merge congruence cited above. Capacity filtering
preserves the invariant: insertion and forgetting retain the same loads and
peak, while edge introduction and joins add nonnegative load, so no later
operator can rescue an over-capacity child trace; a dominating row cannot
raise its peak.

For the anchored variant, every bag contains `z`, the leaf has `B={z}` and
two zero gaps, no vertex is inserted before `z`, and `z` is never forgotten.
Joins require the common anchor. By induction there is no interior visit in
the pre-`z` gap, and restriction of any `z`-first parent order is `z`-first in
each piece. This is an operational invariant, not an inference from `[0]`.
The theorem therefore covers `z`-first orders on a *valid supplied* nice
decomposition whose root owns the intended physical graph; the root relation
is empty exactly when no such order satisfies capacity `K`.

Independent labelled permutations check mutual coverage after every operator
on small unanchored and anchored pieces, including isolated vertices,
parallel edges, true joins, two decompositions of one graph, and small `K`.
Joining two three-leaf stars at their center rejects `K=2` for the six-leaf
star, exposing false support from independent gap minima. These finite tests
check the implementation; the operator induction above supplies the general
claim. With boundary size `b` and `t(K)` typical words, a coarse row inventory
is `b! * t(K)^(b+1)`, with separate metadata and row-product construction
costs. There is no supplied-decomposition generator or live update driver,
practical `K=100` bound, output-linear claim, or LIFO, stereo, role-choice,
shared-chemistry, production admission, or EventSchedule replacement result.

### Private boundary-cut plan over the relation (2026-10-15)

The test-only `boundary_cut_plan` prepares an immutable, capacity-independent
nice decomposition of a labelled loop-free physical multigraph. It chooses the
remaining vertex with minimum current filled degree, breaking ties by original
vertex ID. Eliminating `v` records the bag consisting of `v` and its remaining
neighbors, completes those neighbors to a clique, and removes `v`. When
anchored, `z` is retained last and auxiliary fill connects it to every other
vertex. Fill edges only shape bags; original physical edge IDs alone carry load.
The internal alternate-order constructor demonstrates that elimination is a
decomposition choice, not a restriction on writer atom permutations.

For each eliminated vertex, the earliest later neighbor is its parent; a
component without such a neighbor attaches to the empty virtual root, or to
`z` in the anchored case. Clique completion gives the separator containment:
every remaining neighbor in the child's bag also belongs to the parent's bag.
Consequently, bags containing any vertex form a connected subtree. The
construction recursively joins child pieces at the current bag, introduces
missing bag vertices, introduces each original edge at its earlier eliminated
endpoint, then forgets vertices outside the parent bag. Every original edge's
two endpoints are in that owner's bag, and its ID is introduced exactly once.
Leaf, introduce-vertex, introduce-edge, forget, and join nodes have stable
child-before-parent IDs and one parent each. Root pieces join at empty or
`{z}` boundary. Structural validation checks transitions, joins' exact shared
vertices, edge ownership, vertex coverage including isolates, connected bags,
and root coverage. The planner stores no descendant footprint sets; validation
computes temporary sets and discards them.

The conditional relation theorem above then applies to this generated valid
decomposition: for each capacity, bottom-up evaluation uses its existing
operators and retains unrestricted labelled orders (or `z`-first orders).
Independent raw permutation cut profiles mutually dominate its root rows for
every simple graph through four vertices, each anchor, and capacities from
zero through the number of edges. Additional finite checks cover disconnected
components, isolates, parallel physical edges, two elimination orders, and the
six-leaf star's rejection at capacity two and acceptance at three. These finite
checks exercise the construction; they are not a general compiler proof.

Capacity quotienting alone cannot certify joint writer productivity. Domination
is valid only within a complete shared traversal/stereo interface assignment;
such scopes may require larger bags before a fact can be forgotten. The next
gate is one real immutable-fact projection and walker delta, without rebuilding
a candidate future graph or adding a second authority. This plan carries no
future trace orders, caches, runtime admission, or production bypass.
