# South Star 1 Ordinary Bounded Dialect v0

This package is a confined proof kernel for a small, explicit SMILES support
model. It is not the public Grimace runtime and it is not an RDKit writer model.

## Definition Boundary

The v0 support definition is:

```text
MoleculeFacts
  + ordinary_policy_for_facts(...)
  + OrdinarySmilesSemantics()
  -> enumerate_exhaustive_stereo_support(...)
```

RDKit may appear only at ingestion and audit boundaries. It is not part of
enumeration, candidate filtering, support definition, or witness repair.

## Input

The core input is a fixed finite `MoleculeFacts` graph:

- atom and bond facts are already perceived and fixed;
- components are fixed;
- stereo sites and ligand occurrences are explicit facts;
- policy choices are finite;
- ring-label domains are finite and least-free normalized.

RDKit Mol-state ingestion is available through
`ordinary_molecule_facts_from_rdkit`. For tetrahedral stereo, that adapter
currently assumes RDKit atom ids preserve the lexical order produced by
`Chem.MolFromSmiles`. Arbitrarily renumbered RDKit `Mol` objects are not
supported stereo-ingestion provenance in v0.

SMILES-source ingestion is available through
`ordinary_molecule_facts_from_smiles`. It uses RDKit's sanitized parse for
graph facts, but extracts raw tetrahedral source tags from the unsanitized
source parse before RDKit cleanup can remove them. Ordinary double-bond stereo
is read from the sanitized parse because RDKit exposes that relation after
SMILES cleanup.

## Supported v0 Surface

The current ordinary bounded dialect supports:

- plain neutral ordinary atoms accepted by `ordinary_policy_for_facts`;
- single, double, triple, and aromatic tree-bond spellings currently modeled by
  the ordinary policy;
- single and aromatic ring closures with bounded normalized labels;
- optional joint non-single ring closures where exactly one endpoint carries
  the double or triple bond-order marker;
- tetrahedral `@` / `@@` sites represented as explicit local relations;
- ordinary directional double-bond stereo represented as finite carrier-scope
  relations;
- disconnected inputs under the declared component traversal policy;
- RDKit audit as external falsification by parse-back and fact isomorphism.

## Explicitly Excluded v0 Surface

The current dialect rejects or omits:

- enhanced stereo groups;
- non-tetrahedral atom stereo;
- atropisomerism;
- `STEREOANY` and unknown/unspecified RDKit bond stereo classes;
- arbitrary-renumbered RDKit `Mol` stereo ingestion;
- non-single ring closures under the default policy;
- recursive ligand-equivalence refinement beyond the current immediate ligand
  color checks;
- RDKit writer-parity quirks unless they are explicitly modeled at the adapter
  or audit boundary.

## Conformance Corpus

The executable v0 conformance corpus lives in `tests/south_star1`:

- `tests.run_south_star_semantics` is the canonical unittest runner for the
  South Star 1 proof-kernel suite;
- `test_audit_rdkit.py` contains the RDKit external audit matrix and support
  stability checks;
- `test_boundary.py` enforces that only `rdkit_adapter.py` and
  `audit_rdkit.py` import RDKit;
- semantic tests exercise the RDKit-free policy, site-builder, CSP, renderer,
  support image, and fact-isomorphism layers.

The audit matrix is a falsifier for this spec, not the source of truth. A
passing RDKit audit means generated strings parse back to isomorphic facts under
the declared adapter, not that RDKit defines the support.

### Serialization Language Boundary

South Star keeps the broad proof grammar separate from writer-shaped runtime
serialization.

`EXHAUSTIVE` is the skeleton/slots/CSP/render certificate path. It enumerates
explicit traversal skeletons, tree/ring bond partitions, and local event
orders as proof objects.

`writer_shaped` is a separate serialization-language mode. It must use a live
writer-state transition system, not the traversal-skeleton proof grammar or
spanning-tree enumeration. Until that kernel is wired, writer-shaped runtime
requests fail closed instead of falling back to the exhaustive path.

## Witness Certificates

A certified witness records the traversal skeleton key, presentation prefix
key, stereo-CSP solution, relation rows, and annotation-selection certificate
that justify a rendered string. These are internal South Star proof objects:
they explain why the finite CSP accepted a witness, but they do not use RDKit
and they do not repair or filter generated support.

`SupportImage` is the unique rendered string image. Certified witnesses keep
witness multiplicity because distinct traversal, prefix, or CSP witnesses may
render to the same support string.

### Certificate Replay

Witness certificates are replayable by an RDKit-free checker. The checker
reconstructs the traversal, presentation, and CSP relations from
`MoleculeFacts`, policy, semantics, skeleton, slots, and assignment, then
verifies that the certificate covers all required relations and that every
certified row is valid.

A support enumeration manifest records skeleton, prefix, CSP, solution,
witness, and support counts plus deterministic support and witness hashes. The
manifest is an audit ledger for a finite enumeration run; individual witness
certificates remain the proof objects for emitted strings.

### Support Completeness Certificates

A witness certificate proves that one emitted witness satisfies the finite
relations. A support-completeness certificate records the finite enumeration
ledger: each skeleton, prefix, CSP, and solution branch is either accepted as a
certified witness, rejected by a named finite constraint, rejected by annotation
selection, or collapsed by rendered-support quotienting.

Structural replay reconstructs the enumeration domains from `MoleculeFacts`,
policy, and semantics, then verifies that the trace covers the finite search
space and that the manifest counts and hashes match. A valid
support-completeness certificate must not merely have the right counts: every
rejected branch must carry a named rejection reason that is replayed against the
corresponding finite domain object.

Regeneration comparison is available as an optional diagnostic: it compares a
submitted result with a freshly generated traced result. That is useful for
tests and debugging, but it is not the definition of certificate validity.

South Star support-completeness certificates are currently full traces.
Compressed traces may be added later, but they must preserve replayability. The
completeness trace is still RDKit-free; RDKit source audits remain external
falsifiers rather than support-definition machinery.

### Proof-Term Stability

South Star proof objects use public proof-term constructors for skeleton,
prefix, CSP, stereo-solution, witness, and duplicate nodes. The generator and
checker share this vocabulary through `proof_terms.py`; the checker does not
depend on private enumeration-helper names for proof keys. Trace JSON has an
explicit schema version, and unknown schema versions are rejected rather than
silently interpreted.

### Compiled Support Artifact

South Star can compile a support problem into a proof-carrying finite artifact.
The artifact contains canonical facts and policy identity, a traversal-space
certificate, prefix-space certificates, explicit stereo-CSP domains and
relations, feasible/selected solution partitions, render programs, witness
certificates, support-completeness traces, and a manifest.

`support_artifact.py` is the compiler side: it translates
`MoleculeFacts + policy + semantics` into explicit tables and proof objects.
The compiler may call the live traversal, prefix, CSP, renderer, and support
enumeration implementation because its job is to materialize the finite
problem.

`support_artifact_checker.py` is the artifact proof checker. It is RDKit-free
and producer-free: it does not import RDKit, the RDKit adapters or audits,
ordinary semantics, the traversal generator, the witness generator, or the
support enumerator. It consumes only the artifact's finite tables and checks
schema versions, canonical hashes, node/edge well-formedness, traversal and
prefix coverage, CSP relation-row satisfaction, annotation selection,
render-program output, support quotienting, trace reachability, and manifest
hashes.

This split is intentional:

- compiler trust: translation of live South Star objects into an explicit
  artifact;
- checker trust: internal consistency of the finite artifact and its proof
  ledger;
- external audit: RDKit source parse-back remains a falsifier, not part of the
  artifact checker.

### Artifact De-Self-Certification

The artifact checker does not trust boolean fields inside traversal
certificates, literal render strings, or certificate-local domain declarations.
It recomputes traversal validity from canonical facts and encoded parent/event
data, reconstructs rendering from structured atom/bond/ring/punctuation pieces,
checks prefix domains against the policy snapshot, and checks witness relation
rows against the artifact's compiled relation tables.

### Traversal and Prefix Space Completeness

The artifact checker independently reconstructs the finite traversal grammar
space from canonical graph facts and policy root domains. It checks root
choices, spanning forest edge sets, parent orientations, ring-edge complements,
and local event-order products without calling the live traversal generator.

The checker also reconstructs the finite prefix space from policy domains and
slot-bundle certificates: atom text domains, bond text domains, and bounded
ring-label assignments. The artifact is rejected if any legal skeleton or
prefix is missing, or if any extra skeleton or prefix is present.

### Semantic Relation Table Checking

The artifact checker does not trust compiled stereo-CSP relation tables merely
because they are present in the artifact. It independently reconstructs the
ordinary bounded semantic relations from canonical facts, policy JSON,
traversal decisions, slot-bundle certificates, and prefix assignments:

- tetrahedral local-order/parity relations;
- directional carrier scopes and signed reference-pair relations;
- tree-bond mark decode relations;
- ring-pair joint decode relations;
- no-accidental-stereo relations for potential unspecified sites.

The checker rejects an artifact whose compiled relation rows differ from these
reconstructed finite semantic tables.

### Support Artifact Schema

Support Artifact v1 has a closed schema. Unknown fields, unknown enum values,
duplicate proof nodes, duplicate domains or relations, dangling references,
noncanonical relation-row ordering, malformed facts JSON, and malformed policy
JSON are rejected before semantic replay.

The schema validator is RDKit-free and producer-free. It defines the artifact
input language consumed by the support artifact checker. Checked JSON loading
validates the JSONable object first, decodes it into artifact dataclasses, and
then validates dataclass-level cross references and canonical hashes before the
semantic artifact checker runs.

### Exhaustive Online Runtime Track

The proof-artifact track materializes finite certificates. The current online
runtime track is also exhaustive: it uses the same facts, policy, and ordinary
semantics, but maintains only a DFS traversal state, output buffer, and
reversible residual stereo constraints. It may emit witness strings with
multiplicity and does not deduplicate support globally. This track is not the
future `writer_shaped` writer-state runtime.

`stereo_templates.py` extracts small static tetrahedral and directional stereo
templates from `MoleculeFacts`. `residual_constraints.py` provides the
trail-based residual store and the first incremental tetrahedral/directional
stereo factors for the future online DFS enumerator. These modules are
RDKit-free and do not depend on the support-artifact or support-enumeration
tracks.

`prepared_runtime.py` defines the South Star preparation boundary. Preparation
validates `MoleculeFacts`, fixes the writer-surface flags, builds the finite
policy and parser semantics when callers do not provide them, extracts static
stereo templates, records a policy-derived token inventory superset, and stores
basic graph metadata, including atom ids, component ids, component atom domains,
and atom-to-component membership. Prepared molecules also own the reusable
`GraphIndex`, the all-root component domains, and an explicit-root domain cache.
Prepared online and offline entrypoints consume those cached structures instead
of rebuilding graph indexes or validating facts through the raw root-domain
helper on every query. Query-time runtime options such as rooting are kept out
of the prepared identity. Negative `rooted_at_atom` values enumerate all roots;
an explicit nonnegative `rooted_at_atom` restricts traversal roots to that atom
in its component while other components keep their root domains. Raw online and
offline traversal paths use the same `component_root_domains_for_facts(...)`
helper; prepared paths use the prepared root-domain cache. Rooted count-law
tests compare all-root witness counts against sums over explicit roots in
connected molecules and against sums over roots of a fixed component in
disconnected molecules.

`prepared_bench_matrix.py` provides a structural conformance matrix for the
prepared runtime. It compares prepared offline support enumeration with
prepared online determinized walks across execution modes, records support and
witness/EOS completion counts, and exposes query-time cache-reuse counters.
The matrix is correctness-oriented rather than a timing benchmark: tests assert
that prepared queries do not rebuild graph indexes, root-domain metadata, or
stereo templates, and that residual retained states do not carry rendered suffix
payloads.
`canonical=True` and `do_random=False` are rejected in the current online
runtime. Per-query decoder state still owns DFS traversal, output, ring,
residual trail, and frame-stack state.

`exhaustive_online_traversal.py` provides a lazy exhaustive traversal/event stream. It
enumerates roots, spanning forests, parent orientations, ring endpoints,
branches, continuations, and component dots by DFS without materializing the
traversal-skeleton space. The first online traversal tests compare yielded
trace keys against the offline finite-space skeleton keys on small molecules.

`online_stereo_witness.py` integrates the exhaustive online traversal stream
with the reversible residual constraint store. It emits witness strings with
multiplicity and performs no global deduplication. The first online runtime
supports the ordinary bounded tetrahedral and directional semantics, bounded
ring labels, and support-wise maximal marker selection within the current
traversal/prefix branch.

`writer_state.py`, `writer_events.py`, `writer_graph_obligations.py`,
`writer_transitions.py`, `writer_stereo.py`, `writer_frontier.py`,
`writer_snapshot.py`, and `writer_support.py` are the initial `writer_shaped`
writer-state kernel. The MVP supports ordinary acyclic prepared molecules,
including tetrahedral and directional stereo when the required carriers are
emitted by tree traversal. `writer_graph_obligations.py` provides the derived
graph-obligation view over the current writer prefix: a deterministic
current-component edge partition, residual components of unwritten graph work,
boundary incidences into syntactically open writer atoms, visited-visited
closure candidates, and structural block-cut metadata. Static writer graph
metadata is cached on the prepared molecule, while per-prefix edge partitions
and residual attachments are built through a single
`WriterGraphObligationContext`; the derived context is not stored in
`WriterStateKey`. Residual attachments carry explicit actionability classes:
acyclic tree entry, cyclic tree entry, closure-open-ready, or blocked states.
Branch and inline decisions consume tree-entry attachment classes derived from
the edge partition; they do not preselect a spanning tree, cycle basis, ring
cut, or render program. Cyclic residual attachments and closure-candidate edges
can be classified structurally. Raw writer transitions can now enter
single-boundary cyclic attachments and can open and pair closure endpoints for
internally constructed cyclic states when successor graph obligations remain
actionable, but public initial writer support still fails closed for cyclic
prepared graphs.
`WriterRingState` now owns explicit open and closed closure records plus ring
label state, and the edge partition classifies open and closed closure bonds
before residual attachments are derived. Initial writer support still requires
every prepared component to be a connected tree. Snapshot validation has a
separate graph-surface policy: it may audit internally coherent retained
closure state and actionable cyclic residual attachments, but blocked residual
attachments and closure candidate edges remain invalid.
Every writer transition emits typed semantic events; those events update a
writer-owned residual stereo snapshot that is part of the canonical writer
state key. Support counting, EOS evidence, next emitted-text choices,
completion counts, cursor snapshots, and support streaming all route through
the same determinized weighted frontier. EOS is represented by finalized
terminal cursor evidence, so terminal local-order and residual stereo closure
are persisted rather than recomputed as a discarded viability check. Writer
snapshots currently use a strict single-frontier-frame shape and validate a
structural prepared identity before resume. Snapshot validation also audits
each retained writer state against the prepared graph, runtime root domains,
residual attachment ownership, closure-state lifecycle records, local-order
occurrence records, delayed stereo factor records, and residual-store factor
snapshots before exposing a resumed cursor. Ring endpoint events have concrete
payloads and are consumed by writer stereo as ring-pair delayed-factor hooks.
Public cyclic support, supported ring-pair stereo factors, residual suffix
storage, RDKit parity, and exhaustive traversal fallback still fail closed in
`writer_shaped`.

`online_decoder.py` provides exact prefix feasibility and determinized token
frontier queries by running the online DFS with prefix-constrained render sinks.
It does not build a support trie, support image, compiled artifact, or global
deduplication table. The one-pass frontier collector runs a single
prefix-constrained DFS and commits a next character or token text only after
finding a complete valid witness. This is not yet the full main-branch
determinized decoder state: it returns the merged frontier for a prefix, but
does not retain a reusable merged successor state.

`online_decoder_state.py` provides stateful decoder choices. The
branch-preserving decoder may return multiple choices with the same emitted
token text when they correspond to different online generator branches. The
determinized decoder merges same-text choices into one choice whose next state
contains the union of branch continuations. This is lazy determinization: it
preserves a residual branch frontier for the next state but does not build a
global transition table.

The lazy determinized decoder stores frontier decision prefixes at the point
where an emitted token crosses the requested prefix boundary. A frontier path is
committed only after some complete witness extension is found, but the next
state stores the boundary prefix rather than the full completing path. This
keeps successor states residual and avoids turning decoder states into lists of
full witness completions. `OnlineDecoderChoice.multiplicity` counts merged
frontier paths; `completion_count` is diagnostic witness-count information.

Frontier compaction is an intentional over-approximation of residual branch
identity. The default compact frontier keeps traversal decisions and relies on
the emitted prefix plus residual DFS checks to re-filter syntax and stereo
choices. This may increase search but must not change the determinized token
frontier. Tests compare traversal-only compaction against full decision-prefix
compaction.

`online_decoder_api.py` exposes main-branch-shaped online decoder facades. The
branch-preserving facade may return multiple choices with the same emitted text.
The determinized facade returns at most one choice per emitted text and merges
frontier continuations. These APIs remain table-free: they do not build a
support trie, support image, compiled artifact, or global deduplication table.
EOS is explicit and optional.

EOS is computed by the same prefix-frontier DFS as ordinary next-token choices.
The decoder does not perform a separate full-witness scan to decide terminality.
EOS is committed only after a complete valid witness whose rendered string is
exactly the current prefix.

The first stateful decoder execution mode is prefix-replay based: a next state
stores a rendered prefix and a compact decision frontier, and later choices are
recomputed by running the online DFS under that filter. The `CACHED_COMPLETIONS`
execution mode stores compact completion-backed token continuations. It avoids
restarting from the molecule root after a choice, but it does so by replaying
previously discovered valid token completions. It is table-free with respect to
global support enumeration, but it is not yet a true residual DFS continuation.

The `RESIDUAL_CONTINUATIONS` execution mode stores suspended
`OnlineSearchSnapshot` states captured at token boundaries. A snapshot is
exposed only after the branch is proven to have at least one complete valid
witness extension. Later decoder calls resume those snapshots directly rather
than restarting DFS from the molecule root or replaying cached completed token
streams.

Residual continuation snapshots are checkpointed execution states. The
residual frontier sink checkpoints pending token-boundary snapshots together
with emitted text, so nested backtracking cannot lose or falsely commit a
pending continuation. Token and EOS frontier entries committed after a complete
valid witness are query-global evidence for the current frontier query and are
intentionally not rolled back during later DFS backtracking. Determinized
residual choices merge continuations by explicit residual continuation identity;
multiplicity counts residual states, while `completion_count` counts completing
witnesses that prove those states viable.

The current residual snapshot resumes an event-level render cursor. The snapshot
contains a frozen render program and event/piece cursor for the already selected
trace/prefix/mark context; it no longer stores a precomputed rendered-piece
suffix, but it is not yet a minimal general scheduler frame for all future DFS
alternatives. Resuming consumes the active render cursor before rendering the
next piece, so retained successor snapshots have at most one active render
cursor frame. Snapshot frame stacks use frozen typed payloads, not stringly
typed frame names plus unstructured object tuples. The snapshot owns the
residual constraint state needed for resumption; restoring one snapshot must not
mutate or depend on a residual store object owned by the producer VM or by
sibling snapshots.
Directional candidate admissibility is separated from active-sink rendering.
In support-maximal annotation mode, directional mark assignments are first
checked for finite semantic admissibility and captured as value-owned residual
snapshots, then support-maximal filtering runs, and only retained candidates are
rendered into the prefix/frontier sink. Discarded nonmaximal candidates must not
commit token or EOS frontier evidence.
Residual continuation state-size statistics separate candidate sink evidence
from retained decoder state. Candidate statistics count all token-boundary and
EOS snapshots observed while proving a frontier query. Retained statistics count
only snapshots stored in returned successor decoder states; EOS contributes
terminal path/count evidence but no retained residual snapshot. The statistics
report structural counts for residual variables/factors, frame-stack depth,
decision-path length, ring state, output snapshot length, duplicate merge fan-in,
and any render-cursor program payload. They are representation audits, not
Python byte-size measurements, and keep render-cursor costs explicit.

`online_search_vm.py` is the exhaustive explicit-stack event-level runtime. It
owns traversal progression, syntax-slot choices, residual stereo propagation,
ring-label state, output buffering, and decision recording. It does not iterate
prebuilt traversal traces and does not call the recursive online witness
enumerator. This VM is the substrate for `RESIDUAL_CONTINUATIONS` in exhaustive
mode: decoder states can store an `OnlineSearchSnapshot` at a token boundary
and resume from it, instead of replaying from the root or replaying cached
completions.
Snapshot resumption is routed through a typed frame dispatcher. The current
dispatcher resumes `RenderCursorFrame`, `PrefixEnumerationFrame`,
`DirectionEnumerationFrame`, and `SupportMaximalFrame` payloads through a
central resumable-frame registry.

### Residual snapshot validation boundary

Any `OnlineResidualContinuation` retained by `RESIDUAL_CONTINUATIONS` must
contain at least one known dispatcher-handled resumable frame. Context-only
frame stacks, unknown frame payloads, duplicate active resumable frames, and
unhandled topmost resumable frames are invalid residual continuations. This
invariant is checked at continuation capture, retained-state audit, and
snapshot resume.

`online_serialization_stream.py` exposes a direct serialization stream on top
of the online decoder facade. It walks decoder states with EOS enabled and
emits each determinized support string once, using `RESIDUAL_CONTINUATIONS` by
default. `support_count` is the number of distinct emitted serialization
strings, `witness_completion_count` is the sum of EOS `completion_count` values
over those strings, and each emitted item reports the EOS frontier
`multiplicity` for that string. This stream is an online runtime API: it does
not use the support enumerator, support image, compiled artifact, or global
deduplication table to produce strings.

## Experimental Options

`OrdinaryStereoSiteOptions(ligand_equivalence="exact_graph_automorphism")`
broadens potential stereo-site recognition beyond the v0 immediate-color rule.
It uses an anchored label-preserving graph automorphism relation to decide
whether two ligand occurrences are equivalent.

This option is opt-in and is not part of the v0 conformance corpus. The v0
default remains `ligand_equivalence="immediate_color"`.

The graph exact mode is atom/bond graph exact. It does not make pre-existing
stereo labels part of the automorphism relation.

`OrdinaryStereoSiteOptions(
ligand_equivalence="exact_stereochemical_graph_automorphism"
)` is the stronger experimental relation. It uses the same atom/bond graph
automorphism search, then requires all non-ignored specified stereo facts to be
compatible under the shared `stereo_mapping.py` parity/reference-pair relation.
The candidate site's own existing site id is ignored during site construction
to avoid circularly using a site's stereo label as evidence for its own ligand
distinctness.

RDKit ingestion remains one-way and non-definitional for these experimental
options. The default one-pass adapter builds ordinary potential sites before it
overlays RDKit specified stereo, so a remote-stereo-dependent candidate that
requires `exact_stereochemical_graph_automorphism` is not discoverable in that
mode. `RdkitOrdinaryExtractionOptions(stereo_site_discovery_passes=2)` enables
an explicit operational second pass: overlay any discoverable specified stereo,
rebuild missing potential sites with those labels available, then require the
final overlay to resolve all supported RDKit stereo.

`RdkitOrdinaryExtractionOptions(stereo_site_discovery_mode="specified_closure")`
is the mathematical experimental mode for raw specified RDKit stereo. The
adapter extracts raw specified stereo records first, materializes them as a
temporary specified context, checks every raw site for ordinary eligibility with
its own site id ignored, and then rebuilds the full potential-site universe
under the accepted specified context.

### Specified Stereo Closure Certificate

The `specified_closure` mode accepts a raw specified stereo graph iff each raw
record is eligible under the context of all other raw specified records, with
its own candidate site ignored. The implementation exposes a certificate per
raw record recording the matched site, context records, ignored self-site ids,
and accept/reject reason.

### RDKit Mol-State vs SMILES-Source Ingestion

`ordinary_molecule_facts_from_rdkit(mol, ...)` snapshots the RDKit `Mol` state
provided by the caller. If RDKit sanitization or cleanup has removed a stereo
tag, that tag is not part of the Mol-state facts.

`ordinary_molecule_facts_from_smiles(smiles, ...)` is the source-text contract.
It uses RDKit for graph parsing, extracts tetrahedral source tags from the
unsanitized parse before cleanup, and reads ordinary double-bond stereo from
the sanitized parse. This is the appropriate external audit path for generated
South Star SMILES support strings under `specified_closure`.

The Mol-state specified-closure audit remains a useful diagnostic. Some
generated strings have raw tetrahedral tags that are retained by unsanitized
RDKit parse but removed from the sanitized `Mol` state; the diagnostic trace
classifies those cases as `SPECIFIED_TETRA_RECORD_LOSS`. The source-ingestion
audit is expected to preserve those raw records and round-trip them through
South Star facts.
