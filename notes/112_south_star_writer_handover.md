# South Star Writer — Engineering Handover

## Repository state

Repository:

```text
numpde/grimace-py
```

Branch:

```text
south-star-1
```

Baseline when this handover was written:

```text
15bbaf95948e50d55c0695460507ec25533318e5
Make component boundary replay proof-based
```

The active work package is:

```text
Promote ordinary aromatic writer support
```

Do not begin unrelated chemistry, schema, count-kernel, or runtime work until
that package reaches its acceptance gates or stops at its declared precondition
boundary.

## Mission

South Star is an exact, branch-preserving SMILES writer engine. Its purpose is
not merely to generate valid SMILES. It represents the exact declared writer
support as a live transition system:

```text
current writer state
-> checked immediate transitions
-> emitted text
-> exact successor state
-> EOS
```

The system preserves:

```text
support identity
writer-witness multiplicity
exact support counts
exact completion counts
branch identity
snapshot/resume
facts-bound semantic evidence
```

The long-term product is a compact, fast online decoder that behaves like the
supported ordinary RDKit writer surface without invoking RDKit at decode time.
RDKit remains an ingestion and version-pinned audit boundary, not the runtime
semantic authority.

## End goal

The intended product path is:

```text
RDKit molecule or immutable MoleculeFacts
-> prepared writer facts and policy
-> verified continuation asset
-> Rust-backed online decoder
```

The online decoder must support:

```text
next emitted-text choices
deterministic advance
EOS availability
exact support count
exact completion count
exact rational next-choice probabilities
snapshot/resume
optional lazy proof reconstruction
```

The durable asset retains enough provenance to reconstruct and verify any
selected local transition or terminalization without embedding the complete
rich writer-state graph in the hot runtime core.

The declared support surface should grow as bounded, proof-complete vertical
slices. Unsupported cases must reject through typed blockers rather than
accidental failures.

## Architectural layers

### Facts and preparation

`MoleculeFacts` owns immutable molecular semantics:

```text
atoms
bonds
components and component order
implicit-hydrogen counts
tetrahedral sites
directional sites
ligand occurrences
stereo targets and reference orders
```

`SouthStarPreparedMol` owns reusable static structures:

```text
validated facts
ordinary spelling policy
parser semantics
graph index
block-cut metadata
component root domains
stereo templates
token inventory
```

RDKit is used only to extract facts and generate pinned parity evidence.

### Live writer kernel

The writer state contains:

```text
component cursor and component roots
active atom frame
branch stack
visited atoms
written tree bonds
pending graph obligations
ring state and ring-label lifecycle
policy-state atom and bond text
stereo residual state
```

The checked frontier computes only live immediate actions. It is the authority
for:

```text
available emitted texts
branch successors
terminal availability
execution blockers
capabilities
local evidence
```

The engine must not pre-enumerate full support to expose the next transition.

### Local proof artifacts

Two count-free proof artifacts cover the live kernel.

Nonterminal transition:

```text
writer_branch_transition_artifact v3
```

It contains exactly:

```text
source snapshot
selected text projection
selected branch support
```

Terminal transition:

```text
writer_terminalization_artifact v1
```

It contains exactly:

```text
source snapshot
selected terminal projection
selected terminal support
```

Both provide:

```text
closed structural verification
live reconstruction verification
producer-free facts-bound verification
```

These are the semantic proof authorities for individual online actions.

### Rich support artifact

Current schema:

```text
writer_support_artifact v11
```

This is the legacy complete-support proof table. It contains:

```text
support strings
replay paths
branch and terminal objects
support-image coverage
frontier-count envelope
count-certificate DAG
offline replay evidence
```

It remains useful for bounded default-corpus certification and compatibility
tests. It is not the preferred online runtime representation.

### Continuation automaton and durable asset

The weighted continuation automaton compiles the counts-disabled live frontier
into a compact acyclic semantic graph. It provides exact:

```text
support recurrence
completion recurrence
EOS count
per-choice count
integer probability numerator and denominator
```

The durable format is:

```text
writer_continuation_asset v1
```

Bundle layout:

```text
manifest.json
chunks/<content-digest>.json
```

The asset separates:

```text
compact semantic runtime core
replay-addressed provenance
```

For the shared directional-ring benchmark:

```text
support count:       3,744
completion count:    3,744
semantic nodes:      2,101
semantic edges:      2,843
core chunk:          approximately 875 KB
compact provenance:  approximately 22.7 MB
```

This replaced approximately 445.7 MB of embedded-cursor provenance.

### Rust runtime

The continuation asset loads into an immutable PyO3 Rust core through:

```python
MolToSmilesContinuationDecoder.from_asset(...)
```

Rust owns:

```text
choices
advance
EOS
counts
exact probabilities
copy/cache behavior
snapshot/resume
```

Counts use arbitrary-precision `BigUint`.

Python continues to own:

```text
asset structural verification
live provenance replay
facts-bound local proofs
RDKit ingestion
offline compilation
```

Core-only runtime must not load provenance, RDKit, prepared facts, live
frontiers, or legacy count machinery.

Proof-capable mode immediately binds the supplied prepared identity to the asset
and tracks raw cursor provenance alongside the Rust semantic cursor.

## Non-negotiable proof rules

### Coherent-forgery threat model

Assume an attacker can mutate and recompute:

```text
serialized terms
nested evidence
object IDs
cursor and state digests
chunk digests
asset manifests
top-level artifact digests
```

A valid hash proves only serialization integrity. It does not prove semantics.

### Three validation layers

Every proof-bearing representation distinguishes:

```text
structural validation
provenance and branch ownership
producer-free semantic replay
```

Structural validity must never imply semantic validity.

### Independent authority

Every serialized semantic field must be reconstructed from an independent
authority:

```text
MoleculeFacts
runtime options
serialized policy identity
source and successor writer states
branch events
fresh ResidualStore execution
continuation recurrences
```

A field may not be accepted solely because its own digest is correct.

### Branch-local ownership

Global replayed-digest sets establish only that evidence replayed somewhere.
Branch-local indexes establish that the evidence belongs to the exact branch
being proved. Both are required.

### No flag-based proof credit

The following are descriptive only:

```text
is_noop
is_empty
is_discharged
terminal_clean
operation strings
capability names
term presence
reciprocal links
rendered text
self-consistent digests
```

None may independently confer proof credit.

## Major completed milestones

### Runtime and state kernel

Implemented:

```text
branch-preserving live frontier
exact successor-state construction
graph and ring obligation lifecycle
residual stereo state
snapshot/resume
support and completion counting
terminal support
public decoder adapters
```

### Default offline-complete baseline

Established:

```text
auditable capability ledger
typed blockers
exact checked object kinds
exact checked relation families
empty unchecked families for accepted default cases
version-pinned RDKit audit fixtures
```

### Bracket atoms

Supported and pinned:

```text
simple charged nitrogen
simple negative oxygen
simple isotope carbon
```

Typed blockers include:

```text
charged isotope
unsupported oxygen charge
unsupported negative nitrogen
unsupported charged oxygen isotope
```

### Tetrahedral stereo

Implemented producer-free replay for:

```text
atom-token restriction
local-order closure
parity restriction
factor propagation
factor discharge
variable projection
terminal tetra closure
```

Replay is anchored to exact writer states and cannot be inferred from `@` or
`@@` text.

RDKit-ingested cases:

```text
[C@H](F)(Cl)Br   -> 6 support / 6 completion
[C@@H](F)(Cl)Br  -> 6 support / 6 completion
```

The two support images are disjoint.

### Directional stereo

Implemented:

```text
single acyclic carriers
shared acyclic carriers
single directional ring openings
directional ring pairs
shared two-site directional ring carriers
non-single directional ring closures
exact ring coupling replay
```

Ordinary implicit-H alkene support is included. Implicit hydrogens are fixed
ligand references, not residual carrier variables.

RDKit-ingested cases:

```text
F/C=C/Cl   -> 2 / 2
F/C=C\Cl   -> 2 / 2
```

### Ring closure semantics

Supported:

```text
single ring closures
joint double ring closures
joint triple ring closures
directional single-ring carriers
directional non-single ring closures
ring-label allocation, release, and reuse
```

Non-single ring marker placement is replayed semantically rather than inferred
from rendered strings.

### Component composition

Fixed-order disconnected composition is supported.

`MoleculeFacts.components` owns fragment order. A nonnegative root fixes only
the root of its containing component; other components retain their ordinary
root domains.

DOT branches are semantically replayed through:

```text
completed-component graph partition
producer-free local-order closure
component-index increment
unchanged root vector
exact next-root frame
residual component isolation
complete successor-state equality
```

Pinned cases:

```text
CC.O                     1 / 1
O.CC                     1 / 1
CC.CC                    1 / 2
C1CC1.O                  1 / 2
[NH4+].C#N               2 / 2
[C@H](F)(Cl)Br.O         6 / 6
F/C=C/Cl.O               2 / 2
```

`CC.CC` proves that identical support strings do not collapse distinct writer
witnesses.

### Count-kernel redesign

The rich legacy count DAG became too large for the shared-ring case:

```text
approximately 98,000 nodes
approximately 98,000 edges
approximately 29 MB manifest
```

The continuation automaton replaced it for online runtime purposes:

```text
19,595 primitive cursors
-> 2,101 semantic nodes
-> 2,843 semantic edges
```

The legacy count path remains available and green for existing bounded suites.

### Rust execution

The Rust continuation decoder provides approximately:

```text
547 KB resident shared-ring core
3,744-string traversal in 145 ms
sub-microsecond choices, advance, count, and probability operations
```

Existing Python and Rust support images match across the tested corpus.

## Current declared product contract

The default capability ledger includes accepted surfaces for:

```text
acyclic graphs
branched graphs
single rings
double and triple non-single ring closures
branched rings
simple bracket charges
simple isotope bracket atoms
specified tetrahedral stereo
specified acyclic directional stereo with implicit H
fixed-order disconnected composition
```

Each accepted row declares:

```text
support count
completion count
support digest
structural artifact acceptance
live artifact verification
facts-bound verification
offline completeness
frontier agreement
count agreement
snapshot/resume
continuation-asset agreement
Rust runtime agreement
lazy branch proof completeness
lazy terminal proof completeness
RDKit audit pinning status
```

Unsupported rows must declare typed preparation or frontier blockers and must
not produce assets.

## Active package: ordinary aromatic writer support

The current work package promotes:

```text
neutral unbracketed aromatic C, N, O, and S
aromatic tree bonds
aromatic ring bonds
fused aromatic rings
aliphatic/aromatic substitution
acyclic aromatic/aromatic single bridges
disconnected aromatic composition
```

Precondition corpus:

```text
c1ccccc1
n1ccccc1
c1occc1
c1sccc1
c1ccc2ccccc2c1
Cc1ccccc1
c1ccccc1-c1ccccc1
c1ccccc1.O
```

Critical semantic distinction:

```text
toluene:
    the aliphatic/aromatic single bond may be elided

biphenyl:
    the aromatic/aromatic fact-level SINGLE bridge must emit "-"
```

An omitted bond between two aromatic atoms denotes aromatic bonding and cannot
represent biphenyl's central single bond.

### Existing substrate

The ordinary policy already contains:

```text
lowercase aromatic atom tokens
elided or explicit ":" aromatic bond modes
aromatic ring-endpoint domains
```

The likely gaps are:

```text
exact facts-bound aromatic atom-text authority
endpoint-aware single-bond semantics
exact policy-state atom/bond delta replay
aromatic ring endpoint replay against the serialized policy
default ledger and pinned audit coverage
```

### Required package discipline

Do not add an aromatic-specific residual transition term. Aromatic spelling is a
finite policy/text relation, not residual stereo chemistry.

Keep all existing schemas unchanged unless the closed serialized vocabulary
truly changes. The continuation and Rust engines should require no
modifications.

### Required blockers to preserve

Remain unsupported:

```text
[nH] and other bracketed aromatic atoms
charged or isotopic aromatic atoms
atom maps
Kekule output
aromatic B/P without a separate bounded package
atropisomerism or aromatic stereo
unsupported aromatic single ring-closure bridges
```

## Validation expectations

When a package changes support-artifact or facts-verifier code, the complete
legacy facts verifier is mandatory:

```bash
PYTHONPATH=python:. python3 -m unittest \
  tests.south_star1.test_writer_support_artifact_fact_verifier -q
```

Expected baseline:

```text
112 tests
0 failures
0 errors
1 existing skip
```

Other recurring gates include:

```text
default capability ledger
default parity corpus
default offline-complete corpus
branch-transition artifacts
terminalization artifacts
continuation automata and assets
Rust continuation integration
count/frontier/support envelopes
boundary tests
pinned RDKit fixtures
```

Repository hygiene:

```bash
python3 -m compileall -q python/grimace/_south_star1 tests/south_star1
git diff --check
cargo fmt --check
cargo test --lib
cargo clippy --lib
```

There are 14 known pre-existing Clippy warnings in unrelated Rust stereo
modules. Do not broaden the active package merely to remove them.

Slow tests should be explicitly identified. Manual execution is acceptable for
very slow gates only when the equivalent production path is unchanged and the
omission is clearly recorded.

## Working method

For each package:

1. Run the declared precondition before changing schemas or product ledgers.
2. Abort cleanly if the precondition fails.
3. Implement one proof-complete vertical slice:

   ```text
   producer
   structural schema
   branch-local provenance
   producer-free replay
   proof accounting
   coherent-forgery matrix
   product integration
   ```

4. Maintain a field-to-authority table.
5. Assume coherent whole-artifact re-signing.
6. Require typed rejection reasons.
7. Preserve explicit boundaries for adjacent unsupported chemistry.
8. Run the complete affected validation gates before committing.
9. Commit only with a clean worktree.
10. Push and verify local HEAD equals `origin/south-star-1`.

Avoid implementing a positive runtime feature first and deferring semantic
hardening to later commits. The desired unit of progress is a proof-complete
vertical slice.

## High-value files and modules

Core preparation and facts:

```text
facts.py
rdkit_adapter.py
prepared_runtime.py
ordinary_policy.py
ordinary_semantics.py
```

Live writer:

```text
writer_state.py
writer_transitions.py
writer_frontier.py
writer_runtime.py
writer_graph_obligations.py
writer_stereo.py
writer_stereo_non_neighbor.py
```

Proof artifacts:

```text
writer_support_artifact_envelope.py
writer_support_artifact_checker.py
writer_support_artifact_offline_verifier.py
writer_support_artifact_fact_verifier.py

writer_branch_transition_artifact.py
writer_branch_transition_artifact_checker.py
writer_branch_transition_artifact_fact_verifier.py

writer_terminalization_artifact.py
writer_terminalization_artifact_checker.py
writer_terminalization_artifact_fact_verifier.py
```

Shared replay equations:

```text
writer_local_order_closure_replay.py
writer_component_completion_replay.py
writer_residual_transition_terms.py
```

Continuation runtime:

```text
writer_continuation_automaton.py
writer_continuation_asset.py
writer_continuation_rust.py
rust/src/continuation.rs
```

Product contracts and audits:

```text
tests/south_star1/default_writer_capability_ledger.py
tests/south_star1/test_writer_default_capability_ledger.py
tests/south_star1/test_writer_default_parity_corpus.py
tests/south_star1/test_writer_default_continuation_corpus.py
tests/fixtures/rdkit_south_star_*/
```

Design notes before this handover extend through:

```text
notes/111_component_boundary_proof_accounting.md
```

## Current success criterion

The active aromatic package is complete only when every promoted aromatic case
has:

```text
exact support and completion counts
reparse-to-facts identity
support artifact offline completeness
complete local branch proofs
complete terminal proofs
continuation-asset agreement
Rust support-image agreement
snapshot/resume agreement
version-pinned RDKit audit evidence
```

Existing graph, ring, bracket, stereo, disconnected, count, continuation, and
Rust behavior must remain unchanged.

Treat the live transition kernel, local proof artifacts, continuation asset, and
Rust decoder as established architecture. Add new chemistry by extending facts,
policy, typed blockers, and producer-free proof relations—not by redesigning the
online engine.
