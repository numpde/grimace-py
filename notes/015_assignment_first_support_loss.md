# Assignment-First Support Loss

Branch: `stereo-constraint-model`

## Question

Why does replacing runtime carrier resolution with
`resolved_selected_neighbors_from_assignment_state` lose support on coupled
diene witnesses?

The tested experiment was intentionally narrow:

```rust
fn resolved_selected_neighbors(
    runtime: &StereoWalkerRuntimeData,
    state: &RootedConnectedStereoWalkerStateData,
) -> Vec<isize> {
    resolved_selected_neighbors_from_assignment_state(
        runtime,
        &state.stereo_selected_neighbors,
    )
}
```

The experiment was reverted. This note records what failed and what state is
missing from the row model.

## Observed Loss

With the experiment applied and the extension rebuilt, the assignment-first
path only pruned outputs; it did not add outputs outside current support.

Affected pinned witnesses:

- `coupled_single_candidate_diene`, `C/C=C/C=C/C`: current runtime support
  `5`, assignment-first support `3`.
- `coupled_two_candidate_branched_diene`, `CC/C(C)=C/C=C/C`: current runtime
  support `28`, assignment-first support `22`.

Representative lost outputs:

```text
C(=C\C)/C=C/C
C(=C\C=C\C)/C
C(=C(/C)CC)/C=C/C
C(=C(\CC)C)/C=C/C
C(=C\C)/C=C(/CC)C
C(=C\C)/C=C(\C)CC
C(=C\C=C(/C)CC)/C
C(=C\C=C(\CC)C)/C
```

The Python diagnostic suite exposes the same shape. Under assignment-first
promotion:

- `coupled_single_candidate_diene` loses four
  `coupled_one_candidate_begin_side` terminal observations, from `10` to `6`.
- `coupled_two_candidate_branched_diene` loses six
  `coupled_one_candidate_begin_side` terminal observations, from `14` to `8`.
- `coupled_two_candidate_begin_side` remains unchanged at `14`.

So the failure is not "all coupled diene token phase" and not the
two-candidate begin-side branch. It is specifically the one-candidate
begin-side branch that still depends on delayed traversal context.

## Shape Of The Lost Rows

For each lost branched-diene output, terminal diagnostics agree on the same
important facts:

- the final carrier assignment is valid and singleton;
- the final token-phase assignment is valid and singleton;
- the inference branch is `coupled_one_candidate_begin_side`;
- `first_emitted_candidate` is unknown at the begin-side observation;
- `rdkit_token_flip_adjustment` is `true`;
- the visible marker sequence may include earlier branch/tree markers from
  other sides before the begin-side token observation is fully explained.

Example diagnostic inputs from a lost row:

```text
branch: coupled_one_candidate_begin_side
component_phase: flipped
component_begin_atom: 4
begin_side: 1
candidate_count: 1
selected_begin_neighbor: 5
selected_begin_token: /
first_emitted_candidate: None
rdkit_token_flip_adjustment: true
```

The current runtime keeps this alive because procedural token inference can
combine:

- the selected begin-side carrier;
- component phase;
- selected begin token;
- absence of a first-emitted candidate;
- RDKit token-flip adjustment.

The assignment-first experiment asks the carrier assignment state to resolve
the selected neighbor before that combined observation exists as a model fact.
That prematurely narrows the carrier domain and removes rows that are later
valid once token-phase and RDKit-adjustment facts are considered together.

## Missing Row-State Dimension

The missing dimension is not another local carrier repair.

The row state needs a joint observation dimension for delayed coupled
begin-side token interpretation:

- carrier choice for each side;
- component token phase;
- component begin atom or begin side;
- whether the selected begin-side carrier is the first emitted candidate,
  unknown, or known not first;
- RDKit writer token-flip adjustment;
- visible marker placement facts already observed on other sides.

Today these are split across:

- carrier assignment rows;
- token-phase assignment rows;
- marker-placement rows;
- procedural inference inputs on the walker state.

Assignment-first promotion fails because it promotes only the carrier slice.
The correct next step is to route carrier resolution through surviving rows
that already include token-phase and marker-placement/observation state, or to
add an equivalent joined row/projection before using assignment-state
resolution as runtime truth.

## Consequence For The Roadmap

`resolved_selected_neighbors_from_assignment_state` is still useful as a
shadow diagnostic, but it is not a safe runtime replacement by itself.

Before `Route directional tokens from row survivors`, the model must answer a
stronger query:

> Given the current carrier facts plus token-observation facts plus marker
> obligation/domain facts, which side carriers remain possible?

Only that joined survivor set can replace the current procedural carrier
resolution without pruning valid coupled-diene support.

## Verification

Commands run during this investigation:

```text
.venv/bin/maturin develop --release
cargo test --lib native_online_walker_matches_reference -- --nocapture
cargo test --lib
PYTHONPATH=python:. python3 -m unittest tests.integration.test_stereo_constraint_model -q
```

The experimental patch was reverted after collecting the failure data. The
working tree should keep current runtime behavior unchanged.
