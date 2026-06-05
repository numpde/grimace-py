# SMILES sampling walk

## Goal

Document the internal walkthroughs used by public SMILES sampling:

```text
Prepared graph + seed + sampler policy -> one legal Grimace token path
```

The walk is for sparse next-token supervision. It is not RDKit random-writer
sequence parity and not uniform sampling over final SMILES strings.

The public Python API is `grimace.MolToSmilesSample(...)`; the public record
shape and accepted mode matrix are owned by
`notes/027_public_sampling_api_plan.md`.

Current implementation status:

```text
Python runtime state transitions + Rust seeded sampler -> one legal token path
```

That deliberately keeps the semantic source of truth in the already-tested
transition surface. A later Rust-side connected walker should be an
optimization of the same transition contract, not a second decoder.

## Result shape

Use a compact flat payload internally:

```text
tokens: tuple[str, ...]
selected_indices: tuple[int, ...]
choice_counts: tuple[int, ...]
choice_tokens: tuple[str, ...]
choice_branch_counts: tuple[int, ...]
```

For step `i`:

```text
start = sum(choice_counts[:i])
stop = start + choice_counts[i]
choices = choice_tokens[start:stop]
branch_counts = choice_branch_counts[start:stop]
selected_index = selected_indices[i]
token = tokens[i]
```

Required invariants:

```text
len(tokens) == len(selected_indices)
len(tokens) == len(choice_counts)
sum(choice_counts) == len(choice_tokens)
len(choice_tokens) == len(choice_branch_counts)
choice_counts[i] > 0
0 <= selected_indices[i] < choice_counts[i]
tokens[i] == choices[selected_indices[i]]
len(set(choices)) == choice_counts[i]
choice_branch_counts[j] >= 1
"".join(tokens) is accepted by the decoder
```

No EOS row is recorded. Stop at the first accepting decoder state; continuing
past an accepting prefix would require an explicit stop-vs-extend policy.

Public API planning for this payload lives in
`notes/027_public_sampling_api_plan.md`. The internal requirement here is only
that the flat payload contains enough information to build the public
token-level step report without re-walking the decoder.

## Multiplicity

The determinized decoder exposes unique next-token texts. A text can hide
multiple branch-preserving choices.

`choice_branch_counts[j]` is the number of branch-preserving choices merged
behind that exposed token at that prefix. It is not probability mass, final
completion count, or chemically meaningful by itself.

Implemented token-level sampling policies:

```text
uniform_token
branch_multiplicity
```

`uniform_token` samples exposed tokens uniformly. `branch_multiplicity` samples
with weight `choice_branch_counts`. Neither policy samples final SMILES strings
uniformly.

Branch-preserving sampling is implemented as a separate internal walker that
samples one hidden branch-preserving transition, then reports the selected
visible token bucket.

## RNG boundary

`rust/src/rng.rs` stays crate-private and chemistry-agnostic. It owns only:

```text
deterministic random source
unbiased bounded integer selection
weighted integer selection
```

Current internal shape:

```text
RandomSource { next_u64 }
Rng<S = SplitMix64>
Rng::uniform_index(len)
Rng::weighted_index(weights)
```

Keep sampler names and decoder logic outside `rng.rs`. The normal path can use
`Rng::from_seed_u64(seed)`; tests can inject a `RandomSource`.

Reproducibility contract:

```text
same prepared graph + same call flags + same seed + same Grimace version
=> same walk
```

Do not claim Python `random` parity, RDKit random-writer parity, cryptographic
security, or cross-version identity. The public API validates Python seed shape
before constructing the Rust sampler.

## Decoder source of truth

Do not add another graph traversal. The walk must consume the existing decoder
transition primitive:

```text
GroupedTransition {
    text,
    branch_count,
    successors,
}
```

Existing Rust connected-decoder methods are projections:

```text
next_token_support -> text
advance_token      -> selected transition successors
walk step          -> text + branch_count + selected transition successors
```

The current Python-composed walk consumes `_token_state_transitions()`, which
is the Python runtime projection of the same idea across connected,
all-roots, merged, and disconnected states. A token-level walk samples one
exposed token per prefix, then advances to that token's merged successor
frontier. A branch-preserving policy can sample hidden branch transitions, but
the public result should still be projected back to the same token-level step
shape.

## Runtime boundary

Reuse existing public normalization before any walk:

```text
RDKit Mol or PreparedMol
-> option coercion
-> supported flag validation
-> PreparedMol writer-flag validation
-> prepared graph selection by surface_kind
-> Rust walk or runtime composition
```

Dispatch by prepared graph `surface_kind`, not just `isomericSmiles`.

## Roots and fragments

Connected non-stereo all-roots can use the existing all-roots frontier.

Connected stereo all-roots must not eagerly instantiate every rooted stereo
decoder on public initialization. Either keep the existing lazy runtime
composition, or add a Rust equivalent that instantiates roots only as needed.

Connected all-roots stereo has a pinned core invariant: merged branches sharing
a visible prefix must share terminal status. That keeps accepting-state stop
semantics well-defined for connected all-roots walking.

Disconnected molecules are runtime-composed. A walk should mirror that:

```text
walk active fragment
record "." as a forced exposed token
walk next fragment
```

If recorded, the separator step has one choice token `"."` and branch count `1`.

## Tests

Reuse these existing oracles:

```text
tests/integration/test_runtime_state_invariants.py
tests/integration/test_public_all_roots_identities.py
tests/integration/test_public_prepared_equivalence.py
tests/integration/test_public_runtime_writer_flags.py
tests/parity/nonstereo/test_kernel_walker.py
tests/parity/stereo/test_kernel_walker.py
```

Walk tests cover, and should continue to cover:

```text
flat result invariants
step choices match manual determinized decoder choices
branch counts match branch-preserving duplicate counts
joined tokens are in MolToSmilesEnum support
same seed + same prepared graph + same call flags reproduces the walk
uniform_token uses exposed-choice count only
branch_multiplicity uses choice_branch_counts
RDKit Mol and byte-round-tripped PreparedMol agree
writer-flag mismatch rejects
unsupported canonical/doRandom options reject
```

Cases should include rooted/all-roots non-stereo, rooted/all-roots stereo,
disconnected molecules, duplicate same-token merges, visible divergence after a
merge, kekule output, explicit bonds, explicit hydrogens, and atom-map handling.

## Open decisions

```text
whether SplitMix64 remains the long-term source
whether all-roots stereo walking eventually moves Rust-side or stays
runtime-composed
```
