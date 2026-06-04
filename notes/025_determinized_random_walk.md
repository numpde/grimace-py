# Determinized random walk

## Goal

Add an internal Rust-side walkthrough of the determinized SMILES decoder:

```text
Prepared graph + seed + sampler policy -> one legal Grimace token path
```

The walk is for sparse next-token supervision. It is not RDKit random-writer
sequence parity and not uniform sampling over final SMILES strings.

No public Python API is committed yet.

## Result shape

Use a compact flat payload internally:

```text
tokens: tuple[str, ...]
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
```

Required invariants:

```text
len(tokens) == len(choice_counts)
sum(choice_counts) == len(choice_tokens)
len(choice_tokens) == len(choice_branch_counts)
choice_counts[i] > 0
tokens[i] in choices
len(set(choices)) == choice_counts[i]
choice_branch_counts[j] >= 1
"".join(tokens) is accepted by the decoder
```

No EOS row is recorded. Stop at the first accepting decoder state; continuing
past an accepting prefix would require an explicit stop-vs-extend policy.

## Multiplicity

The determinized decoder exposes unique next-token texts. A text can hide
multiple branch-preserving choices.

`choice_branch_counts[j]` is the number of branch-preserving choices merged
behind that exposed token at that prefix. It is not probability mass, final
completion count, or chemically meaningful by itself.

Supported sampling policies:

```text
uniform_token
branch_multiplicity
```

`uniform_token` samples exposed tokens uniformly. `branch_multiplicity` samples
with weight `choice_branch_counts`. Neither policy samples final SMILES strings
uniformly.

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
security, or cross-version identity. Validate Python seed shape at the
Python/Rust boundary if a public API is added.

## Decoder source of truth

Do not add another graph traversal. The walk must consume the existing connected
decoder transition primitive:

```text
GroupedTransition {
    text,
    branch_count,
    successors,
}
```

Existing decoder methods are projections:

```text
next_token_support -> text
advance_token      -> selected transition successors
walk step          -> text + branch_count + selected transition successors
```

A walk samples one exposed token per prefix, then advances to that token's
merged successor frontier. It does not sample one hidden branch-preserving
successor.

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

New walk tests should cover:

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
public API shape, if any
whether the flat payload is returned directly or wrapped
whether SplitMix64 remains the long-term source
seed argument type and validation
whether disconnected separators are ordinary tokens
whether all-roots stereo walking is fully Rust-side or runtime-composed
whether branch multiplicity is stable enough to expose beyond the walk result
```
