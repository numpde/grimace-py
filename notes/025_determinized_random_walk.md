# Determinized random walk

## Intent

Investigate and add a Rust-side random walkthrough of the determinized SMILES
decoder.

The walkthrough returns one legal token path plus the legal determinized
next-token surface seen at each visited prefix. It is meant for training data
construction, especially sparse next-token supervision, not for RDKit random
sequence parity or uniform sampling from the final SMILES support.

The public API is intentionally undecided. This note describes the internal
payload and implementation boundary needed to test the idea without committing
the package surface too early.

## Internal result shape

A convenient candidate Rust/Python boundary payload is a flat record:

```text
tokens: tuple[str, ...]
choice_counts: tuple[int, ...]
choice_tokens: tuple[str, ...]
choice_branch_counts: tuple[int, ...]
```

For step `i`, the choices are the slice:

```text
start = sum(choice_counts[:i])
stop = start + choice_counts[i]
choice_tokens[start:stop]
choice_branch_counts[start:stop]
```

Invariants:

```text
len(tokens) == len(choice_counts)
sum(choice_counts) == len(choice_tokens)
len(choice_tokens) == len(choice_branch_counts)
choice_counts[i] > 0
tokens[i] in choice_tokens[start:stop]
len(set(choice_tokens[start:stop])) == choice_counts[i]
choice_branch_counts[j] >= 1
"".join(tokens) is accepted by the decoder as terminal
```

The candidate payload uses Grimace token strings. Vocab ids, EOS rows, trigger
tokens, probabilities, tensor layout, and sparse-loss weighting should remain
outside this internal Grimace result unless implementation evidence shows that
moving one of them into Grimace removes real duplication.

There is no terminal/EOS row in this shape. If a downstream training target
needs one, it can append it after interpreting the Grimace token path.

This is not a public API commitment. It is the smallest useful internal shape
for implementation and tests. A later public surface can wrap the same data in
a more ergonomic object, expose only part of it, or use a different view after
the training use case is clearer.

## Multiplicity

The determinized decoder exposes unique next-token texts, but each exposed token
may hide multiple branch-preserving choices.

`choice_branch_counts[j]` records:

```text
the number of branch-preserving decoder choices merged behind that exposed token
at that prefix
```

It is not:

```text
probability mass
number of final completions
uniform-over-SMILES support count
chemically meaningful by itself
```

It is useful for diagnostics and for optional local sampling weighted by the
branch-preserving decoder's multiplicity. Because this count is tied to the
current decoder state representation, it should be treated as an internal
surface until tests establish exactly what it means across non-stereo, stereo,
all-roots, and disconnected cases.

## Sampling modes

The useful local policies are:

```text
uniform_token
branch_multiplicity
```

`uniform_token` chooses uniformly among exposed determinized token texts.

`branch_multiplicity` chooses an exposed token with probability proportional to
its `choice_branch_counts` entry at the current prefix.

Neither mode samples final SMILES uniformly. Uniform-over-final-support sampling
would require memoized completion counts over the determinized state graph and
should be a separate feature if needed.

## RNG boundary

The internal walk should receive deterministic randomness explicitly: a seed for
normal use, or an injected source in tests. Keep RNG crate-private and
chemistry-agnostic.

The staged RNG module is `rust/src/rng.rs`. During tests-first staging, include
it from `lib.rs` under `#[cfg(test)]` and keep the module items private. Promote
the module and only the consumed items to `pub(crate)` in the same change that
adds a production walk consumer.

It owns:

```text
deterministic random sources
unbiased bounded integer selection
weighted integer selection
not tokens, molecules, decoders, or sampler policy names
```

The internal shape is one `RandomSource { next_u64 }` trait, one `Rng<S =
SplitMix64>` wrapper, and two selection methods: `uniform_index(len)` and
`weighted_index(weights)`. Tests and future implementations can inject another
source through `Rng::from_source(...)`; the normal path uses
`Rng::from_seed_u64(seed)`.

`uniform_index(len)`:

```text
requires len > 0
uses rejection sampling, not naive modulo
returns an index in 0..len
```

`weighted_index(weights)`:

```text
allows zero entries
requires at least one positive weight
uses a checked total weight
errors if the sample space is too large for the implementation's u64 draws
returns index i with probability weights[i] / sum(weights)
```

Start with a small local `SplitMix64` source behind the `RandomSource` trait.
If a library source is later preferable, add it behind the same trait. Avoid
thread-local randomness and unspecified `rand` defaults.

The reproducibility contract should be scoped narrowly:

```text
same prepared graph + same call flags + same seed + same Grimace version
=> same walkthrough
```

For RDKit `Mol` input, the RDKit version matters only insofar as it changes the
prepared graph that reaches the Rust decoder.

It should not claim RDKit random-writer sequence parity, Python `random` parity,
cryptographic security, or cross-version identity if decoder ordering changes.

Reject negative Python seeds or define an explicit conversion to `u64` at the
Python/Rust boundary, not inside `rng.rs`.

RNG tests pin:

```text
fixed SplitMix64 seeds have fixed short next_u64 prefixes
uniform_index(0) errors
weighted_index([]) and weighted_index([0, 0, ...]) return NoPositiveWeights
weighted_index with exactly one positive weight always selects that index
large sample spaces either work deliberately or error deliberately
custom RandomSource injection drives deterministic uniform/weighted choices
```

The sampler belongs outside `rng.rs`, in the code that owns the determinized
walk. It owns decoder-specific policy names and maps a grouped choice surface to
one index. This keeps the random source replaceable while keeping the sampler
choice closed, explicit, and testable. The current sampler policies do not need
token text; keep token-dependent policies out until a real use case requires
them. Do not use a Python callback sampler for this Rust-side path; it would
cross the Python boundary at every token.

## DRY implementation

Do not add a second graph traversal algorithm.

The existing connected Rust decoders already own the right primitive:

```text
next determinized token choices
advance by chosen token
terminal/prefix state
```

The likely internal refactor is a grouped choice object:

```text
GroupedChoice {
    text,
    branch_count,
    successor
}
```

Existing decoder methods can then remain projections:

```text
next_token_support      -> text
grouped_successors      -> text + successor
determinized_walk       -> text + branch_count + sampled successor
```

This lets the new walk record multiplicity without changing the current public
decoder API. If exposing multiplicity on the public decoder later becomes
useful, it can use the same grouped-choice source of truth.

## Runtime normalization

Any temporary wrapper or eventual public entrypoint should reuse the existing
runtime normalization:

```text
RDKit Mol or PreparedMol
-> public option coercion
-> supported flag validation
-> PreparedMol writer-flag validation
-> prepared graph selection by surface_kind
-> Rust decoder walk or runtime composition
```

Supported prepared/call options stay single-sourced in
`python/grimace/_mol_to_smiles_options.py` and
`python/grimace/_runtime_inputs.py`.

Dispatch must follow the prepared graph `surface_kind`, not only
`isomericSmiles`. For example, `isomericSmiles=False` with explicit directional
bonds can still require the stereo surface.

## All-roots and disconnected molecules

Connected non-stereo all-roots can use the existing all-roots frontier.

Connected stereo all-roots must preserve the current lazy behavior. Do not make
the public all-roots stereo path eagerly instantiate one full rooted stereo
decoder per root at initialization. A Rust walk helper may instantiate roots
only as needed for the current prefix and chosen token, or the implementation
may keep using the existing lazy runtime adapter until a Rust equivalent is
ready.

Disconnected molecules are currently composed at the runtime layer. A walk can
probably follow the same structure:

```text
walk active fragment
record "." as a forced exposed token between fragments
walk next fragment
```

If the internal payload records the fragment separator as a token, the dot step
has:

```text
choice_counts entry = 1
choice_tokens entry = "."
choice_branch_counts entry = 1
token = "."
```

## Tests to port or repurpose

Use existing determinized-state tests as the oracle where they apply:

```text
tests/integration/test_runtime_state_invariants.py
tests/integration/test_public_all_roots_identities.py
tests/integration/test_public_prepared_equivalence.py
tests/integration/test_public_runtime_writer_flags.py
tests/parity/nonstereo/test_kernel_walker.py
tests/parity/stereo/test_kernel_walker.py
```

New tests should assert:

```text
flat result invariants
step choices equal manual MolToSmilesDeterminizedDecoder choices
branch counts match branch-preserving duplicate counts at the same prefix
final joined tokens are in MolToSmilesEnum support
same seed + same prepared graph + same call flags reproduces the same walk
uniform_token selection uses exposed-choice count only
branch_multiplicity selection uses choice_branch_counts
RDKit Mol and byte-round-tripped PreparedMol agree
writer-flag mismatch still rejects
unsupported canonical/doRandom defaults still reject
```

Include cases for rooted and all-roots non-stereo, rooted and all-roots stereo,
disconnected molecules, duplicate same-token merges, visible divergence after a
merge, kekule output, explicit bonds, explicit hydrogens, and atom-map handling.

## Open decisions

Do not treat these as settled until implementation and tests make them concrete:

```text
public API shape, if any
whether the flat payload is returned directly or only used internally
whether SplitMix64 remains the long-term RNG source
seed argument type and validation
whether disconnected separators are recorded as ordinary tokens
whether all-roots stereo walking is fully Rust-side or temporarily runtime-composed
whether branch multiplicity is stable enough to expose outside the walk result
```
