# Lazy decoder state implementation

This note tracks the implementation plan for avoiding eager all-roots decoder
state construction while preserving the public decoder API.

## Current problem

Unrooted connected non-stereo already has one Rust decoder mode for all roots:

```text
RootedConnectedNonStereoDecoder(graph, root_idx=-1)
```

Unrooted connected stereo currently does not use an equivalent lazy public
runtime path. Python prepares the connected stereo graph once, then eagerly
constructs one rooted stereo decoder per atom and merges them. That makes
`MolToSmilesDecoder(..., rootedAtAtom=-1, isomericSmiles=True)` and
`MolToSmilesDeterminizedDecoder(...)` pay the all-roots cost at init.

Rust already exposes mutable connected decoder operations:

- `next_token_support()`
- `next_choice_texts()`
- `advance_token(token)`
- `advance_choice(index)`

The first no-regret implementation should use those private runtime hooks before
adding new public API or deeper Rust surface.

## Checklist

- [x] Confirm the test baseline.
  - `tests.integration.test_lazy_decoder_state_contract` was intentionally red
    before implementation and is now green.
  - `tests.integration.test_public_decoder` is green.
  - `tests.integration.test_runtime_state_invariants` is green.

- [x] Make public choices support lazy `next_state`.
  - Keep `MolToSmilesChoice.text` and `MolToSmilesChoice.next_state`.
  - Allow `next_state` to be computed once on first access.
  - Keep the existing eager successor-tuple path for current states.
  - Do not export a new class or public method.

- [x] Add private lazy choice-entry plumbing.
  - Add a minimal private entry shape such as `(text, state_factory)`.
  - Ask a state for lazy entries when it can provide them.
  - Fall back to current eager successor-state methods.
  - Keep exhaustive audit/oracle helpers able to use eager traversal.

- [x] Add a private lazy all-roots connected-stereo state.
  - Store the prepared connected stereo graph.
  - Store root indices.
  - Derive first-token information only when choices are requested.
  - Do not instantiate rooted stereo decoders at construction time.
  - `prefix()` returns `""`.
  - `is_terminal()` is false for nonempty graphs.
  - `copy()` is cheap.
  - `cache_key()` is stable enough for traversal caches.

- [x] Wire unrooted stereo construction.
  - In `_make_fragment_state_adapter`, keep the current connected non-stereo
    `rootedAtAtom=-1` path.
  - For connected stereo with `rootedAtAtom=-1`, return the lazy all-roots state.
  - Do not call `_core.RootedConnectedStereoDecoder` during unrooted stereo init.

- [x] Build only selected successors for public choices.
  - Branch-preserving choices instantiate only the selected root decoder.
  - Determinized choices instantiate only roots whose first token matches the
    selected token.
  - Do not call `_determinized_choice_successors` while producing initial
    determinized lazy choices.

- [x] Preserve disconnected behavior.
  - Check an active lazy fragment inside `_DisconnectedStateAdapter`.
  - Add minimal wrapping only if disconnected public choices otherwise force
    eager successors.
  - Keep `"."` separator behavior unchanged.

- [x] Preserve exact correctness.
  - Public decoder outputs must still match `MolToSmilesEnum`.
  - Runtime-state invariant tests must still prove exact initial support.
  - PreparedMol byte-round-tripped input must behave like RDKit `Mol` input.

- [x] Run focused verification after each slice.
  - `python -m unittest tests.integration.test_lazy_decoder_state_contract -q`
  - `python -m unittest tests.integration.test_public_decoder tests.integration.test_runtime_state_invariants -q`
  - PreparedMol runtime tests if disconnected or prepared-input planning changes.

- [x] Run broad verification before merge.
  - `make test`
  - Run package/release lanes only if this branch is promoted toward release.

## Non-goals for this slice

- No new public API.
- No semantic change to token order.
- No rewrite of Rust stereo internals unless the Python adapter cannot satisfy
  the contract.
- No cosmetic refactor of public decoder tests.
