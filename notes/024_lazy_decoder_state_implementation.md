# Lazy decoder state implementation

This note records the implemented lazy decoder-state boundary.

## Implemented shape

Unrooted connected non-stereo already has one Rust decoder mode for all roots:

```text
RootedConnectedNonStereoDecoder(graph, root_idx=-1)
```

Unrooted connected stereo now has a lazy Python runtime state. Python prepares
the connected stereo graph once at decoder initialization, stores atom count,
and delays rooted stereo decoder construction until choices are requested.

Rust already exposes mutable connected decoder operations:

- `next_token_support()`
- `next_choice_texts()`
- `advance_token(token)`
- `advance_choice(index)`

The public decoder API still exposes `next_choices` and `choice.next_state`.
Internally, runtime states expose transition factories. Public choices realize
only the selected `next_state`; exhaustive tests and inventories explicitly
realize transitions through the single helper `_realize_state_transitions(...)`.

## Checklist

- [x] Confirm the test baseline.
  - `tests.integration.test_lazy_decoder_state_contract` was intentionally red
    before implementation and is now green.
  - `tests.integration.test_public_decoder` is green.
  - `tests.integration.test_runtime_state_invariants` is green.

- [x] Make public choices support lazy `next_state`.
  - Keep `MolToSmilesChoice.text` and `MolToSmilesChoice.next_state`.
  - Allow `next_state` to be computed once on first access.
  - Build public choices from lazy state transitions, not eager successor tuples.
  - Do not export a new class or public method.

- [x] Add private lazy transition plumbing.
  - Add a minimal private transition shape such as `(text, state_factory)`.
  - Ask states for lazy transitions when public choices need labels.
  - Make transition factories the only decoder-state traversal primitive.
  - Keep exhaustive audit/oracle helpers able to explicitly realize transitions.

- [x] Add a private lazy all-roots connected-stereo state.
  - Store the prepared connected stereo graph.
  - Store atom count.
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
  - Do not eagerly enumerate sibling successor states while producing public
    choices.

- [x] Remove the eager successor-state adapter surface.
  - Do not keep `choice_successor_states()` or `grouped_successor_states()` on
    runtime state adapters.
  - Route exhaustive traversal through module-level transition realization.
  - Guard the boundary with a static runtime-state test.

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
