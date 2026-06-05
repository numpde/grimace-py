# Public SMILES sampling API plan

## Goal

Add a public API for drawing one complete Grimace-supported SMILES path while
retaining the legal next-token context observed along the path.

The operation is sampling, not just decoding:

```text
molecule + writer flags + seed + validated sampling pair
-> one complete legal SMILES token path + per-prefix token choices
```

It is not RDKit `MolToSmiles(..., doRandom=True)` parity, not uniform sampling
over final SMILES strings, and not a new traversal beside the existing decoder
transition surface.

## Public shape

Public function name:

```python
grimace.MolToSmilesSample(..., seed=...)
```

Public result record names:

```python
grimace.SmilesSample
grimace.SmilesSampleStep
```

Public record fields:

```python
sample.tokens
sample.smiles
sample.decoder_view
sample.sampling_mode
sample.steps

step.choice_tokens
step.choice_branch_counts
step.selected_index
step.selected_token
```

Records are immutable, small, and minimal frozen slotted dataclasses. The
public runtime value for `decoder_view` and `sampling_mode` is `str`; runtime
validation owns the accepted values.

`seed` is a required keyword-only unsigned 64-bit integer. Do not add a silent
random default.

Every public step reports the same token-level view:

```text
choice_tokens are unique visible token texts
len(choice_tokens) == len(choice_branch_counts)
choice_branch_counts are positive ints
0 <= selected_index < len(choice_tokens)
selected_token == choice_tokens[selected_index]
```

Branch-preserving sampling chooses among duplicate hidden branch transitions
internally, but the public report still maps the selected hidden branch back to
its visible token bucket.

`decoder_view` records the state space used to produce and advance the draw. It
does not change the public step layout; steps remain token-level reports.

## Validated mode matrix

`decoder_view` and `sampling_mode` are a validated pair, not independent knobs.

Accepted pairs:

```text
decoder_view="determinized", sampling_mode="uniform_token"
decoder_view="determinized", sampling_mode="branch_multiplicity"
decoder_view="branch_preserving", sampling_mode="branch_preserving"
```

Do not infer behavior from arbitrary string combinations.

`doRandom` remains part of the existing RDKit-like writer-surface option set.
It is not the sampling policy for this API. The sampling policy is the
validated `decoder_view`/`sampling_mode` pair.

## Selection spaces

All public samples report token-level steps. The selection and advancement
space depends on the validated pair.

`determinized/uniform_token`:

```text
report choices: token transitions
sample from:    token transitions, uniformly
advance by:     selected token transition
```

`determinized/branch_multiplicity`:

```text
report choices: token transitions
sample from:    token transitions, weighted by branch_count
advance by:     selected token transition
```

`branch_preserving/branch_preserving`:

```text
report choices: token transitions
sample from:    branch transitions, uniformly
advance by:     selected branch transition
report selected_index: index of the selected branch text's token bucket
```

At a given prefix, `branch_preserving/branch_preserving` and
`determinized/branch_multiplicity` can have the same visible token
probabilities. They are not equivalent: the former advances one concrete hidden
branch, while the latter advances the merged token state.

## Serious alternatives

### Public function vs decoder method

`MolToSmilesSample(mol, ...)` is preferable to `decoder.sample(...)`.

The sample operation starts from a molecule, public writer flags, and a seed.
It returns a full path. A decoder method would raise awkward questions about
whether sampling starts from the current prefix, consumes the decoder, or
returns a partial draw. That may become useful later, but it is not the first
public API.

### Rich sample object vs tokens only

A tokens-only API is too small:

```python
("C", "C", "(", "=", "O", ")", "O")
```

The intended training/debug consumer needs the legal alternatives and branch
counts at every sampled prefix. Returning only tokens would force a second API
or another decoder pass.

### Frozen records vs dict payloads

Dicts are flexible but weak. A tiny immutable record gives discoverability,
stable attribute names, better type stubs, and simpler installed-package
correctness tests. Serialization can be added later if there is a real need.

### Determinized-only first vs all validated pairs first

Implementing only determinized sampling would have been smaller, because the
private token walker already consumes token transitions and records token
choices and branch counts.

Implementing all validated pairs in the first public slice was the cleaner
choice. The public result shape is shared, and branch-preserving sampling is a
different selection/advancement space over the same private transition
contract. Adding it before release avoided publishing a mode matrix with a
reserved but unusable pair.

### Python adapter first vs Rust sampler first

The first public API is a thin Python adapter over the tested private runtime
transition surface. A Rust connected sampler remains an optimization path. It
must return the same semantic payload rather than becoming a second decoder.

## Implementation checklist

Preflight:

- [x] Confirm `main` is clean enough for a public API branch.
- [x] Open and use `feature/public-smiles-sampling`.
- [x] Re-read `notes/025_determinized_random_walk.md` and
      `notes/026_transition_surface_plan.txt` before editing.
- [x] Confirm existing private walker tests pass before adding public tests.

Public API tests first:

- [x] Add red tests that `grimace.MolToSmilesSample` is exported.
- [x] Add red tests that `grimace.SmilesSample` and
      `grimace.SmilesSampleStep` are exported.
- [x] Add red tests for `MolToSmilesSample` result fields.
- [x] Add red tests for sample-step fields.
- [x] Keep private sampling wrapper defaults out of `_sampling.py`; public
      defaults belong to `grimace.__init__` and the option inventory.
- [x] Assert `sample.smiles == "".join(sample.tokens)`.
- [x] Assert every step has unique `choice_tokens`.
- [x] Assert every `choice_branch_counts` entry is a positive int.
- [x] Assert `selected_index` is in range.
- [x] Assert `selected_token == choice_tokens[selected_index]`.
- [x] Assert sampled `sample.smiles` belongs to `MolToSmilesEnum` support on
      small exhaustive cases.
- [x] Assert same molecule, flags, seed, `decoder_view`, and `sampling_mode`
      reproduce the same sample within the same Grimace version.
- [x] Assert all accepted pairs are allowed:
      `determinized/uniform_token`,
      `determinized/branch_multiplicity`, and
      `branch_preserving/branch_preserving`.
- [x] Assert invalid `decoder_view` values reject.
- [x] Assert invalid `sampling_mode` values reject.
- [x] Assert invalid `decoder_view`/`sampling_mode` pairs reject.
- [x] Assert `branch_preserving/branch_preserving` reports the same token-level
      step shape as the determinized modes.
- [x] Assert invalid seeds reject at the public boundary.
- [x] Assert bool seeds reject even though `bool` is an `int` subclass.
- [x] Assert omitting required `seed` rejects.
- [x] Cover RDKit `Mol` and byte-round-tripped `PreparedMol`.
- [x] Cover rooted and all-roots non-stereo.
- [x] Cover rooted and all-roots stereo.
- [x] Cover disconnected molecules and forced `"."` separator reporting.
- [x] Cover explicit public writer flags through the same supported-runtime
      helper used by other public tests.

Private walker extension:

- [x] Extend `_TokenWalkResult` with `selected_indices`.
- [x] Record selected indices at the same time choices are recorded.
- [x] Update private walker invariants to check selected indices directly.
- [x] Do not infer selected index from selected token after walking.
- [x] For branch-preserving sampling, record `selected_indices` as public
      token-bucket indices, not raw branch-transition indices.
- [x] Add private transition-surface tests that token-transition
      `branch_count`s equal branch-transition text counts for representative
      core, lazy all-roots, merged, and disconnected states.
- [x] Factor step recording so token-level reporting is shared by token and
      branch-preserving walkers.
- [x] Keep token-transition walkers advancing by selected token transition.
- [x] Add a branch-preserving walker that samples `_branch_state_transitions()`
      but records `_token_state_transitions()`.
- [x] Add a private walker fixture where branch-preserving and determinized
      sampling select the same visible token but advance to different future
      choice sets.
- [x] Map the selected branch transition text to exactly one token-transition
      bucket.
- [x] Treat a missing selected branch text in token transitions as an internal
      invariant failure.
- [x] Treat duplicate token-transition texts as an internal invariant failure.
- [x] Keep `_runtime_walks.py` private.
- [x] Keep chooser functions private.
- [x] Do not expose branch identities in the public sample record.

Public implementation:

- [x] Add private validation for the mode matrix.
- [x] Add private validation for unsigned 64-bit seeds at the public boundary.
- [x] Keep `seed` required; do not silently use process randomness.
- [x] Add `python/grimace/_sampling.py` as the single owner of public sample
      records, mode validation, and the public sampling wrapper.
- [x] Add `SmilesSampleStep`.
- [x] Add `SmilesSample`.
- [x] Add `MolToSmilesSample(...)` function in `grimace.__init__`.
- [x] Reuse `_runtime_kwargs(locals())` for writer options.
- [x] Keep `doRandom` in the writer-option path; do not overload it as the
      sampler mode.
- [x] Reuse existing runtime normalization through a private helper that returns
      the initial decoder state; do not instantiate public decoder classes just
      to reach private `_state`.
- [x] Map `sampling_mode="uniform_token"` to the seeded uniform transition
      chooser.
- [x] Map `sampling_mode="branch_multiplicity"` to the seeded branch-count
      chooser.
- [x] Map `decoder_view="branch_preserving",
      sampling_mode="branch_preserving"` to the seeded uniform branch-transition
      chooser.
- [x] Return public records from the internal flat walker result.
- [x] Export only the public function and public record classes.
- [x] Keep `_SplitMix64Sampler` private to `_core` and typed only for the
      private Python bridge.

Boundary and SSoT checks:

- [x] Confirm no tests import new public objects from private modules.
- [x] Confirm the public function delegates through one intended private
      wrapper instead of scattering `_runtime_walks.py` imports.
- [x] Confirm option names are not duplicated outside the existing option SSoT.
- [x] Confirm mode validation is owned by one module.
- [x] Let tests spell the public mode strings when pinning the public
      contract.
- [x] Let implementation-coverage tests derive the accepted mode matrix from
      the owning module so newly accepted pairs cannot escape coverage.
- [x] Confirm package import exposes all expected public attributes.
- [x] Confirm unsupported canonical/default runtime behavior rejects through
      the existing public option machinery.

Docs:

- [x] Update API docs only after tests pass.
- [x] Describe this as one sampled Grimace-supported SMILES path.
- [x] Document that it is not RDKit random-writer parity.
- [x] Document that it is not uniform over final SMILES strings.
- [x] Document the validated mode matrix.
- [x] Document `branch_count` as local prefix multiplicity.
- [x] Add a compact example that consumes `sample.steps`.
- [x] Avoid adding broad claims outside current limitations.

Verification:

- [x] Run focused public sampling tests.
- [x] Run private runtime walk tests.
- [x] Run public decoder tests affected by `branch_count`.
- [x] Run `make test`.
- [x] Run `cargo test` if Rust or stubs change.
- [x] Run `make docs` after docs changes.
- [x] Run `make test-package` before merge/release.
- [x] Confirm installed-package correctness includes the new public sampling
      tests or a registered public API smoke target.

Release readiness:

- [ ] Decide whether the new API warrants a release note.
- [ ] Add release-note entry only after final behavior is settled.
- [x] Re-check README for scope creep or duplicate claims.
- [ ] Confirm branch history is coherent before merge.
- [ ] Merge only after the full confidence pass is green.
