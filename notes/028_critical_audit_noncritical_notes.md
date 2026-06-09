# Critical audit noncritical notes

Audit start: 2026-06-08 19:57 EAT.
Second pass start: 2026-06-08 20:09 EAT.
Third pass start: 2026-06-08 20:44 EAT.
Fourth pass start: 2026-06-08 22:07 EAT.
Fifth pass start: 2026-06-09 07:51 EAT.

- `rust/src/rooted_stereo.rs` and `rust/src/bond_stereo_constraints.rs`
  intentionally mark several stereo carrier-selection and token-flip paths as
  "Suspicious current model". This is visible technical debt rather than a new
  critical finding: it is constrained to stereo edge-emission semantics, has
  known-gap/evidence framing, and should be replaced by first-class deferred
  stereo constraints when the stereo model is revisited.

Scope: record observations from the critical-issue audit that are below the
critical threshold. Critical issues are reported directly instead of recorded
here.

## Noncritical observations

- `scripts/prepared_mol_zstd_dictionary_generate.py --output-root ... --force`
  removes the computed artifact directory if it already exists. The Make/Compose
  lane binds the package-data directory explicitly and validates the path, so this
  is not a normal-lane escape, but the direct script surface is still powerful.
- `tests/rdkit_serialization/_support.py` intentionally treats some large
  isomeric deterministic-writer membership cases as RDKit drift when the
  deterministic output is not observed in a substantial rooted-random sample.
  The exact-support lane is separate and still fail-closed.
- The local checkout contains ignored generated artifacts (`__pycache__`,
  `target/`, and a host-built `python/grimace/_core...so`). They are not tracked,
  are excluded by `.dockerignore`/release validation, and are not part of the
  containerized correctness lanes, but they can affect ad hoc host imports.
- `PreparedMol.from_bytes()` selects shipped zstd dictionaries by frame
  dictionary ID and checks that loaded dictionary bytes report the manifest's
  dictionary ID. Manifest SHA-256/size are enforced by release artifact
  validation, not re-hashed on every runtime dictionary load.
- `tests/contract/` is not wired into `make ci`; the CI lanes run
  `tests/checks`, Rust tests, installed-package correctness, pinned RDKit
  parity, exact public invariants, and package tests. Contract tests are still
  useful SSoT/docs/schema guards, but they are not currently a default CI lane.
- Rust frontier prefix helpers rely on `debug_assert!` for prefix homogeneity
  and return the first prefix in release builds. Public/integration tests audit
  reachable decoder state graphs, so this is not an observed correctness break,
  but the invariant is not enforced at the exact helper boundary in optimized
  builds.
- `tests/contract/test_public_api_docs.py` checks several documented public
  signatures against live callables, but it does not include the token inventory
  function signatures even though those are documented on the API page. The
  option-inventory contract still covers the callable signatures themselves.
- The CI workflow ignores `README.md` and `docs/**` on pushes to `main`. That
  keeps docs-only pushes cheap, but it also means checked-in docs health checks
  run for docs changes only via pull requests, manual dispatch, or nearby code
  changes.
- Internal runtime-state audits deliberately allow an accepted state to still
  have outgoing transitions, because composed states can model acceptance and
  continuations separately. Public decoder tests cover terminal no-choice
  behavior on sampled paths and empty molecules, but they do not exhaustively
  assert that every reachable public terminal state has empty `next_choices`.
- `PreparedMol.from_bytes()` requires zstd frames to carry content size and
  checksum before decompression, but it does not apply an explicit
  Grimace-level maximum decompressed-size policy before calling zstd. The Rust
  raw-byte reader bounds internal vector lengths against the available payload,
  so this is not a format-safety break for valid raw payloads, but untrusted
  compressed input can still be a memory-pressure surface.
- The release workflow publishes the GitHub release and PyPI artifacts as
  sibling jobs after wheel/sdist validation. Both revalidate artifacts before
  acting, but PyPI publication does not depend on the GitHub release job
  succeeding, so a GitHub-release-only outage could still leave a PyPI-only
  release.
- `tests.rdkit_serialization.test_writer_support_counts` skips when the local
  RDKit version has no count fixture, unlike the pinned RDKit parity runner
  which fails closed on missing parity fixtures. Release lanes pin RDKit
  `2026.3.1`, so the current release path has fixtures, but the count-only
  evidence family is less fail-closed in ad hoc local environments.
- `pyproject.toml` declares `requires-python = ">=3.11"` while the release
  workflow builds Linux wheels only for CPython 3.12 and 3.13. Python 3.11 users
  would rely on the sdist/build path rather than a checked wheel. That is a
  packaging-support posture question, not a current correctness failure.
- Public `rootedAtAtom` coercion accepts arbitrary negative integers and the
  runtime treats them like all-roots. Documentation and examples use `-1`, so a
  stricter root sentinel policy would be cleaner, but this is not a critical
  correctness failure for valid documented inputs.
- `Cargo.toml` specifies `rustc-hash = "2.1.1"` while `Cargo.lock` currently
  resolves `rustc-hash` to `2.1.2`. The lockfile still pins builds, and
  container lanes use locked builds, but the manifest is a semver range rather
  than an exact dependency pin.
- `pyproject.toml` publishes lower-bounded runtime dependencies
  (`rdkit>=2026.3`, `zstandard>=0.25`) while release validation pins concrete
  fixture versions. That is normal for installability, but correctness evidence
  is tied to the pinned fixture versions and may not transfer automatically to
  future RDKit releases.
- `scripts/mine_rdkit_writer_support_count_candidates.py` and
  `scripts/generate_rdkit_writer_support_counts.py` are direct local tooling
  surfaces whose `--output` paths can point outside their usual report/fixture
  trees. They refuse to overwrite without `--force` and are not default
  container lanes, but their path policy is looser than the stricter
  Make/Compose lanes.
- `tests/helpers/rdkit_exact_small_support.py` loads `expected` and
  `expected_inventory` via raw indexing plus `list(...)` before the shared
  sorted-unique-string validator. Checked-in fixtures load and serializer
  fixtures exercise similar validation, but this family could fail with a
  less contextual `KeyError` or accept string iteration before rejecting
  noncanonical data. A small required-list helper would make the fixture schema
  tighter and more uniform.
- `python/grimace/_core.pyi` documents internal extension classes such as
  `PreparedSmilesGraph`, low-level rooted walkers, and `to_dict()`. They are
  underscore-module internals rather than `grimace.__all__` public API, but
  typed users can still discover and call them. That is useful for internal
  tests and debugging, but it keeps the low-level Rust surface easier to depend
  on than a strictly opaque boundary would.
- `MolToSmilesFlags` is an importable internal dataclass, and direct
  construction can bypass the option coercion performed by `make_flags()` and
  the public wrappers. Current public entrypoints route through the coercion
  path, but internal callers/tests can still create odd typed values that later
  get interpreted with `bool(...)` in writer-flag projection.
- `tests/rdkit_serialization/_support.py` has an intentional deterministic
  writer-membership escape hatch for large isomeric cases: when RDKit's
  deterministic output is not observed in a bounded rooted-random sample, the
  helper treats it as RDKit drift and returns without asserting Grimace
  membership. That matches the current random-writer contract, but the
  `rdkit_writer_membership` family name/test docstring read stronger than the
  effective assertion for those cases.
- `tests.helpers.public_runtime.prepared_input_variants()` is named broadly but
  currently returns only RDKit mol, reference prepared graph, and core prepared
  graph. PreparedMol raw/zstd equivalence is covered in separate PreparedMol and
  zstd contract tests, so this is not a missing release invariant, but the
  split matrix makes the boundary coverage less obvious during source review.
- `rust/src/rooted_stereo.rs` has Rust unit-test helpers that optionally import
  RDKit through the checkout's Python path and `.venv` site-packages when
  available. That keeps native stereo regression tests close to the Rust
  implementation, but it is more host-shaped than the strict container lanes
  and silently skips those checks when RDKit is absent.
- `tests/checks/test_docs_pages.py` validates Markdown links and image sources,
  but not raw HTML `<a href="...">` links. The current raw HTML links inspected
  during the pass resolve, but the link checker does not eliminate that whole
  class of docs drift.

## Solution planning

The entries below are planning notes only. They do not imply that every
observation should be fixed immediately. The goal is to make the clean path
explicit so future work can shrink surface area rather than add ad hoc patches.

### 1. Suspicious stereo carrier model

Issue: `rooted_stereo` still relies on precomputed directional-carrier phases
for coupled double-bond/ring-closure stereo. The code is isolated and known-gap
framed, but the model is not the long-term semantic boundary.

Serious alternatives:

- Leave the current model and only expand known-gap fixtures.
- Add more static special cases for the coupled stereo failures.
- Move all stereo enumeration to an RDKit oracle path.
- Replace stereo support with a post-generation filter.
- Introduce first-class online stereo constraints attached to emitted edge
  facts.

Principled direction: implement online stereo constraints. Directional bond
tokens should be selected when traversal emits an edge, with constraints carried
in decoder state and checked at ring closure and shared-carrier reuse. This
removes the static phase guess instead of adding more guesses.

Checklist:

- [ ] Define the minimal edge-emission facts needed for RDKit-equivalent
      directional stereo: directed edge, carrier edge, double-bond component,
      chosen slash/backslash token, and ring/open-chain ownership.
- [ ] Add a small Rust data type for pending stereo constraints independent of
      traversal mechanics.
- [ ] Make `bond_stereo_constraints.rs` produce immutable constraint metadata,
      not preselected carrier tokens.
- [ ] Teach rooted stereo advancement to update/check constraints when an edge
      token is emitted.
- [ ] Convert current known-gap cases into passing fixtures one cluster at a
      time.
- [ ] Delete the "Suspicious current model" path only after the known-gap lane
      proves equivalence.

### 2. Direct zstd dictionary generator `--force`

Issue: the direct script can delete the computed output artifact directory when
`--force` is passed. The Make/Compose lane validates the bind target, but the
script itself remains a powerful local surface.

Serious alternatives:

- Keep as-is and rely on Make/Compose as the supported lane.
- Remove `--force` entirely.
- Keep `--force`, but require the output root to be the package-data directory.
- Replace deletion with atomic write into a temp directory plus rename.
- Require an explicit `--replace-artifact ARTIFACT_ID` matching the computed
  artifact directory.

Principled direction: make replacement artifact-scoped and atomic. The script
should never recursively remove an arbitrary output root child just because it
matches a computed path; it should stage the artifact, validate it, then replace
only the exact expected artifact directory when the user explicitly names it.

Checklist:

- [ ] Replace boolean `--force` with `--replace-artifact YYYYMMDD_hash`.
- [ ] Write new artifacts to a sibling temp directory under the same output
      root.
- [ ] Validate the staged artifact before any replacement.
- [ ] If replacement is requested, require the requested artifact name to equal
      the computed artifact name.
- [ ] Rename the old artifact to a temporary backup, rename the staged artifact
      into place, then remove the backup.
- [ ] Update Make/Compose posture tests to require the explicit replacement
      contract.

### 3. Large isomeric writer-membership drift handling

Issue: some large isomeric deterministic RDKit outputs are treated as drift when
not seen in bounded rooted-random sampling. This is honest, but the assertion
strength differs from ordinary writer-membership cases.

Serious alternatives:

- Keep the escape hatch in the membership helper.
- Move those cases entirely to known gaps.
- Split the fixture family into strict membership and bounded-observation
  diagnostics.
- Add a separate `rdkit_deterministic_unobserved` classification.
- Require support-count or decoder-path evidence before accepting drift.

Principled direction: split the assertion family. Membership should mean
"RDKit output must be accepted." Bounded unobserved deterministic outputs should
live in a separate evidence family with a name that states the weaker contract.

Checklist:

- [ ] Identify every fixture path/case using the deterministic-unobserved escape.
- [ ] Add a new fixture family name for bounded deterministic-observation
      diagnostics.
- [ ] Move weaker cases out of `rdkit_writer_membership`.
- [ ] Make `rdkit_writer_membership` fail closed with no drift escape.
- [ ] Update docs/testing-fixtures.md counts and wording.
- [ ] Add a contract test that the strict membership helper contains no
      deterministic-unobserved bypass.

### 4. Ignored generated artifacts in local checkout

Issue: ignored build artifacts can affect ad hoc host imports, even though
container lanes and release validation exclude them.

Serious alternatives:

- Do nothing; strict workflows already avoid host state.
- Add a documented cleanup command.
- Make checks fail when ignored generated artifacts exist.
- Add a container-only host hygiene check that reports, but does not fail.
- Add a `make clean-host-artifacts` lane with explicit paths.

Principled direction: add an explicit cleanup lane and a non-default diagnostic.
Do not make normal correctness depend on host cleanliness; make cleanup easy and
auditable when someone chooses to use host imports.

Checklist:

- [ ] Define the generated artifacts that are safe to remove.
- [ ] Add `make clean-host-artifacts` with exact paths and no broad globs beyond
      already-ignored generated classes.
- [ ] Add a checks test that the cleanup lane and `.gitignore` agree.
- [ ] Add a short docs/development note warning that host imports can see stale
      ignored extension builds.
- [ ] Keep CI and release lanes containerized and independent of this cleanup.

### 5. Runtime dictionary manifest hash not rechecked

Issue: runtime dictionary loading validates dictionary ID against the manifest,
while SHA-256 and size are enforced by release validation rather than every
runtime load.

Serious alternatives:

- Keep runtime fast and trust package-data validation.
- Recompute SHA-256 on every dictionary load.
- Recompute SHA-256 only on first load before caching.
- Add an optional environment-driven integrity check.
- Move manifest validation into Rust and enforce all fields there.

Principled direction: hash once on first load, then cache. Dictionary loading is
rare enough that one SHA-256 pass is acceptable, and it makes installed package
integrity self-contained rather than depending only on release-time checks.

Checklist:

- [ ] Extend `_zstd_dictionary_from_manifest()` to read expected SHA-256 and
      size from the manifest.
- [ ] Validate dictionary byte length and SHA-256 before constructing
      `ZstdCompressionDict`.
- [ ] Keep the validated dictionary cached by dictionary ID.
- [ ] Add tests that a patched manifest or dictionary payload mismatch fails
      before decompression.
- [ ] Keep release artifact byte-equivalence checks; runtime validation is a
      defense-in-depth layer, not a replacement.

### 6. `tests/contract/` not in `make ci`

Issue: contract tests are useful SSoT/schema/docs guards, but the default CI
lane does not run them.

Serious alternatives:

- Leave contract tests manual.
- Fold contract tests into `make checks`.
- Add a new `make contract` lane and include it in `make ci`.
- Split slow contract tests from cheap static contract tests.
- Move high-value contract tests into `tests/checks`.

Principled direction: add `make contract` and include it in `make ci` if the
runtime cost is modest. Keeping a named lane preserves intent better than
silently mixing contracts into generic checks.

Checklist:

- [ ] Measure current `tests/contract` runtime inside the checks container.
- [ ] Add a `contract` service or reuse the checks service if dependencies are
      identical.
- [ ] Add `make contract` with the same strict container posture as checks.
- [ ] Include `contract` in `make ci` only if the measured cost is acceptable.
- [ ] Update docs/development/make-lanes.md.
- [ ] Add a posture test proving CI runs the contract lane.

### 7. Rust frontier prefix homogeneity uses `debug_assert!`

Issue: `frontier_prefix()` assumes prefix-homogeneous frontiers in release
builds and only asserts in debug builds.

Serious alternatives:

- Keep as a debug-only internal invariant.
- Convert the helper to return `PyResult<String>` and check always.
- Add a separate checked helper only for public paths.
- Store prefix once in the frontier object and avoid recomputation.
- Encode prefix homogeneity in the frontier type constructor.

Principled direction: encode it at the boundary that creates merged/frontier
states. If a frontier can only be constructed through a checked constructor,
prefix reads can stay cheap without relying on a debug-only assertion.

Checklist:

- [ ] Find all `frontier_prefix()` callers and frontier construction sites.
- [ ] Introduce a small frontier wrapper or constructor that validates
      nonempty/homogeneous prefix where required.
- [ ] Replace raw `Vec<State>` frontiers in decoder modes where practical.
- [ ] Keep low-level tests for malformed mixed-prefix frontiers.
- [ ] Remove or downgrade the standalone `frontier_prefix()` helper once the
      invariant lives in construction.

### 8. API docs signature coverage misses token inventories

Issue: API docs signature tests omit `MolToSmilesTokenInventory` and
`MolToSmilesTokenInventorySuperset`, even though those signatures are documented.

Serious alternatives:

- Rely on the option-inventory signature contract.
- Add the two missing functions to the docs signature test.
- Generate the API signature list from `grimace.__all__`.
- Generate docs signatures from live callables.
- Move all signature tests to a single manifest.

Principled direction: keep explicit docs signature tests but make the list
complete for every documented callable. Automatic docs generation is larger than
needed; explicit coverage is clearer here.

Checklist:

- [ ] Add the two token inventory functions to `test_public_api_docs.py`.
- [ ] Add a guard that every backtick signature in `docs/api/python.md` is
      matched by one test case.
- [ ] Keep option-inventory tests as the SSoT for default flag lists.
- [ ] Avoid generating docs from tests; this is a small completeness fix.

### 9. Docs-only pushes skip CI

Issue: `README.md` and `docs/**` changes on `main` do not trigger CI, so docs
health checks depend on PRs, manual dispatch, or nearby code changes.

Serious alternatives:

- Keep current path-ignore for cheap docs-only pushes.
- Remove `paths-ignore` and run full CI for docs changes.
- Add a lightweight docs-only workflow.
- Add branch protection requiring PR checks for docs changes.
- Run only docs/checks lanes on docs paths.

Principled direction: add a lightweight docs workflow for docs/README changes.
Do not run full Rust/package lanes for text-only changes; do run link/build
checks so published Pages drift is caught.

Checklist:

- [ ] Add `.github/workflows/docs.yml` for `README.md`, `docs/**`, and docs
      config/script paths.
- [ ] Run `make docs` and the docs/checks subset in the strict container lane.
- [ ] Pin actions and use read-only permissions like CI.
- [ ] Update workflow posture tests to cover the docs workflow.
- [ ] Keep main CI path-ignore if the docs workflow covers docs health.

### 10. Accepted internal states may have outgoing transitions

Issue: internal composed states can be accepted and still have outgoing
transitions. Public terminal states are expected to have no next choices, but
that invariant is not exhaustively asserted across reachable public states.

Serious alternatives:

- Document the internal distinction and leave tests as-is.
- Assert only selected terminal examples.
- Exhaustively audit reachable public decoder states for terminal/no-choice.
- Change internal `is_terminal` to mean no outgoing choices.
- Split internal acceptance from public terminality.

Principled direction: split names semantically. Internal state should expose
`is_accepting()`; public decoder should expose `is_terminal` after applying the
public stopping rule. Tests should assert the public invariant exhaustively.

Checklist:

- [ ] Rename internal protocol method from `is_terminal()` to `is_accepting()`
      or add a wrapper that makes the distinction explicit.
- [ ] Define public terminality as accepting with no public continuation.
- [ ] Audit branch-preserving and determinized public states over representative
      reachable state graphs.
- [ ] Add tests: every reachable public state with `is_terminal` has
      `next_choices == ()`.
- [ ] Keep walker sampling stopping on acceptance if that remains the desired
      draw semantics.

### 11. PreparedMol zstd decompression has no Grimace size cap

Issue: zstd frames require checksum and content size, but Grimace does not apply
an explicit maximum decompressed-size policy before decompression.

Serious alternatives:

- Rely on raw binary bounds and zstd content size.
- Reject frames without content size, but still allow any size.
- Add a conservative hard maximum raw PreparedMol payload size.
- Add a caller-configurable max size.
- Stream decompression into the Rust binary reader.

Principled direction: add a fixed public safety cap first. A configurable cap is
surface area; streaming into Rust is larger work. A hard cap protects untrusted
input while keeping `from_bytes()` simple.

Checklist:

- [ ] Choose an initial cap based on observed prepared-mol sizes plus large
      headroom.
- [ ] Parse zstd frame content size and reject sizes above the cap before
      decompression.
- [ ] After decompression, verify actual length equals the announced size when
      available.
- [ ] Add tests for over-cap frame metadata and over-cap decompressed payloads.
- [ ] Document the cap under PreparedMol bytes limitations.
- [ ] Revisit streaming only if legitimate workloads exceed the cap.

### 12. GitHub release and PyPI publish are sibling jobs

Issue: PyPI can publish even if the GitHub release job fails, because both jobs
depend only on wheel/sdist validation.

Serious alternatives:

- Keep sibling jobs; both validate artifacts independently.
- Make PyPI depend on GitHub release.
- Make GitHub release depend on PyPI.
- Add a final verification job after both.
- Publish to PyPI first, then create GitHub release only after PyPI success.

Principled direction: make PyPI depend on GitHub release if GitHub release notes
and downloadable artifacts are part of the release contract. This avoids a
PyPI-only public release when the GitHub release page fails.

Checklist:

- [ ] Decide whether GitHub release is release-contractual or informational.
- [ ] If contractual, change `publish-pypi.needs` to include `release`.
- [ ] Keep PyPI artifact revalidation even when it depends on release.
- [ ] Add workflow posture test for release ordering.
- [ ] Document the order in release notes/process docs.

### 13. Writer support-count tests skip on missing local RDKit fixture

Issue: support-count tests skip for unpinned local RDKit versions. Release lanes
pin RDKit, but ad hoc local evidence is less fail-closed.

Serious alternatives:

- Keep skip behavior for local convenience.
- Fail closed always.
- Fail closed only under an environment variable.
- Split local discovery from release correctness.
- Route count tests only through the pinned parity runner.

Principled direction: fail closed in release/package runners, skip only in
plain local discovery. This matches the existing pinned parity posture without
making every local RDKit upgrade unusable.

Checklist:

- [ ] Add an environment variable or runner context for strict fixture mode.
- [ ] Make `tests.run_installed_package_correctness` and pinned parity set
      strict mode.
- [ ] In strict mode, missing support-count fixtures should fail, not skip.
- [ ] Keep ordinary `unittest discover` behavior skippable for exploratory
      environments.
- [ ] Document the difference in testing-fixtures.md.

### 14. Python 3.11 metadata without 3.11 wheels

Issue: package metadata allows Python 3.11, but release wheels are only cp312
and cp313 Linux. Python 3.11 users take the sdist path.

Serious alternatives:

- Keep `>=3.11` and document sdist-only 3.11.
- Raise `requires-python` to `>=3.12`.
- Add cp311 wheels.
- Add cp311 source-build CI only.
- Mark classifiers only for built wheels and leave metadata broad.

Principled direction: either build cp311 wheels or raise metadata. The no-regret
choice depends on whether 3.11 is an intended supported runtime. Metadata should
not imply support that release testing does not exercise.

Checklist:

- [ ] Decide whether Python 3.11 is supported or merely possible.
- [ ] If supported, add cp311 to release wheel matrix and artifact validator.
- [ ] If not supported, set `requires-python = ">=3.12"` and remove 3.11 docs
      expectations.
- [ ] Update runtime docs release matrix.
- [ ] Add workflow posture tests for whichever policy is chosen.

### 15. Arbitrary negative `rootedAtAtom` means all-roots

Issue: public coercion accepts any negative integer as unrooted/all-roots, while
docs use `-1`.

Serious alternatives:

- Preserve RDKit-like acceptance of any negative root.
- Normalize all negative values to `-1` at the public boundary.
- Reject values below `-1`.
- Warn for values below `-1`.
- Keep runtime broad but document the exact behavior.

Principled direction: normalize at the boundary and document `-1` as the only
semantic sentinel. Rejection could break users for little gain; silent
normalization keeps compatibility while avoiding internal mixed sentinels.

Checklist:

- [ ] Ensure public option coercion converts any negative root to `-1`.
- [ ] Keep tests proving `-2` and `-3` behave like `-1` if compatibility is
      intentionally preserved.
- [ ] Add docs wording: negative roots are normalized to `-1`; use `-1`.
- [ ] Keep Rust prepared root mapping strict for explicit rooted fragment calls.

### 16. Cargo manifest dependency range differs from lock

Issue: `Cargo.toml` uses a semver range for `rustc-hash`, while `Cargo.lock`
pins the actual build.

Serious alternatives:

- Keep semver range plus locked builds.
- Pin exact dependency versions in `Cargo.toml`.
- Use `=2.1.2` only for security-sensitive dependencies.
- Add a check that manifest minimum equals lock version.
- Vendor Rust dependencies.

Principled direction: keep semver ranges for Rust library dependencies, but add
a posture note/check that release/container builds use `--locked`. Exact
manifest pins are less idiomatic and can make dependency updates noisier.

Checklist:

- [ ] Confirm every container/release cargo invocation uses `--locked`.
- [ ] Add a docs note that `Cargo.lock` is the release dependency SSoT.
- [ ] Add a check that `Cargo.lock` is present and not ignored.
- [ ] Do not exact-pin manifest dependencies unless a specific supply-chain
      reason appears.

### 17. Runtime dependency lower bounds versus pinned evidence

Issue: install metadata allows future RDKit/zstandard versions, while evidence
is pinned to concrete fixture versions.

Serious alternatives:

- Keep lower bounds and document evidence pinning.
- Upper-bound RDKit and zstandard.
- Exact-pin runtime dependencies.
- Runtime-warn when RDKit version lacks fixtures.
- Add version-keyed evidence expansion for new RDKit releases.

Principled direction: keep lower bounds but make evidence scope explicit and
fail closed in strict correctness lanes. Exact pins would harm installability;
silent parity claims across future RDKit versions would be worse.

Checklist:

- [ ] Ensure docs state parity evidence is RDKit-versioned.
- [ ] Make strict test runners fail if the active RDKit version lacks fixtures.
- [ ] Add a release checklist item to add fixtures before claiming a new RDKit
      version.
- [ ] Consider an upper bound only if a known future RDKit breaks runtime, not
      merely because evidence is versioned.

### 18. Mining/count scripts allow broad output paths

Issue: direct local scripts can write outside their usual report/fixture trees.
They are not default lanes, but their path policy is looser than Make/Compose.

Serious alternatives:

- Leave local tooling flexible.
- Restrict outputs to one fixed directory.
- Add `--allow-outside-repo` for exceptional use.
- Require output parent to exist and be under an approved root.
- Move write policy into a shared script helper.

Principled direction: use a shared output-path policy helper. Default outputs
should stay under repo-approved roots; exceptional outside paths should require
an explicit opt-in flag.

Checklist:

- [ ] Add a small `scripts/_path_policy.py` helper for repo-root resolution and
      approved output roots.
- [ ] Apply it to mining/count scripts.
- [ ] Require `--allow-outside-repo` for arbitrary paths.
- [ ] Preserve `--force` overwrite checks.
- [ ] Add unit tests for symlink, parent traversal, and outside-repo paths.

### 19. Exact small-support loader uses raw indexing/list conversion

Issue: fixture loading can produce less contextual errors and briefly accepts
string iteration before shared validation rejects it.

Serious alternatives:

- Leave as-is because fixture tests cover current data.
- Add a local required-list helper.
- Reuse the serializer fixture list validator.
- Move all fixture schema helpers into one module.
- Convert fixtures to a typed schema library.

Principled direction: centralize small typed JSON helpers in existing fixture
helper modules. Avoid a schema framework; use simple required-list validators
with contextual errors.

Checklist:

- [ ] Add `required_string_list()` or equivalent beside existing
      `required_string`/`required_int` helpers.
- [ ] Require list type before iterating.
- [ ] Validate sorted/unique string lists in the helper when requested.
- [ ] Use the helper in exact small-support loaders.
- [ ] Add loader tests for missing field, string-as-list, duplicate, and
      unsorted cases.

### 20. `_core.pyi` exposes low-level internals

Issue: type stubs document internal extension classes, making private Rust
objects easy for typed users to discover.

Serious alternatives:

- Keep full internal stubs for internal tests and typed development.
- Remove `_core.pyi`.
- Split public and private stubs.
- Mark internal classes with leading underscores in Rust/Python exports.
- Move low-level extension APIs behind a separate private module name.

Principled direction: keep internal stubs but make the boundary explicit. The
module is private (`grimace._core`), so hiding stubs would hurt maintainability
more than it helps. The public contract should be enforced by `grimace.__all__`
and docs.

Checklist:

- [ ] Add a top-of-file comment in `_core.pyi` stating it is private extension
      typing, not public API.
- [ ] Keep public docs limited to top-level `grimace` symbols.
- [ ] Add/keep tests asserting private `_core` symbols are not re-exported.
- [ ] Avoid adding new `_core` methods unless a public wrapper or internal test
      needs them.

### 21. Direct `MolToSmilesFlags` construction bypasses coercion

Issue: the internal dataclass can be constructed with odd typed values and later
interpreted by bool/int conversions.

Serious alternatives:

- Keep it as a plain internal dataclass.
- Add `__post_init__` validation/coercion.
- Make the dataclass private and expose only `make_flags()`.
- Replace the dataclass with a frozen named tuple from the option parser.
- Move flags into Rust.

Principled direction: make `MolToSmilesFlags` validate exact internal types in
`__post_init__`, and keep coercion in `make_flags()`. Direct construction should
be allowed for tests, but it should not create invalid internal states.

Checklist:

- [ ] Add `__post_init__` with exact `bool` checks for bool fields and exact
      `int` check for `rooted_at_atom`.
- [ ] Keep public/RDKit-like coercion in `_mol_to_smiles_options.py`.
- [ ] Update tests that intentionally construct flags to use valid internal
      types.
- [ ] Add tests that direct invalid construction fails early.
- [ ] Avoid duplicating defaults; keep defaults sourced from the option
      inventory tests.

### 22. Writer-membership family name overstates weaker cases

Issue: the same deterministic-unobserved escape described in item 3 also makes
the fixture-family name/docstring read stronger than the effective assertion.

Serious alternatives:

- Rename the whole family to a weaker name.
- Split strict and weak cases.
- Keep name but add in-test comments.
- Promote weak cases to known gaps.
- Add a reporting field for assertion strength.

Principled direction: same as item 3: split strict and weaker evidence. Names
should encode assertion strength; comments are not enough.

Checklist:

- [ ] Reuse the item 3 migration.
- [ ] Make `rdkit_writer_membership` strictly assert membership.
- [ ] Add a separate family for bounded deterministic-observation diagnostics.
- [ ] Update correctness coverage report to count assertion strengths
      separately.
- [ ] Update docs so readers can see which cases are strict parity evidence.

### 23. `prepared_input_variants()` name hides matrix split

Issue: the helper name sounds broad but omits PreparedMol raw/zstd variants,
which are covered elsewhere.

Serious alternatives:

- Rename the helper to describe its actual graph variants.
- Add PreparedMol variants to the helper.
- Split helpers by boundary: graph variants vs PreparedMol variants.
- Delete the helper and inline cases.
- Add a higher-level registry for all public input variants.

Principled direction: split by boundary and name narrowly. Graph-prepared
variants and PreparedMol serialization variants exercise different contracts;
one broad helper obscures that.

Checklist:

- [ ] Rename `prepared_input_variants()` to `prepared_graph_input_variants()`.
- [ ] Add a separate `prepared_mol_input_variants()` only where raw/zstd bytes
      equivalence is the test subject.
- [ ] Update test names to state which boundary they cover.
- [ ] Add a small docs/testing note explaining the two prepared-input matrices.

### 24. Rust stereo tests optionally import host RDKit

Issue: some Rust unit-test helpers import RDKit from checkout/`.venv` when
available and silently skip otherwise.

Serious alternatives:

- Keep optional native tests as local-only convenience.
- Remove Python/RDKit imports from Rust unit tests.
- Convert those tests into Python integration tests.
- Make the Rust tests require RDKit in the container lane.
- Add explicit ignored Rust tests for RDKit-backed checks.

Principled direction: move RDKit-dependent cases to Python integration/parity
tests and keep Rust unit tests pure Rust. Rust tests should not depend on host
Python path state.

Checklist:

- [ ] Identify Rust tests using `prepared_graph_from_smiles()`.
- [ ] Port those cases to Python integration tests that prepare graphs through
      the normal containerized RDKit path.
- [ ] Keep pure Rust tests using hand-built `PreparedSmilesGraphData`.
- [ ] Delete optional `.venv` path injection from Rust tests.
- [ ] Add a contract test or grep check preventing future Rust tests from
      importing `rdkit` via Python.

### 25. Docs checker misses raw HTML links

Issue: docs link checks cover Markdown links and image sources, but not raw HTML
anchors.

Serious alternatives:

- Avoid raw HTML in docs.
- Extend the existing checker with a simple HTML href regex.
- Parse generated HTML with a standard parser.
- Use a link-checking tool in the docs container.
- Add a Jekyll build plugin.

Principled direction: avoid raw HTML where Markdown suffices, and extend the
existing source checker for the few raw HTML attributes that remain. This keeps
the check offline, dependency-light, and close to the docs source.

Checklist:

- [ ] Replace raw HTML links with Markdown where possible.
- [ ] Extend `tests/checks/test_docs_pages.py` to collect raw `href` and `src`
      attributes from Markdown files.
- [ ] Resolve local targets using the same rules as Markdown links.
- [ ] Reject external raw HTML links unless explicitly allowlisted.
- [ ] Add a test fixture or real docs case proving raw HTML links are checked.
