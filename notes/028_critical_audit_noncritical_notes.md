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
