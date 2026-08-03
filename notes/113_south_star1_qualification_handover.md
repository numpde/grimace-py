# South Star 1 qualification handover

## Common code baseline

Coupled tetrahedral stereo continuation authority passed for Remote-A and
Remote-B at code baseline `6af09d9a40e907f6d2f93e017bb8e4693b39f81b`.
Both cases passed exact Rust support and completion counts, pinned support
digests, snapshot/resume, whole-asset facts-bound recertification, complete
retrieval of 3,848 branch proofs and 216 terminal proofs, and the pinned
replayed-operation and obligation-family contract. Every proof shard reported
zero calls to count-DAG, rich-support, or legacy-enumeration paths.

The materialized-authority zero-H and adjacent cases passed their public
asset, public recertification, offline-complete, support-artifact, support
reparse, continuation, and stereo-audit gates. The 34-case fast corpus passed
in four bounded non-stereo shards plus the full fast stereo fixture.

## Authority model

The continuation asset is the scalable authority for coupled stereo. Every
branch and terminal locator is independently facts-bound and producer-free
verified, while exact support counts, completion counts, support digest,
public proof retrieval, and snapshot/resume remain required product gates.

Exhaustive materialized support artifacts remain the stronger auxiliary
representation for small bounded cases. Qualification authority is declared
per product surface; it is never selected because a test timed out. The
continuation and materialized authorities therefore remain distinct while
sharing the same semantic and ledger contracts where applicable.

## Final checks

The focused Python suite passed with 42 tests and 4 expected skips. Python
compilation, `cargo fmt --check`, all 86 Rust library tests, and `git diff
--check` passed. The final pushed branch is clean and matches its origin.
