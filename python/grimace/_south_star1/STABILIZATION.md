# South Star Stabilization Contract

This checkpoint describes the current internal South Star stereo
serialization and decoder contract. It is a stabilization boundary for ongoing
implementation work, not a public API promise.

## Guaranteed

- Offline finite support enumeration remains the reference semantics for the
  supported South Star fixtures.
- Prepared online enumeration agrees with the offline reference on support
  strings, support counts, and witness/EOS completion counts in the prepared
  matrix.
- Prepared molecules own and reuse their graph index, online traversal graph,
  root-domain cache, stereo templates, and atom/component metadata.
- Prepared matrix and workload probes enforce zero forbidden post-prepare graph,
  root-domain, stereo-template, facts-validation, policy-validation, and
  traversal-graph rebuild work.
- `iter_online_serializations(...)` is a bounded determinized support-string
  stream by default, while `collect_online_serializations(...)` is the
  materializing helper and `count_online_serializations(...)` is the exact
  count-only traversal.
- Prefix-query, cumulative decoder-walk, and branch-covering walk workloads
  check mode agreement for legal next-token sets, per-token completion counts,
  EOS availability, and EOS completion counts.
- `RESIDUAL_CONTINUATIONS` uses resumed snapshots, reduces root replay compared
  with `PREFIX_REPLAY` on the prepared workloads, and retains zero rendered
  suffix payload.
- Residual scheduler-frame resumption covers `RenderCursorFrame`,
  `PrefixEnumerationFrame`, `DirectionEnumerationFrame`, and
  `SupportMaximalFrame`.
- Retained or resumed residual snapshots must contain at least one known,
  dispatcher-handled resumable frame. Context-only frame stacks, unknown frame
  payloads, duplicate active resumable frames, and unhandled topmost resumable
  frames are invalid residual continuations.

## Not Guaranteed

- Full RDKit dialect parity.
- Public API stability for South Star internals.
- Arbitrary or enhanced stereo support beyond the current South Star scope.
- Minimal residual state, serialized snapshot compatibility, or formal
  asymptotic optimality.
- Cross-prepared snapshot provenance safety. A valid frame stack does not by
  itself prove the snapshot belongs to the same facts, policy, root option, or
  prepared object used for resume.

## Internal APIs

The following are internal and experimental:

- `OnlineSearchSnapshot`
- `OnlineResidualContinuation`
- `OnlineSearchVM.from_snapshot(...)`
- scheduler frame payload classes
- prepared matrix and workload helper modules

Production residual continuation construction goes through
`capture_residual_continuation(...)`. Direct construction of residual value
objects is a test and low-level helper escape hatch; retained-state audit and
snapshot resume still reject invalid values.

## Residual Validation Boundaries

Residual snapshot validity is checked at:

- continuation capture via `capture_residual_continuation(...)`;
- retained-state audit in residual state-size collection;
- snapshot resume via `OnlineSearchVM.from_snapshot(...)` and
  `resume_online_search_from_snapshot(...)`.

## Evidence Suites

The current stabilization evidence is defended by:

- prepared enumeration matrix tests;
- prepared prefix-query workload tests;
- cumulative prepared decoder-walk workload tests;
- branch-covering prepared decoder-walk workload tests;
- residual frame-stack audit tests;
- online serialization stream and count tests;
- South Star semantic runner and South Star unittest discovery.
