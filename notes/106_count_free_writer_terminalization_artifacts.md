# Count-free writer terminalization artifacts

Writer terminalization is now a local proof boundary. A
`writer_terminalization_artifact` v1 contains exactly the source snapshot, one
selected terminal projection, and the shared terminal-support payload. Building
or checking it does not enumerate support strings or construct count envelopes
or count DAGs.

`writer_support_artifact` v10 embeds the same terminal-support payload. Its new
`WriterTerminalizationTerm` commits to graph completion, source and finalized
states, residual snapshots, terminal work, lifecycle evidence, and execution
capabilities. Terminal obligation credit is registered only after semantic
replay; `terminal_clean`, `is_discharged`, operation labels, and digest links do
not independently establish proof.

The live kernel records `WriterLocalOrderClosed` even when EOS makes no residual
change. Consequently `stereo_mode="noop"` means residual no-op, not whole-state
identity: the verifier still reconstructs the exact local-order closure. The
tetra mode additionally replays restriction, propagation, factor discharge,
projection, and the empty finalized residual snapshot.

| Serialized field | Independent authority |
| --- | --- |
| Source state | source snapshot cursor |
| Finalized state | selected terminal projection |
| Active atom | source writer state |
| Graph completion | facts and source graph/ring state |
| Graph-work digests | branch-local terminal graph manifests |
| Stereo mode | source residual and terminal work |
| Residual snapshots | source and finalized writer states |
| Terminal residual work | producer-free residual-store replay |
| Lifecycle evidence | active-atom local-order evolution and linked work |
| Capabilities | operations actually replayed |
| Multiplicity | selected support and source cursor weight |

The support-artifact, branch-transition-artifact, count schemas, and production
budgets otherwise retain their existing ownership boundaries. Branch artifacts
remain v2; count schemas are unchanged.

## Completion baseline

Before the exact evidence reconstruction work, the complete legacy facts-bound
module ran 111 tests in 4,339.457 seconds. It had no semantic failures and one
compatibility error:

| Test | Expected | Actual | First changed function |
| --- | --- | --- | --- |
| `test_linked_lifecycle_requires_replayed_residual_work` | linked lifecycle remains unchecked without replayed residual work | `TypeError` for a missing `replayed_terminal_graph_digests` keyword | `_obligation_manifest_checked` |

The helper now treats an omitted terminal-graph replay set as empty, preserving
the pre-terminalization direct-call contract without granting proof credit.
