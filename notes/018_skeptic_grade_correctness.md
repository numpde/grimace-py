# Skeptic-Grade Correctness Evidence

This note scores the current correctness story by the standard of a hard
skeptic, not by ordinary project usefulness.

Scale:

- `0`: entirely absent.
- `2`: ad hoc evidence exists, but it is not organized as a claim.
- `5`: useful evidence exists, but a reviewer still has to infer too much.
- `8`: convincing for the current public scope, with remaining known limits.
- `10`: implemented completely convincingly: executable, documented,
  generated or drift-checked where possible, and hard to misinterpret.

## Target Questions

A skeptical reviewer should be able to answer:

1. What is the exact public claim?
2. Which independent oracle checks that claim?
3. Which finite domains are exhausted rather than sampled?
4. Which known failures are intentionally outside the passing claim?
5. Which reports are generated from checked-in evidence rather than written by
   hand?

## Maturity Checklist

- `[5/10]` Executable public spec.
  Current: docs and tests cover the supported runtime surface, but the
  contract is spread across runtime tests, docs, and API behavior.
  To reach 10: one compact spec suite exercises `canonical=False`,
  `doRandom=True`, supported writer flags, roots, disconnected fragments,
  token boundaries, prepared-molecule equivalence, and unsupported modes.

- `[5/10]` Independent reference oracle.
  Current: a slow reference path exists for important bounded checks, but its
  role is not presented as the audit oracle for the optimized runtime.
  To reach 10: the reference oracle is clearly isolated from optimized runtime
  code and checks exact support, decoder reachability, determinized
  equivalence, inventory, and all-roots union for bounded cases.

- `[2/10]` Exhaustive generated small domains.
  Current: evidence is mostly curated fixtures, upstream-derived cases, and
  mined candidates. Some small supports are exact, but generated finite
  domains are not the main proof strategy.
  To reach 10: at least one connected nonstereo finite graph domain is
  exhausted, then extended to disconnected molecules and selected writer
  flags, with checked-in domain reports and explicit skip rules.

- `[6/10]` Metamorphic invariants.
  Current: exact small-support fixtures already exercise enum, decoder,
  determinized decoder, inventory, superset, prepared round trip, and
  deviation acceptance. The invariant matrix is implicit.
  To reach 10: each affordable fixture family declares which invariants it
  runs, and checks enforce enum/decoder equivalence, determinized language
  equivalence, inventory equality, superset containment, all-roots union,
  prepared round trip, and deviation boundaries.

- `[4/10]` Minimized known gaps.
  Current: known stereo gaps are pinned and executable, but they are not yet
  reduced into named minimal mechanisms.
  To reach 10: every known gap has an upstream/source case, smallest
  reproducer found so far, suspected implementation family, grouping by
  failure mechanism, and a clear promotion path after a fix.

- `[1/10]` RDKit version matrix.
  Current: public parity evidence is pinned to RDKit `2026.03.1`; release
  reproducibility is good, cross-version confidence is mostly untested.
  To reach 10: an explicit validation lane runs checked-in fixtures against
  selected RDKit versions and reports unchanged fixtures, intentional
  version deltas, and version-pinned behavior.

- `[6/10]` Generated evidence documentation.
  Current: coverage reports are generated and baseline checks catch drift, but
  user-facing Markdown still contains static snapshots.
  To reach 10: user-facing counts and coverage tables are generated from
  fixture loaders and the serializer ledger, or static snapshots are fully
  drift-checked by tests.

- `[4/10]` Negative boundary tests.
  Current: deviation diagnostics test important cases, including string versus
  external-token boundaries, but negative coverage is not organized by
  boundary type.
  To reach 10: negative tests are grouped by wrong root, fragment boundary,
  token boundary, ring closure, branch close, stereo direction token,
  unsupported options, and expected deviation position.

## Priority Order

| Rank | Area | Why this comes next |
|---:|---|---|
| 1 | Exhaustive generated small domains | This shifts evidence from "many examples" to "bounded domain exhausted under a clear spec." |
| 2 | Independent reference oracle | Generated domains only convince if the comparison oracle is simple, auditable, and separate from the optimized path. |
| 3 | Executable public spec | The spec defines the boundary for generated tests, invariants, and negative cases. |
| 4 | Metamorphic invariants | These catch cross-API inconsistencies after the spec and oracle are clear. |
| 5 | Minimized known gaps | This turns visible debt into actionable mechanisms rather than an opaque failure list. |
| 6 | Negative boundary tests | These are strongest once the accepted language and token boundary contract are explicit. |
| 7 | Generated evidence documentation | Important for trust, but it should summarize real evidence rather than lead it. |
| 8 | RDKit version matrix | Valuable, but best added after the current-version correctness story is tighter. |

## First No-Regret Step

Start with one generated exhaustive small-domain lane for connected nonstereo
molecules, with a deliberately small atom/bond alphabet.

It should:

- generate a finite graph domain
- filter to RDKit-parseable molecules
- cap exact support size
- compare optimized runtime output against the independent reference oracle
- run enum, decoder, determinized decoder, inventory, superset, prepared
  round trip, and deviation invariants
- produce a checked-in report of domain size, skipped cases, and covered
  features

This is the first step because it raises both the weakest score and the most
important kind of evidence: exhaustive bounded-domain correctness.
