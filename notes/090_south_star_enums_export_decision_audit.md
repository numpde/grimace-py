# South Star EnumS Export Decision Audit

Branch: `south-star`

Date: 2026-05-22

Task: `South Star 248: Re-audit EnumS export decision`

## Purpose

Re-check the `MolToSmilesEnumS` export decision after:

- the proposed public surface was specified;
- the private dry-run wrapper was added;
- documentation, performance, and release-note review gates became checkable.

This note is not a release decision. It records the current evidence and the
remaining decision boundary.

## Current Evidence

Current readiness snapshot:

- public API promotion gates: `13`;
- checked review artifacts: `3`;
- first-domain cases: `6`;
- expanded-support cases: `74`;
- derived-support cases: `1`;
- supported feature areas: `35`;
- unsupported fail-fast categories: `16`;
- unified-reference promotion checks: `80`;
- promoted checks: `80`;
- public API blocker case ids: none;
- shared-pipeline promotion candidates: `80`.

The checked review artifacts are:

- `documentation_contract`: `docs/enum-s.md`;
- `performance_evidence_boundary`: `docs/enum-s.md`;
- `release_notes_scope`: `docs/release-note-checklists/enum-s-export.md`.

Verification run for this audit:

```bash
PYTHONPATH=python:. python3 -m unittest tests.run_south_star_package_readiness -q
```

Result: `159` tests passed in `89.746s`.

The full South Star suite was also run immediately before this audit slice:

```bash
PYTHONPATH=python:. python3 -m unittest tests.run_south_star_semantics -q
```

Result: `292` tests passed in `122.516s`, with `1` skipped.

## Decision Boundary

The current gate evidence no longer points to an obvious technical blocker for
exporting a narrow first public `MolToSmilesEnumS` surface.

The remaining question is an explicit API/product decision:

- export `MolToSmilesEnumS(mol) -> tuple[str, ...]` now, using the private
  dry-run wrapper as the implementation spine; or
- hold the surface private despite passing the current promotion gates.

If exporting, the first public surface should stay narrow:

- input: RDKit `Mol`;
- output: `tuple[str, ...]`;
- fixed default South Star policy;
- no public diagnostics;
- fail fast with `SouthStarUnsupportedFeatureError`;
- explicitly semantic support, not RDKit writer parity.

The export release notes must follow
`docs/release-note-checklists/enum-s-export.md`.

## Recommendation

Open a Decision row rather than silently exporting in the next implementation
slice. The evidence supports export readiness under the current gate, but
public API exposure is a commitment beyond test pass/fail status.

