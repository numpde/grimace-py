# South Star Export Maturity Criteria

Branch: `south-star`

Date: 2026-05-21

Task: `South Star 175: Define EnumS export maturity criteria`

## Purpose

`MolToSmilesEnumS` now has a coherent private fixture surface, but coherence is
not the same as public package readiness. This note turns the export question
into explicit criteria.

## Export Postures

### Private Continuation

Use this posture when the work is still primarily model-building.

Criteria:

- current fixtures are unified-reference-backed;
- package-readiness tests pass;
- unsupported categories are explicit;
- important normal chemistry surfaces are still gated;
- naming, docs, and release posture are still being explored.

This is the current posture.

### Experimental Export

Use this posture only if users are expected to try the semantic enumerator
while accepting sharp scope limits.

Required criteria:

- every exported feature family is unified-reference-backed;
- every output passes grammar and semantic parse-back checks;
- unsupported categories fail before enumeration with stable diagnostics;
- docs present the unsupported frontier as part of the API contract;
- release notes say this is semantic enumeration, not RDKit writer parity;
- performance claims are absent unless a semantic benchmark artifact is updated
  for the release;
- the public name, likely `MolToSmilesEnumS`, is explicitly provisional.

Experimental export still needs an explicit Decision row because it changes
the product contract even if the implementation already passes tests.

### Narrow Feature-Family Export

Use this posture only if the project wants a smaller public promise than the
full private surface.

Required criteria:

- exported feature families are listed explicitly;
- non-exported private feature families stay private even if tests pass;
- the API can report unsupported categories without implying general SMILES
  support;
- examples avoid suggesting RDKit writer parity;
- package docs explain how the narrow export differs from both
  `MolToSmilesEnum` and the broader private South Star work.

This may be attractive if the first public value is educational or diagnostic,
not production enumeration.

### Stable Public Export

Use this posture only after the semantic surface is broad enough that users do
not mostly experience fail-fast boundaries on ordinary molecules.

Required criteria:

- broad atom/bond text coverage;
- broad ring and aromatic coverage, or explicit long-term exclusions;
- resolved public naming;
- stable error taxonomy;
- release-facing docs and examples;
- CI/readiness coverage for the exported package path;
- benchmark language that is either absent or backed by release-grade semantic
  benchmark evidence.

The current branch is not near this posture.

## Current Recommendation

Keep `MolToSmilesEnumS` private.

The reason is not current fixture correctness. The current fixture surface is
clean. The reason is that important ordinary surfaces are still intentionally
gated:

- fused aromatic systems;
- aromatic modified atoms;
- aromatic directional overlays;
- broader non-organic bracket atom text beyond the first `Si`/`Se` slice;
- metal and dative chemistry;
- query semantics;
- some ring-system/stereo interactions;
- under-specified non-token stereo facts.

The next useful Decision row should be opened only when one of these is true:

- the project wants an experimental export despite the frontier;
- the project wants a narrow feature-family export;
- the implementation surface becomes broad enough to reassess stable export.

Until then, export work should remain Backlog implementation or documentation
work, not a standing Decision.
