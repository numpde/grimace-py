# South Star Post Atom-Text Expansion Frontier

Branch: `south-star`

Date: 2026-05-22

Task: `South Star 200: Recompute unsupported frontier after atom-text expansion`

## Purpose

This note refreshes the unsupported frontier after the recent atom-text slices:

- bracket-only main-group atom text for `As`, `Ge`, and `Sb`;
- mapped bracket-only aromatic tellurium text;
- fixture coverage for the newly supported atom-text cases.

The goal is to keep the next work driven by semantic family boundaries rather
than by stale unsupported examples.

## Probe

Reusable inventory script:

`tmp/exploration/frontier/001_recompute_post_atom_text_expansion.py`

The script recomputes support-gate triage for the adversarial corpus and counts
feature areas in the expanded South Star fixture corpus.

## Adversarial Triage Before South Star 203

The adversarial corpus now has:

- candidates: 22;
- supported by gate: 19;
- unsupported by gate: 3.

Unsupported candidates:

| Candidate | Source | Unsupported categories | Boundary targets |
| --- | --- | --- | --- |
| `unsupported_feature_triggers:query_unspecified_bond` | `C~C` | `query_bond`, `unsupported_bond_type` | `query_bond`, `unsupported_bond_type` |
| `unsupported_feature_triggers:dative_bond` | `N->[O]` | `dative_bond`, `unsupported_bond_type` | `dative_bond` |
| `unsupported_feature_triggers:unsupported_isotope_aromatic_element` | `[15te]1cccc1` | `aromatic_ring_surface` | `aromatic_ring_surface` |

Supported boundary targets now include both aromatic selenium and aromatic
tellurium map-modifier cases:

- `aromatic_selenium_text`: 2 adversarial candidates;
- `aromatic_tellurium_text`: 2 adversarial candidates.

## Fixture Surface

The expanded fixture corpus now has these atom-text-relevant counts:

- `aromatic_selenium_text`: 2;
- `aromatic_tellurium_text`: 2;
- `modified_aromatic_atom_text`: 6;
- `non_organic_bracket_atom_text`: 7;
- `charged_atom_text`: 3;
- `radical_atom_text`: 3;
- `combined_atom_text_modifiers`: 2;
- `atom_map_text`: 1;
- `isotope_atom_text`: 1.

This means the old frontier witness `[GeH3]C` is no longer a blocker: it is now
covered by pinned bracket-only main-group atom text. Likewise `[te:7]1cccc1` is
no longer an unsupported aromatic-ring-surface witness.

## Frontier Split Before South Star 203

The current live unsupported frontier is narrower than notes/055:

1. Query or unspecified bond semantics: `C~C` remains outside the fixed-molecule
   SMILES support model.
2. Dative/coordination semantics: `N->[O]` remains outside ordinary bond text
   and is still coupled to dative-specific chemistry.
3. Non-map modifiers on bracket-only aromatic elements: `[15te]1cccc1` is the
   live small witness for isotope-modified bracket-only aromatic element text.

The remaining broad semantic frontier from earlier notes still exists, but it
is not represented by the current small adversarial unsupported set:

- larger ring-system/stereo interactions;
- aromatic directional overlays;
- query/metal/coordination semantics;
- non-token or under-specified stereo facts.

Those should stay separate work streams. The atom-text work did not solve them;
it only removed stale text-policy blockers.

## Recommended Next Slices Before South Star 203

Immediate next work should be one of these:

1. Probe non-map modifiers on bracket-only aromatic elements. This is already
   represented by `South Star 201` and should decide whether isotope, charge,
   explicit-H, radical, or chiral fields compose with the same atom-text
   modifier model.
2. Decide the benchmark artifact coverage policy. This is represented by
   `South Star 202`; the current guard is useful but forces noisy timing
   artifact churn for every semantic fixture addition.
3. After that, return to a larger semantic family rather than more atom-text
   breadth: aromatic directional overlays, ring/tetrahedral interaction
   expansion, or query/dative semantics.

The no-regret order is to process `South Star 201` before more runtime
implementation. It names the only current atom-text-like unsupported witness
and prevents the next change from becoming a one-off isotope exception.

## Update After South Star 203

`[15te]1cccc1` and the analogous selenium/isotope-map forms are no longer
unsupported frontier witnesses. They are now bracket-only aromatic element text
cases: isotope prefixes and optional atom-map suffixes are admitted for
aromatic selenium and tellurium while charge, explicit hydrogens, radicals, and
chirality remain outside that policy boundary.

The small unsupported adversarial surface is therefore back to query/dative
bond semantics. `aromatic_ring_surface` remains a support-gate category for
future aromatic policy failures, but the current corpus no longer has a small
natural parseable witness for it. Before relying on that category in tests
again, add a new pinned witness or deliberately remove it from the live
frontier inventory.
