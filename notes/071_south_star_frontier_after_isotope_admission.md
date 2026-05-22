# South Star Frontier After Isotope Admission

Branch: `south-star`

Date: 2026-05-22

Task: `South Star 204: Refresh frontier after isotope admission`

## Purpose

Refresh the South Star frontier after admitting isotope prefixes, optionally
with atom-map suffixes, for bracket-only aromatic selenium and tellurium atom
text.

This replaces the stale assumption that `[15te]1cccc1` is a live
`aromatic_ring_surface` witness. It is now supported and pinned as expanded
South Star fixture coverage.

## Probe

Reusable inventory script:

`tmp/exploration/frontier/001_recompute_post_atom_text_expansion.py`

The script triages every adversarial corpus candidate with the current support
gate and counts expanded fixture feature areas.

## Current Adversarial Triage

The adversarial corpus now has:

- candidates: 25
- supported by gate: 23
- unsupported by gate: 2

Unsupported candidates:

| Candidate | Source | Unsupported categories | Boundary targets |
| --- | --- | --- | --- |
| `unsupported_feature_triggers:query_unspecified_bond` | `C~C` | `query_bond`, `unsupported_bond_type` | `query_bond`, `unsupported_bond_type` |
| `unsupported_feature_triggers:dative_bond` | `N->[O]` | `dative_bond`, `unsupported_bond_type` | `dative_bond` |

Unsupported category counts:

- `dative_bond`: 1
- `query_bond`: 1
- `unsupported_bond_type`: 2

Supported boundary-target counts:

- `all_fragment_orders`: 2
- `aromatic_selenium_text`: 4
- `aromatic_tellurium_text`: 4
- `charged_atom_text`: 1
- `disconnected_stereo_fragments`: 2
- `explicit_bracket_hydrogen`: 1
- `first_domain_directional_stereo`: 5
- `maximal_eligible_carrier`: 5
- `quadruple_bond_text`: 1
- `radical_atom_text`: 1
- `ring_stereo_monocycle`: 2
- `tetrahedral_atom_stereo`: 2

## Fixture Surface

Expanded fixture feature counts:

- `aromatic_selenium_text`: 4
- `aromatic_tellurium_text`: 4
- `aromatic_text_branch`: 3
- `aromatic_text_monocycle`: 3
- `atom_map_text`: 1
- `branched_saturated_monocycle`: 1
- `charged_atom_text`: 3
- `combined_atom_text_modifiers`: 2
- `directional_tetrahedral_composition`: 1
- `disconnected_markerless_fragments`: 1
- `disconnected_stereo_fragments`: 2
- `double_bond_text`: 1
- `exocyclic_directional_monocycle`: 1
- `explicit_bracket_hydrogen`: 1
- `fused_aromatic_ring_system`: 3
- `independent_directional_stereo_components`: 1
- `isotope_atom_text`: 1
- `markerless_acyclic_tree`: 4
- `modified_aromatic_atom_text`: 6
- `non_organic_bracket_atom_text`: 7
- `nonstereo_polycyclic_skeleton`: 1
- `polycyclic_ring_stereo`: 1
- `quadruple_bond_text`: 1
- `radical_atom_text`: 3
- `ring_stereo_monocycle`: 1
- `ring_tetrahedral_exocyclic_directional`: 1
- `ring_tetrahedral_monocycle`: 2
- `simple_saturated_monocycle`: 1
- `tetrahedral_atom_stereo`: 2
- `triple_bond_text`: 1
- `unsaturated_nonstereo_monocycle`: 3

## Frontier Interpretation

The current small adversarial frontier is no longer atom text. It is ordinary
bond-surface semantics at the boundary of the current fixed-molecule South Star
model:

1. Query or unspecified bond semantics: `C~C`.
2. Dative or coordination semantics: `N->[O]`.

`aromatic_ring_surface` remains a support-gate category because future aromatic
policy failures may still need it. It is not currently represented by a small
natural parseable unsupported witness in the adversarial corpus.

## Recommended Next Step

Do not pick the next runtime expansion directly from this note. First process
`South Star 205`: compare the remaining small frontier against larger known
South Star gaps such as aromatic directional overlays and ring/stereo
interactions, then choose the next semantic family by mathematical leverage and
SSoT testability.
