# South Star Frontier After Ring/Tetra Admission

Branch: `south-star`

Date: 2026-05-22

Task: `South Star 211: Refresh frontier after ring/tetra admission`

## Purpose

Refresh the South Star frontier after admitting the first polycyclic
ring/tetrahedral fixtures under
`unified_reference_polycyclic_ring_tetrahedral_obligations`.

This note supersedes the ring/tetrahedral planning state in notes 071-074 for
frontier triage. Those notes remain the historical rationale for choosing and
shaping the ring/tetrahedral family.

## Probe

Reused inventory script:

`tmp/exploration/frontier/001_recompute_post_atom_text_expansion.py`

The script was run after commits `4916442` and `310ae17`.

## Current Adversarial Triage

The small adversarial corpus still has:

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

Supported boundary-target counts are unchanged from the previous adversarial
checkpoint:

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

## Expanded Fixture Surface

Expanded fixture feature counts now include two polycyclic ring/tetrahedral
witnesses:

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
- `polycyclic_ring_tetrahedral`: 2
- `quadruple_bond_text`: 1
- `radical_atom_text`: 3
- `ring_stereo_monocycle`: 1
- `ring_tetrahedral_exocyclic_directional`: 1
- `ring_tetrahedral_monocycle`: 2
- `simple_saturated_monocycle`: 1
- `tetrahedral_atom_stereo`: 2
- `triple_bond_text`: 1
- `unsaturated_nonstereo_monocycle`: 3

## What Changed

The ring/tetrahedral frontier chosen in note 072 has moved from planning to a
first domain-level implementation:

- `F[C@H]1CC2CCC1C2` is pinned as the cleaner bridged witness with 784 outputs.
- `F[C@H]1CC2CC1C2` is pinned as the compact minimality-stress witness with
  312 outputs.
- Both fixtures are generated from the shared polycyclic traversal,
  ring-closure event, tetrahedral obligation, semantic parse-back, and
  first-occurrence deduplication proof path.
- Runtime output equality is now a cross-check against that proof authority,
  not the fixture source.

The adversarial frontier did not change because the adversarial corpus did not
contain those polycyclic ring/tetrahedral witnesses. The meaningful change is
in the expanded fixture surface and support-gate domain boundary.

## Current Interpretation

The immediate ordinary fixed-molecule gap represented by notes 072-074 is now
covered by representative unified-reference fixtures. The remaining tiny
adversarial gaps are no longer ordinary molecule stereo-enumeration gaps:

- `C~C` is query or unspecified-bond semantics.
- `N->[O]` is dative or coordination-bond semantics.

Both remain valid South Star boundary questions, but neither should be
promoted casually as "just another bond token." Each may require a separate
semantic product model.

## Recommended Next Steps

1. Audit the polycyclic ring/tetrahedral admission as a domain-level proof
   boundary, not a case-level special case. The support gate now admits a class,
   while fixtures provide representative witnesses.
2. Choose the next semantic family deliberately. Compare query bonds, dative
   bonds, aromatic directional overlays, larger mixed ring/stereo composition,
   and multi-center tetrahedral scaling before implementing another expansion.

