# South Star Unsupported Frontier Inventory

Branch: `south-star`

Date: 2026-05-21

Task: `South Star 172: Inventory unsupported frontier witnesses`

## Purpose

The export-gate inventory shows no case-level blockers for the current checked
fixture surface. The remaining question is breadth: which declared fail-fast
categories are still ordinary unsupported frontier, and which ones should be
handled as separate semantic universes?

This note inventories the current declared support-gate blocker categories with
small witnesses. It is an implementation guide, not a support change.

## Witness Inventory

| Category | Small witness | Classification | Next clean family |
| --- | --- | --- | --- |
| `empty_molecule` | `Chem.Mol()` | Boundary edge case. It has no atom text, graph traversal, or semantic identity target. | Low-priority API semantics decision. |
| `query_atom` | SMARTS `[#6]-[#8]` | Query semantics, not a fixed molecule graph. | Keep separate from ordinary SMILES enumeration unless a query-SMILES product is defined. |
| `query_bond` | SMARTS `[#6]-[#8]` | Query semantics, not a fixed bond-order graph. | Same query-SMILES product boundary as `query_atom`. |
| `atom_stereo` | `CCF` with atom 1 manually tagged tetrahedral | Missing or invalid atom-stereo fact family outside the supported tetrahedral slices. | Audit atom-stereo tags that RDKit can carry but South Star should reject or model. |
| `unsupported_atom_text` | `[GeH3]C` | Mostly a text-policy breadth gap for element/modifier rendering beyond the first non-organic bracket-only slice. | Atom-text frontier slice. |
| `metal_atom` | `[NH3]->[Cu]` | Metal chemistry and coordination semantics. | Metal/dative semantic family; do not fold into ordinary atom text. |
| `unsupported_bond_type` | `C$C` | Bond-text and bond-semantics gap for non single/double/triple/aromatic types. | Bond-text frontier slice for ordinary bond orders; dative remains separate. |
| `dative_bond` | `[NH3]->[Cu]` | Coordination semantics plus known serializer quirks. | Metal/dative semantic family. |
| `disconnected_molecule` | `C$C.O` | Composition is blocked because at least one fragment is unsupported. | Usually disappears when fragment families become supported; not a standalone first target. |
| `ring_molecule` | `C1CC2CCCC2[C@H]1F` | Ring topology/stereo interaction outside current supported ring system. | Ring/tetrahedral and polycyclic interaction family. |
| `fused_or_polycyclic_ring` | `F/C=C\\C1CC2CCC1C2` | Polycyclic plus exocyclic stereo outside current supported polycyclic skeleton/stereo slices. | Polycyclic plus exocyclic-directional family. |
| `ring_tetrahedral_interaction` | `C1CC2CCCC2[C@H]1F` | Ring-local ligand-order dependence outside the current monocycle proof family. | Ring/tetrahedral interaction expansion. |
| `ring_stereo` | `C1CC2CCCC2[C@H]1F` with a ring bond manually tagged `STEREOZ` | Ring stereo in an unsupported ring/stereo context. | Ring-stereo expansion after ring-system family is named. |
| `aromatic_ring_surface` | `[se]1cccc1` | Aromatic coverage beyond markerless unmodified monocycles, supported branches, first modified-aromatic atom-text cases, and narrow unmodified fused ring systems. | Aromatic element-breadth or directional policy slice. |
| `aromatic_directional_surface` | `c1ccccc1` with an aromatic bond manually tagged directional | Aromatic directional overlay has no named semantic constraint family. | Aromatic directional policy, separate from markerless text. |
| `unstated_component_equation` | `FC=CCl` with the double bond manually tagged `STEREOZ` but no slash/backslash carrier basis | Internal molecule fact has stereo but no observable marker basis in the current equation language. | Decide whether this is invalid input for EnumS or a new non-token stereo fact family. |

## Current Split

The frontier is best treated as seven work streams:

1. Query semantics: `query_atom`, `query_bond`.
2. Text policy breadth: `unsupported_atom_text`, ordinary `unsupported_bond_type`.
3. Metal/coordination chemistry: `metal_atom`, `dative_bond`.
4. Fragment composition blocked by unsupported fragments: `disconnected_molecule`.
5. Ring-system breadth: `ring_molecule`, `fused_or_polycyclic_ring`,
   `ring_stereo`, `ring_tetrahedral_interaction`.
6. Aromatic breadth: `aromatic_ring_surface`, `aromatic_directional_surface`.
7. Under-specified or non-token stereo facts: `atom_stereo`,
   `unstated_component_equation`.

These streams should not be collapsed into one "support more molecules" row.
They have different semantic contracts and different risk profiles.

## Recommended Next Slice

The next lowest-regret implementation path is still aromatic expansion, but not
"all aromaticity." Split it first:

1. broader aromatic atom-symbol breadth such as `[se]`;
2. aromatic directional overlays;
3. broader aromatic systems outside the narrow fused-ring and modified-atom-text
   slices.

The already-admitted fused aromatic subfamily reuses the same spine: molecule
facts, traversal events, lowercase aromatic atom-text obligations, elided
aromatic bond text, parse-back evidence, and first-occurrence deduplication.
The next aromatic subfamily should keep that same proof shape instead of
adding renderer-local exceptions.

The next parallel planning path is atom/bond text breadth. It is likely easier
than aromatic directional overlays, but it is less strategically important for
the current South Star frontier because the semantic structure is mostly
renderer policy.
