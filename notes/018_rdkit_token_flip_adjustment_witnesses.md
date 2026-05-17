# RDKit Token-Flip Adjustment Witnesses

Branch: `stereo-constraint-model`

## Purpose

This note records the decision behind the `rdkit_component_token_flip_adjustment`
slice. The adjustment is RDKit writer policy, not generic stereo semantics. It
should be decomposed into named observations before any replacement of the
current helper.

## Source References

Local RDKit source copy:

- `tests/fixtures/rdkit_upstream_serializer_sources/2026.03.1/source/Code/GraphMol/SmilesParse/SmilesWrite.cpp:250`
  enters `GetBondSmiles` with an `atomToLeftIdx` traversal-side argument.
- `SmilesWrite.cpp:275` reads the stored `BondDir`.
- `SmilesWrite.cpp:281` to `SmilesWrite.cpp:291` maps
  `ENDDOWNRIGHT`/`ENDUPRIGHT` to visible `\`/`/` for single bonds when
  isomeric SMILES are requested.
- `SmilesWrite.cpp:326` to `SmilesWrite.cpp:336` applies the same visible
  direction-token mapping for aromatic bonds.

Those lines explain the serializer surface that Grimace must match: the visible
token is emitted from a traversal-side view of a stored bond direction. They do
not by themselves define a clean support-state rule. Grimace therefore treats
the current adjustment as an RDKit writer-policy observation pending full
decomposition.

## Pinned Witnesses

The version-keyed fixture
`tests/fixtures/stereo_constraint_model/2026.03.1.json` now pins adjustment
reason counts beside existing token-inference branch counts.

Current witnesses:

| Case | Role | Pinned counts |
| --- | --- | --- |
| `isolated_single_candidate_alkene` | root/begin-side orientation adjustment for `isolated_all_single_candidate` | `root_begin_side_orientation=4`, `value_true=4` |
| `coupled_single_candidate_diene` | root/begin-side orientation adjustment for `coupled_one_candidate_begin_side` | `root_begin_side_orientation=6`, `value_true=6` |
| `coupled_two_candidate_branched_diene` | root/begin-side orientation adjustment for `coupled_two_candidate_begin_side` | `root_begin_side_orientation=14`, `value_true=14` |
| `coupled_adjacent_two_candidate_token_adjustment` | adjacent two-candidate/first-emitted adjustment | `adjacent_two_candidate_first_emitted=1`, `root_begin_side_orientation=13`, `value_true=14` |

The exploration source for these cases is
`tmp/exploration/stereo_assignment/030_mine_token_inference_branch_witnesses.py`.

## Interpretation

The current adjustment has two named reasons:

- `root_begin_side_orientation`: the component begin atom is the traversal root,
  and the selected begin-side orientation is `after_atom`.
- `adjacent_two_candidate_first_emitted`: the selected begin neighbor is the
  traversal root, and an adjacent two-candidate side first emitted a different
  neighbor than the begin atom.

The diagnostic invariant is:

```text
rdkit_token_flip_adjustment.value
  == root_begin_side_orientation XOR adjacent_two_candidate_first_emitted
```

That invariant is intentionally descriptive. It pins the current RDKit-parity
surface without claiming this is principled chemistry semantics.

## Next Implementation Rule

Do not promote `rdkit_component_token_flip_adjustment` as one opaque primitive.
The replacement should consume named RDKit writer-policy observations at the
support-state boundary, with the current helper retained only as a temporary
shadow/equivalence oracle.
