# South Star Mapped Aromatic Selenium Probe

Branch: `south-star`

Date: 2026-05-22

Task: `South Star 189: Probe mapped aromatic selenium modifier`

## Question

Does atom-map text compose cleanly with the bracket-only aromatic selenium
slice, or does `[se:7]1cccc1` require a broader model change?

## Probe

Reusable probe:

`tmp/exploration/aromatic_text/003_probe_mapped_aromatic_selenium_modifier.py`

The probe compares RDKit writer output, current South Star support gates,
grammar conformance, atom-text obligations, and a derived support set obtained
by substituting `[se]` with `[se:7]` in the pinned selenophene support.

## Results

| Case | RDKit writer | Current gate | Grammar | Atom-text obligation |
| --- | --- | --- | --- | --- |
| `[se:7]1cccc1` | `[se:7]1cccc1` | `aromatic_ring_surface` | ok | `[se:7]`, `bracket_aromatic_atom`, `atom_map_suffix` |
| `[Se:7]1cccc1` | `[se:7]1cccc1` | `aromatic_ring_surface` | ok | same |
| `[se]1cccc1` | `[se]1cccc1` | ok | ok | `[se]`, `bracket_aromatic_atom` |

Derived support by replacing `[se]` with `[se:7]` in the pinned
`aromatic_selenium_text_selenophene` support:

- base outputs: 20;
- derived mapped outputs: 20;
- RDKit parse failures: 0;
- South Star grammar failures: 0;
- selenium atom-map failures: 0.

## Interpretation

Mapped aromatic selenium is not a new traversal problem. The existing shared
monocycle traversal and branch-event spine is enough, and the atom-text layer
already knows how to render `[se:7]`.

The current unsupported status is deliberate policy residue from the first
selenium slice: bracket-only aromatic selenium was admitted only for unmodified
atom fields. The implementation delta should therefore be narrow:

- relax the bracket-only aromatic selenium support gate for `atom_map_number`;
- keep isotope, charge, radical, explicit hydrogen, and chirality outside until
  separately probed;
- pin `[se:7]1cccc1` under a modifier-composition feature area or a generalized
  aromatic selenium text feature area;
- reuse the shared monocycle unified-reference proof path.

## Recommendation

Open an implementation row for mapped aromatic selenium after this probe. It
should be a modifier-composition slice, not a general aromatic element-breadth
slice and not a broad "all modified selenium" slice.

## Follow-Up After South Star 192

`South Star 192` implements the narrow modifier-composition slice:

- `[se:7]1cccc1` is pinned as
  `aromatic_selenium_text_mapped_selenophene`;
- the support gate allows atom-map text on bracket-only aromatic selenium;
- mapped tellurium remains outside the slice;
- isotope, charge, radical, explicit-H, and chiral selenium remain outside
  until separately probed.
