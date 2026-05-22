# South Star Mapped Aromatic Tellurium Probe

Branch: `south-star`

Date: 2026-05-22

Task: `South Star 195: Probe mapped aromatic tellurium modifier`

## Question

Does atom-map text compose cleanly with bracket-only aromatic tellurium text, or
should `[te:7]1cccc1` remain outside the current South Star boundary?

## Probe

Reusable probe:

`tmp/exploration/aromatic_text/005_probe_mapped_aromatic_tellurium_modifier.py`

The probe compares RDKit writer output, current South Star support gates,
grammar conformance, atom-text obligations, and a derived support set obtained
by substituting `[te]` with `[te:7]` in the pinned tellurophene support.

## Results

| Case | RDKit writer | Current gate | Grammar | Atom-text obligation |
| --- | --- | --- | --- | --- |
| `[te:7]1cccc1` | `[te:7]1cccc1` | `aromatic_ring_surface` | ok | `[te:7]`, `bracket_aromatic_atom`, `atom_map_suffix` |
| `[Te:7]1cccc1` | `[te:7]1cccc1` | `aromatic_ring_surface` | ok | same |
| `[te]1cccc1` | `[te]1cccc1` | ok | ok | `[te]`, `bracket_aromatic_atom` |
| `[se:7]1cccc1` | `[se:7]1cccc1` | ok | ok | `[se:7]`, `bracket_aromatic_atom`, `atom_map_suffix` |

Derived support by replacing `[te]` with `[te:7]` in the pinned
`aromatic_tellurium_text_tellurophene` support:

- base outputs: 20;
- derived mapped outputs: 20;
- RDKit parse failures: 0;
- South Star grammar failures: 0;
- tellurium atom-map failures: 0.

## Interpretation

Mapped aromatic tellurium is not a traversal, ring, or grammar problem. It is
the same modifier-composition shape as mapped aromatic selenium:

- the shared monocycle traversal spine is unchanged;
- atom-text rendering already produces `[te:7]` with an `atom_map_suffix`
  obligation;
- all derived outputs parse back to one aromatic tellurium atom with map `7`;
- the only current blocker is the support-gate predicate that permits atom maps
  for bracket-only aromatic selenium but not tellurium.

The principled boundary is therefore "atom-map text composes with supported
bracket-only aromatic element text" rather than "selenium maps are special."
This still does not admit isotope, charge, radical, explicit-H, or chiral
modifiers on bracket-only aromatic elements; those remain separate policy
questions.

## Recommendation

Open a narrow implementation row for mapped aromatic tellurium:

1. Generalize the support-gate predicate from mapped selenium to mapped
   bracket-only aromatic element text.
2. Pin `[te:7]1cccc1` under the same shared bracket-only aromatic element-text
   authority or an explicitly named modifier-composition authority if the
   manifest needs finer vocabulary.
3. Reuse the existing bracket-only aromatic element proof path.
4. Keep every non-map modifier outside until separately probed.
