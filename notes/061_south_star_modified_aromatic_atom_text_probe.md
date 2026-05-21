# South Star Modified Aromatic Atom Text Probe

Branch: `south-star`

Date: 2026-05-21

Task: `South Star 180: Probe modified aromatic atom text`

## Question

Can modified aromatic atom text, starting with `[nH]`, enter as an atom-text
and grammar-policy slice, or does it require a new traversal model?

## Probe

The probe script is:

`tmp/exploration/aromatic_atom_text/001_probe_modified_aromatic_atom_text.py`

It inspects sanitized RDKit molecule facts and, for monocycle cases, runs the
existing ring traversal spine with a temporary hypothetical bracket-aromatic
atom renderer. That monkeypatch is deliberately local to the exploration
script; it is not runtime behavior.

Witnesses:

| Witness | Source | Current gate | Hypothetical support | Parse/graph/stereo rejections | Grammar rejections |
| --- | --- | --- | ---: | ---: | ---: |
| pyrrole explicit H | `c1cc[nH]c1` | `aromatic_ring_surface` | 20 | 0 | 20 |
| isotope pyrrole | `c1cc[15nH]c1` | `aromatic_ring_surface` | 20 | 0 | 20 |
| mapped pyrrole | `[nH:7]1cccc1` | `aromatic_ring_surface` | 20 | 0 | 20 |
| mapped pyridine | `c1cc[n:7]cc1` | `aromatic_ring_surface` | 30 | 0 | 30 |
| pyridinium H | `c1cc[nH+]cc1` | `aromatic_ring_surface` | 30 | 0 | 30 |
| pyridine N-oxide | `c1cc[n+]([O-])cc1` | `aromatic_ring_surface` | 60 | 0 | 60 |
| selenophene | `[se]1cccc1` | `aromatic_ring_surface` | not probed | not probed | not probed |

For the first six witnesses, the existing traversal/rendering structure is
mechanically sufficient once a hypothetical bracket-aromatic token is provided.
The South Star semantic oracle rejects the outputs only because the declared
grammar currently excludes bracket aromatic atom tokens such as `[nH]`,
`[15nH]`, `[n:7]`, and `[nH+]`.

## Current Runtime Boundary

The runtime correctly fails today:

- `_aromatic_atom_text_supported` requires unmodified aromatic atoms;
- `_aromatic_atom_text_obligation` raises when `_requires_bracket_atom_text`
  is true;
- `is_south_star_bracket_atom_text_token` recognizes uppercase/bracket atom
  text but not lowercase aromatic bracket symbols;
- the grammar-conformance helper similarly rejects bracket aromatic atom tokens;
- the support gate reports these cases as `aromatic_ring_surface`, not as a
  distinct atom-text frontier.

That is a clean fail-fast boundary, but it is now too coarse for the next
implementation slice.

## Interpretation

Modified aromatic atom text is not a fused-ring or traversal problem. It is an
atom-text and grammar-policy problem:

- token language: bracket aromatic symbols need their own accepted token class;
- field rendering: isotope, explicit hydrogen count, charge, and atom map need
  to be rendered around lowercase aromatic symbols;
- obligations: bracket aromatic atoms need typed obligations distinct from
  unmodified `aromatic_subset` atoms and non-aromatic `bracket_atom` atoms;
- support gates: modified aromatic atom text should have a named predicate or
  feature area instead of being buried under broad `aromatic_ring_surface`;
- fixtures: expected supports should be pinned under the unified-reference
  spine, with parse-back and grammar evidence.

The first implementation does not need to change the polycyclic/fused-ring
closure-traversal spine. It should reuse the existing ring-system traversal
paths and widen only atom text, grammar, proof, fixtures, and support gates.

## Recommended Split

Implement, but split it from broader aromatic element breadth.

1. Add bracket-aromatic atom text policy for already-supported aromatic symbols
   (`b`, `c`, `n`, `o`, `p`, `s`) with existing renderer-capable modifiers:
   isotope, explicit hydrogen count, charge, and atom map.
2. Extend the South Star grammar-conformance helper and bracket-token predicate
   so these tokens are grammar-valid when emitted by the policy.
3. Pin first fixtures for `[nH]`, mapped aromatic nitrogen, isotopic `[nH]`, and
   charged aromatic nitrogen, using existing monocycle/branch traversal.
4. Keep aromatic selenium (`[se]`) as a separate future element-breadth slice.
   It changes the aromatic atom symbol vocabulary, not just modified atom text.

This should spawn implementation Backlog rows. The current probe row should not
directly widen support.

## Follow-Up After South Star 183

`South Star 183` implements the first token/grammar boundary from this split:
bracket aromatic tokens over the current `b/c/n/o/p/s` vocabulary are now valid
South Star grammar tokens and direct atom-text obligations. This does not admit
modified aromatic molecules into `MolToSmilesEnumS` support. The support gate
still reports `[nH]`-style molecules as `aromatic_ring_surface` until
fixture-backed support is added by a separate row.
