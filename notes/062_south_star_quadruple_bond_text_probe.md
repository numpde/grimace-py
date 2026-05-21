# South Star Quadruple Bond Text Probe

Branch: `south-star`

Date: 2026-05-21

Task: `South Star 181: Probe quadruple bond text policy`

## Question

Can ordinary quadruple bond text, starting with `C$C`, enter South Star as a
small bond-text policy slice, or does it require a chemistry/stereo model change?

## Probe

The probe script is:

`tmp/exploration/bond_text/001_probe_quadruple_bond_text.py`

It checks RDKit parse/write behavior, South Star support-gate categories, and
declared South Star grammar membership for representative quadruple-bond inputs.

## External And Source References

OpenSMILES defines `$` as the quadruple-bond symbol; the canonical public spec
states that double, triple, and quadruple bonds are represented by `=`, `#`, and
`$` respectively:

https://opensmiles.org/opensmiles.html

RDKit mirrors this directly in the checked-in 2026.03.1 source copy:

`tests/fixtures/rdkit_upstream_serializer_sources/2026.03.1/source/Code/GraphMol/SmilesParse/SmilesWrite.cpp:322`

There, `Bond::QUADRUPLE` writes `$`. This is a normal writer bond-text case, not
an RDKit extension like dative arrows.

## Probe Results

| Case | Source | RDKit parses? | RDKit writer outputs | South Star gate | Grammar |
| --- | --- | --- | --- | --- | --- |
| carbon-carbon quadruple | `C$C` | yes | `C$C` | `unsupported_bond_type` | rejects `$` |
| explicit bracket carbon | `[C]$[C]` | yes | `C$C` | `unsupported_bond_type` | rejects `$` |
| molybdenum quadruple | `[Mo]$[Mo]` | yes | `[Mo]$[Mo]` | `metal_atom`, `unsupported_atom_text`, `unsupported_bond_type` | rejects `$` |
| OpenSMILES-style bracket example | `[Ga+]$[As-]` | yes | `[Ga+]$[As-]`, `[As-]$[Ga+]` | `metal_atom`, `unsupported_atom_text`, `unsupported_bond_type` | rejects `$` |
| valence-invalid branch | `CC$C` | no | n/a | n/a | n/a |

RDKit sanitization is a real boundary. The small useful ordinary witness is
`C$C`; many organic-looking variants are rejected by RDKit valence before South
Star has a support question.

## Interpretation

For the narrow `C$C` family, quadruple bond support is a bond-text policy gap:

- molecule facts already expose the bond as `QUADRUPLE`;
- there is no stereo carrier, marker slot, or constraint-family interaction;
- graph traversal for a two-atom molecule already exists;
- the renderer needs a `"$"` bond-text obligation;
- the grammar needs `"$"` in the declared bond-token set;
- support gates need `Chem.BondType.QUADRUPLE` in the ordinary supported bond
  set once the fixture is pinned.

The metal and bracket-heavy examples should not be bundled into this slice. They
cross atom-text and metal policy boundaries. `[Ga+]$[As-]` is useful evidence
that `$` is an OpenSMILES bond token, but it is not a good first South Star
support witness because it changes several policies at once.

## Recommendation

Implement a narrow ordinary quadruple-bond slice first:

1. Add `$` to the South Star bond-text policy as `explicit_quadruple_bond`.
2. Add `$` to the declared grammar bond tokens.
3. Admit `Chem.BondType.QUADRUPLE` in the support gate only for otherwise
   supported molecules.
4. Pin `C$C` under expanded support with unified-reference two-atom bond-text
   authority, or add a new authority if the existing two-atom helper is too
   semantically named for markerless text.
5. Keep `[Mo]$[Mo]`, `[Ga+]$[As-]`, and other metal/bracket-heavy cases gated
   until atom-text and metal policy breadth is deliberately widened.

This is an implementable renderer-policy slice, not a Decision row. The only
minor design choice is whether to reuse the existing two-atom markerless
atom-text authority for a `$` bond, or rename/generalize it before pinning the
fixture. Prefer the small generalization if the current name reads as too narrow.
