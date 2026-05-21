# South Star Atom/Bond Text Frontier

Branch: `south-star`

Date: 2026-05-21

Task: `South Star 174: Inventory atom and bond text frontier`

## Current Text Surface

The current atom-text policy is broader than the original seed:

- organic subset atoms: `B`, `C`, `N`, `O`, `P`, `S`, `F`, `Cl`, `Br`, `I`;
- aromatic lowercase atoms: `b`, `c`, `n`, `o`, `p`, `s`;
- bracket hydrogens;
- isotope prefixes;
- formal-charge suffixes;
- radical valence semantics;
- atom-map suffixes;
- explicit hydrogen counts;
- tetrahedral carbon atom text.
- bracket-only non-organic atom symbols in the first non-metal slice: `Si`
  and `Se`.

The current bond-text policy covers:

- elided single bonds;
- explicit double bonds;
- explicit triple bonds;
- elided aromatic bonds.

## Probe Summary

Supported examples:

- `[13CH3:7]C`
- `[15NH3+]C`
- `[NH4+]`
- `[PH4+]`
- `[SH3+]`
- `[Cl-]`
- `[Br-]`
- `[I-]`
- `[OH-]`
- `[SiH3]C`
- `[SeH]`
- `C=C`
- `C#N`

Unsupported examples:

- `[Na+]`: `unsupported_atom_text`, `metal_atom`
- `[Mg+2]`: `unsupported_atom_text`, `metal_atom`
- `C$C`: `unsupported_bond_type`
- `[NH3]->[B]`: `unsupported_bond_type`, `dative_bond`
- `[NH3]->[Cu]`: `unsupported_atom_text`, `unsupported_bond_type`,
  `metal_atom`, `dative_bond`
- `c1cc[nH]c1`: `aromatic_ring_surface`

## Classification

### Ordinary Atom-Text Breadth

`[SiH3]C` and `[SeH]` were the clean ordinary atom-text frontier before
`South Star 177`. They did not require new graph traversal or stereo
semantics. The required implementation shape was:

- expanding supported bracket atom symbols;
- expanding the declared grammar-conformance atom-symbol basis;
- adding fixtures and renderer-obligation tests;
- deciding whether the symbols are bracket-only or can ever be bare tokens.

This is the safest text-policy implementation slice.

`South Star 177` implements this slice narrowly:

- `Si` and `Se` are supported only as bracket atom text;
- `[SiH3]C` and `[SeH]` are pinned as unified-reference fixtures;
- metals, dative bonds, query atoms/bonds, quadruple bonds, and modified
  aromatic atoms remain separate gated families.

### Metals

`[Na+]` and `[Mg+2]` are not just missing symbols. They trip `metal_atom`.
Even if bracket text rendering is easy, semantic expectations around metals,
coordination, and salts should be a separate policy family.

### Dative Bonds

`[NH3]->[B]` and `[NH3]->[Cu]` are not ordinary unsupported bond text. They are
coordination/dative semantics and should stay outside a generic bond-order
renderer expansion.

### Quadruple Bonds

`C$C` is the clean ordinary bond-text frontier. It is a real RDKit bond type
with a distinct text token. It may be implementable as a small bond-text slice,
but it is less important than bracket atom-symbol breadth because it is rare
and may need a clearer OpenSMILES compatibility statement.

### Aromatic Modified Atom Text

`c1cc[nH]c1` is not an ordinary atom-text slice under the current gate. It is
part of aromatic expansion because the atom is in an aromatic molecule-fact
contract. It should not be bundled with non-aromatic `[SiH3]C`.

## Recommended Next Text Slice

The original recommended text-breadth slice was non-metal bracket atom symbols:

1. add a named `non_organic_bracket_atom_text` feature area;
2. start with `[SiH3]C` and `[SeH]`;
3. require bracket spelling for those symbols;
4. update grammar conformance and atom-text obligation tests;
5. add unified-reference fixtures and package-readiness rows;
6. keep metals, dative bonds, query atoms/bonds, and aromatic modified atoms
   gated.

This was a separate implementation row from aromatic branches. It is a small
renderer-policy expansion, not a chemistry/stereo-model expansion.
