# South Star Aromatic Boundary

Tasks: `South Star 74: Decide aromatic semantic boundary`,
`South Star 101: Decide aromatic semantic boundary`,
`South Star 106: Choose aromatic molecule-fact contract`,
`South Star 121: Define aromatic policy family contract`,
`South Star 151d: Model aromatic ring rendering policy`

## Current Decision

Do not support aromatic RDKit molecules in the current South Star surface.
Fail fast with `aromatic_ring_surface`, and report
`aromatic_directional_surface` separately when an aromatic bond also carries a
directional marker.

This is a boundary decision, not a claim that aromatic SMILES are outside the
long-term South Star target. The current implementation depends on RDKit for
parsing and molecule facts. With normal RDKit sanitization, even
`C1=CC=CC=C1` is represented as an aromatic molecule, so a text rule like
"accept kekule-looking input" is not precise enough.

The `South Star 101` review keeps this decision unchanged. The important
boundary is molecule facts, not source spelling: lowercase aromatic input and
kekule-looking aromatic input both become aromatic RDKit molecule facts under
the current parser boundary. South Star should not infer a semantic support
contract from the user's original text once RDKit has canonicalized those facts.

`South Star 106` makes the molecule-fact split executable. Under normal
sanitized RDKit parsing, both `c1ccccc1` and `C1=CC=CC=C1` produce aromatic
atom facts, aromatic bond facts, and the same fail-fast aromatic categories.
A deliberately kekulized molecule with aromatic flags cleared is a different
input-preparation contract: its atom facts are non-aromatic and its bond facts
are explicit single/double bonds. That contract is allowed to flow through the
current non-aromatic ring machinery, but it is not the same as supporting
aromatic RDKit molecule facts or aromatic atom text.

`South Star 121` keeps that split as the active policy-family contract. The
current package boundary is `non_aromatic_molecule_facts`: South Star consumes
RDKit molecule facts, and aromatic atom or bond facts fail fast. The nearest
future aromatic-adjacent support path is therefore not "accept kekule-looking
source text"; it is a named preparation boundary such as
`non_aromatic_kekule_facts`, where aromatic flags have already been cleared and
the bond-order assignment is part of the caller-visible contract. A real
`aromatic_text_policy` remains a separate family because it would define
lowercase atom text, aromatic bond text/elision, and aromatic/Kekule semantic
equivalence directly.

`South Star 151d` makes the policy family explicit in code without changing the
active gate. The family now has one active contract,
`non_aromatic_molecule_facts`, and two named candidate contracts:
`non_aromatic_kekule_facts` for caller-prepared non-aromatic Kekule molecule
facts, and `aromatic_text_policy` for sanitized aromatic molecule facts plus
lowercase/aromatic rendering semantics. The manifest exposes only active
contracts; candidates are planning boundaries, not runtime support.

## Alternatives Considered

1. Exclude aromaticity and require non-aromatic molecule facts.

   This is the current stance. It is clean because the support gate can key off
   explicit RDKit aromatic flags, and it prevents accidental inheritance of
   RDKit's aromatic writer policy. A future non-aromatic kekule-like domain
   would need an explicit molecule-preparation contract that clears aromatic
   flags before enumeration. `South Star 106` records that this is a distinct
   molecule-fact contract, not a source-spelling exception.

2. Support kekulized semantic output only.

   This may be a good future slice, but it needs a precise input contract. If
   input comes from ordinary sanitized RDKit parsing, uppercase alternating
   benzene text is still aromatic. Supporting this path therefore requires a
   named molecule-fact boundary: what counts as a non-aromatic aromatic-system
   representation, how bond orders are chosen, and whether equivalent Kekule
   forms are one semantic class or multiple output strings.

3. Define an aromatic text policy.

   This is the broadest option. It would need explicit rules for lower-case
   aromatic atom text, aromatic bond elision, ring traversal, aromatic
   directional surfaces, and semantic equivalence between aromatic and
   kekulized parsebacks. That is a real grammar policy, not a small renderer
   tweak.

4. Treat aromaticity as parse-back equivalence only.

   This is not a support policy by itself. It may be a useful diagnostic once
   aromatic support exists, but it cannot define enumeration because it does not
   say which strings belong to the support, whether lowercase aromatic text or
   explicit Kekule text is preferred, or how maximal annotation interacts with
   aromatic bonds.

## Support Entry Conditions

A future aromatic slice should not remove the fail-fast gate until it names all
of the following explicitly:

- molecule-fact contract: sanitized aromatic RDKit facts, deliberately
  kekulized non-aromatic facts, or another documented preparation boundary;
- atom-text policy: lowercase aromatic atoms, bracket aromatic atoms, or
  explicit non-aromatic atoms;
- bond-text policy: aromatic bond elision, explicit aromatic bonds, or Kekule
  single/double bonds;
- semantic equivalence relation: whether aromatic and Kekule parsebacks count
  as the same semantic target for South Star correctness;
- maximal annotation policy: whether aromatic directional surfaces are excluded,
  annotated, or represented through a separate constraint family.

Until those are named, `aromatic_ring_surface` and
`aromatic_directional_surface` are the correct behavior.

For the current contract, the named answers are:

- molecule-fact contract: `non_aromatic_molecule_facts`;
- atom-text policy: existing non-aromatic organic/bracket atom text only;
- bond-text policy: existing explicit single/double non-aromatic bond text only;
- semantic equivalence relation: ordinary parse-back graph/stereo identity for
  non-aromatic facts only;
- maximal annotation policy: no aromatic directional carrier family yet, so
  aromatic directional markers remain unsupported overlays.

This means `non_aromatic_kekule_facts` can be explored as a future input
preparation policy without weakening the fail-fast boundary for ordinary
sanitized aromatic RDKit molecules.

## Boundary Tests

The support gate should keep these distinctions visible:

- `c1ccccc1` is unsupported as `aromatic_ring_surface`.
- `C1=CC=CC=C1` is also unsupported after normal RDKit sanitization because the
  molecule facts are aromatic even though the input text is kekule-looking.
- aromatic bonds with directional markers are reported as
  `aromatic_directional_surface` in addition to the aromatic ring surface.

## Future Work

The next aromatic-support task should start by choosing a molecule-fact
contract, not by editing the renderer:

- `non_aromatic_kekule_facts`: enumerate explicit single/double bond graphs
  whose aromatic flags are false.
- `aromatic_text_policy`: enumerate lower-case aromatic SMILES under a named
  semantic equivalence relation.
- `comparison_only_rdkit_aromatic_writer`: keep RDKit writer behavior as a
  comparison target, not as South Star semantic authority.
