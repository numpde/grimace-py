# South Star Aromatic Boundary

Task: `South Star 74: Decide aromatic semantic boundary`

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

## Alternatives Considered

1. Exclude aromaticity and require non-aromatic molecule facts.

   This is the current stance. It is clean because the support gate can key off
   explicit RDKit aromatic flags, and it prevents accidental inheritance of
   RDKit's aromatic writer policy. A future non-aromatic kekule-like domain
   would need an explicit molecule-preparation contract that clears aromatic
   flags before enumeration.

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
