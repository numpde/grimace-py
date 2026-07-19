# Ordinary stereo implicit hydrogens

## Boundary decision

`MoleculeFacts` owns atom hydrogen counts and ligand occurrences. Validation now
requires the two representations to agree and requires neighbor occurrences to
name the exact opposite endpoint of an incident bond. The writer does not infer
hydrogens from valence.

For a specified acyclic directional site, an implicit H is a fixed ligand
reference, not an emitted carrier. The existing explicit-neighbor carrier model
therefore remains authoritative: one residual variable is created for each
emitted substituent bond, and `ligand_factor` records whether that explicit
occurrence or the implicit H is the side reference. No implicit-H residual
variable is serialized or replayed.

The admitted surface is deliberately bounded to one explicit neighbor and at
most one implicit H per side, with a specified target, a reference pair, and
bridge center/carrier bonds. Unspecified, ring, pseudo-ligand, carrierless, and
larger side domains retain the existing typed non-neighbor blocker.

## Field authorities

| Field | Authority |
| --- | --- |
| Atom implicit-H count | `AtomFacts` from source ingestion |
| H ligand occurrence | validated site occurrence at the owning endpoint |
| Directional target | sanitized RDKit E/Z relation |
| Reference pair | extracted directional site facts |
| Carrier variables | emitted explicit neighbor bonds only |
| Ligand factor | fact occurrence versus side reference |
| Residual replay | producer-free facts verifier plus `ResidualStore` |
| Support image | exact writer/continuation recurrence |
| RDKit parity | `rdkit_south_star_stereo_audit/2026.03.1.json` |

## Regression contract

The RDKit-ingested tetra polarities have six rooted supports each. The
directional opposite/together polarities have two each. All four pass support
artifact replay, local branch and terminal proof replay, continuation asset
verification, Rust decoding, and snapshot/resume. The synthetic
`directional_facts()` fixture remains a separate zero-H proof-kernel surface.
