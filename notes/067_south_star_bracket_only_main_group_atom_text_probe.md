# South Star Bracket-Only Main-Group Atom-Text Probe

Branch: `south-star`

Date: 2026-05-22

Task: `South Star 194: Probe bracket-only main-group atom-text policy`

## Question

Can the existing South Star bracket-only atom-text model admit more
non-organic, non-metal main-group symbols as a policy family, instead of adding
one-off element cases?

## Probe

Reusable probe:

`tmp/exploration/atom_text/001_probe_bracket_only_main_group_policy.py`

The probe records RDKit parse/write behavior, sanitized atom facts, current
South Star metal-gate status, current atom-text support-gate status, grammar
conformance, and atom-text obligations for representative simple hydrides,
alkyl hydrides, Kekule rings, and metal-boundary comparators.

## Source Evidence

Relevant RDKit writer source:

- `/home/coder/repos/rdkit/Code/GraphMol/SmilesParse/SmilesWrite.cpp:38`
  defines RDKit's organic subset as atomic numbers
  `B,C,N,O,F,P,S,Cl,Br,I`.
- `/home/coder/repos/rdkit/Code/GraphMol/SmilesParse/SmilesWrite.cpp:96`
  brackets atoms outside that organic subset.

Relevant South Star source:

- `python/grimace/_south_star/atom_text.py` currently admits bracket-only
  non-aromatic symbols `Si`, `Se`, and `Te`.
- `python/grimace/_south_star/support_gates.py` separately blocks metals by
  `METAL_ATOMIC_NUMBERS`.

## Results

| Case | RDKit writer | Current South Star |
| --- | --- | --- |
| `[SiH3]C` | `[SiH3]C` | supported bracket-only atom text |
| `[SeH]` | `[SeH]` | supported bracket-only atom text |
| `[TeH]` | `[TeH]` | supported bracket-only atom text |
| `[AsH3]` | `[AsH3]` | blocked only by `unsupported_atom_text` |
| `[AsH2]C` | `[AsH2]C` | blocked only by `unsupported_atom_text` |
| `[GeH4]` | `[GeH4]` | blocked only by `unsupported_atom_text` |
| `[GeH3]C` | `[GeH3][CH3]` | blocked only by `unsupported_atom_text` |
| `[SbH3]` | `[SbH3]` | blocked only by `unsupported_atom_text` |
| `[As]1=CC=CC=C1` | `[As]1=CC=CC=C1` | blocked only by `unsupported_atom_text` |
| `[Ge]1=CC=CC=C1` | `[Ge]1=[CH]C=CC=[CH]1` | blocked only by `unsupported_atom_text` |
| `[SnH4]` | `[SnH4]` | blocked by `metal_atom` and `unsupported_atom_text` |
| `[Na+]` | `[Na+]` | blocked by `metal_atom` and `unsupported_atom_text` |
| `[Mg+2]` | `[Mg+2]` | blocked by `metal_atom` and `unsupported_atom_text` |

## Interpretation

The clean family boundary is not "add every RDKit bracket atom." The clean
boundary is:

- the atom is outside RDKit's organic subset, so bracket text is required;
- the atom is outside the current South Star metal gate;
- RDKit parses and writes stable ordinary covalent examples for the parsed
  molecule facts being tested;
- the existing bracket-atom renderer can express the required field modifiers
  from facts: isotope, explicit hydrogens, charge, radical semantics, chirality,
  and atom map;
- no traversal, bond-text, aromatic-policy, dative, query, or coordination
  semantics are being smuggled in through the atom-text admission.

Under that boundary, `As`, `Ge`, and `Sb` look like bracket-only atom-text
policy candidates. `Sn`, `Na`, and `Mg` are not atom-text-only targets in the
current model because they cross the explicit metal gate.

This is still a South Star semantic boundary, not RDKit-writer parity. RDKit's
writer behavior is useful source evidence for the bracket requirement and
field spelling, but the admission criterion should remain parsed-fact support:
can the unified reference model render semantically valid SMILES for this
ordinary covalent molecule without adding local writer quirks?

## Recommendation

Do not add isolated `As` or `Ge` special cases. Promote the next runtime slice
as a bracket-only main-group atom-text policy extension:

1. Add a small named symbol set for non-metal, non-organic bracket-only
   main-group symbols.
2. Start with `As`, `Ge`, and `Sb` if fixture generation confirms the expected
   support sets are ordinary products of the existing reference spine.
3. Keep `Sn`, alkali/alkaline-earth atoms, and other metal-gated atoms outside
   this slice.
4. Pin at least one hydride/alkyl case and one Kekule-ring case before widening
   further, because atom text alone must not hide traversal or bond-text gaps.
5. Keep this as data-backed policy breadth; no traversal or chemistry repair
   should be added for the first slice.

## Follow-Up Backlog Shape

The next implementation row should be narrow and reviewable:

- implement bracket-only main-group atom text for a first candidate set such as
  `As`, `Ge`, and `Sb`;
- pin RDKit-versioned fixtures for representative simple cases;
- verify the package-readiness suite and JSON fixture consistency;
- leave metal-boundary comparators as explicit unsupported evidence.
