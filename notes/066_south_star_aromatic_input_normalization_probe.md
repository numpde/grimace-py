# South Star Aromatic-Input Normalization Probe

Branch: `south-star`

Date: 2026-05-22

Task: `South Star 190: Probe As/Si aromatic-input normalization`

## Question

Should `[as]1ccccc1` and `[si]1ccccc1` be treated as aromatic element-breadth
cases, known quirks, or ordinary parsed-object support boundaries?

## Probe

Reusable probe:

`tmp/exploration/aromatic_text/004_probe_aromatic_input_normalization.py`

The probe compares RDKit writer output, sanitized atom/bond facts, South Star
support gates, grammar conformance, and current `MolToSmilesEnumS` prototype
support.

## Results

| Case | RDKit writer | Sanitized facts | Current South Star |
| --- | --- | --- | --- |
| `[si]1ccccc1` | `[Si]1=CC=CC=C1` | all atoms and bonds non-aromatic; alternating double/single ring | supported, 60 outputs |
| `[Si]1=CC=CC=C1` | `[Si]1=CC=CC=C1` | same | supported, 60 outputs |
| `[as]1ccccc1` | `[As]1=CC=CC=C1` | all atoms and bonds non-aromatic; alternating double/single ring | `unsupported_atom_text` |
| `[As]1=CC=CC=C1` | `[As]1=CC=CC=C1` | same | `unsupported_atom_text` |

The silicon normalized input and explicit Kekule silicon input produce
identical South Star support.

## Interpretation

These are not aromatic South Star support targets after RDKit parsing and
sanitization. RDKit accepts the lowercase aromatic-looking input syntax, but the
sanitized molecule is a non-aromatic Kekule ring. South Star should reason from
the parsed molecule facts at this layer, not from the caller's original input
spelling.

For silicon, the semantic support already exists. The missing artifact is a
pinned fixture that documents the normalization boundary:

- source may be `[si]1ccccc1`;
- fixture evidence should state that RDKit sanitizes it to a non-aromatic
  silicon Kekule ring;
- the unified-reference support is ordinary nonstereo monocycle support with
  bracket-only silicon atom text, not aromatic silicon support.

For arsenic, the same normalization happens, but atom-text policy does not yet
admit `As`. That should be treated as a future bracket-only non-organic
atom-text expansion, not as aromatic element breadth and not as an RDKit quirk
requiring special writer-parity handling.

## Recommendation

Open one implementation row for the silicon normalized-input fixture. Open a
separate probe or implementation row for bracket-only arsenic atom text only if
the project wants to broaden non-organic atom text beyond `Si`, `Se`, and the
planned `Te` slice.

## Follow-Up After South Star 193

`South Star 193` pins `[si]1ccccc1` as
`non_organic_bracket_atom_text_silicon_kekule_ring`. This does not add
aromatic silicon support: the fixture records RDKit's sanitized non-aromatic
Kekule molecule facts and uses ordinary nonstereo monocycle traversal with
bracket-only silicon atom text.
