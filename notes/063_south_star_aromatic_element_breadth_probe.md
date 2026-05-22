# South Star Aromatic Element Breadth Probe

Task: `South Star 185: Probe aromatic element vocabulary breadth`

## Question

Can aromatic atom symbols outside the current `b/c/n/o/p/s` vocabulary enter
South Star as a principled aromatic text slice, starting with selenophene
`[se]1cccc1`?

## Probe

Reusable probe:

`tmp/exploration/aromatic_text/001_probe_aromatic_element_breadth.py`

The probe records RDKit parse/write behavior, sanitized atom facts, current
South Star support-gate categories, declared grammar conformance, and current
atom-text obligation behavior for representative bracket aromatic main-group
atoms.

## RDKit Source Evidence

Relevant RDKit writer source:

- `/home/coder/repos/rdkit/Code/GraphMol/SmilesParse/SmilesWrite.cpp:38`
  defines the organic subset as atomic numbers
  `B,C,N,O,F,P,S,Cl,Br,I`.
- `/home/coder/repos/rdkit/Code/GraphMol/SmilesParse/SmilesWrite.cpp:96`
  says atoms outside the organic subset need brackets.
- `/home/coder/repos/rdkit/Code/GraphMol/SmilesParse/SmilesWrite.cpp:186`
  lowercases aromatic symbols, beyond the organic subset, for atomic numbers
  `5,6,7,8,14,15,16,33,34,52`.
- `/home/coder/repos/rdkit/Code/GraphMol/SmilesParse/catch_tests.cpp:139`
  has SMARTS coverage for `[si]`, `[as]`, `[se]`, and `[te]`, but SMILES
  sanitization/writing does not keep all of those as aromatic SMILES cases.

## Probe Results

| Case | Source | RDKit writer | Sanitized atom fact | Current South Star |
| --- | --- | --- | --- | --- |
| selenophene | `[se]1cccc1` | `[se]1cccc1` | `Se`, aromatic | `aromatic_ring_surface`, grammar rejects `[se]` |
| tellurophene | `[te]1cccc1` | `[te]1cccc1` | `Te`, aromatic | `unsupported_atom_text`, `aromatic_ring_surface`, grammar rejects `[te]` |
| arsabenzene input | `[as]1ccccc1` | `[As]1=CC=CC=C1` | non-aromatic `As` | `unsupported_atom_text` |
| silabenzene input | `[si]1ccccc1` | `[Si]1=CC=CC=C1` | non-aromatic `Si` | already in non-aromatic ring scope |
| sulfur baseline | `s1cccc1` / `[s]1cccc1` | `s1cccc1` | `S`, aromatic | supported |
| mapped selenium | `[se:7]1cccc1` | `[se:7]1cccc1` | `Se`, aromatic, map `7` | same missing aromatic token family |
| selenium explicit H / charge | `[seH]1cccc1`, `[se+]1cccc1` | parse fails | n/a | no support question |

## Interpretation

`[se]1cccc1` is the clean first target. It is not an ordinary atom-text
breadth case: the relevant fact is an aromatic `Se` atom, and RDKit writes it
as bracketed lowercase `[se]` because selenium is outside the organic subset
but inside RDKit's aromatic lowercasing set.

The current South Star model already supports `Se` as a non-metal bracket-only
atom symbol. The missing piece is specifically aromatic selenium text:

- add `se` / `[se]` to the aromatic atom-text vocabulary;
- keep the bracketed spelling, not bare `se`, because selenium is outside
  RDKit/OpenSMILES organic-subset bare atom text;
- keep this under aromatic policy, not ordinary non-aromatic bracket text;
- pin at least `[se]1cccc1` before widening to mapped `[se:7]`.

Tellurium is similar at the writer level but also requires admitting `Te` as a
new non-metal bracket-only atom symbol. It should be a second slice. Arsenic and
silicon are not the same aromatic-SMILES slice in sanitized RDKit SMILES here:
RDKit writes the tested six-membered examples as Kekule bracket atoms, not as
aromatic `[as]` or `[si]` strings.

## Recommendation

Open a narrow implementation task for selenium aromatic atom text:

1. Add an `aromatic_element_breadth` or narrower `aromatic_selenium_text`
   feature area.
2. Add `se` to the declared aromatic atom-text token vocabulary and bracket
   grammar.
3. Keep `Se` as bracket-only atom text; render aromatic selenium as `[se]`.
4. Pin `[se]1cccc1` as a unified-reference aromatic monocycle fixture.
5. Keep `[te]`, `[as]`, `[si]`, mapped selenium, charged selenium, and explicit-H
   selenium outside that first implementation slice.

This is an implementation row, not a Decision row. The semantic boundary is
clear enough for a narrow first fixture, and the wider main-group aromatic
vocabulary can be handled as later, source-backed slices.

## Follow-Up After South Star 187

`South Star 187` implements the recommended selenium slice:

- `[se]` is admitted as bracket-only aromatic atom text, not as bare `se`;
- `[se]1cccc1` is pinned as `aromatic_selenium_text_selenophene` under the
  shared bracket-only aromatic element-text authority;
- mapped selenium such as `[se:7]1cccc1` remains outside the first slice;
- tellurium, arsenic, and silicon remain separate from this implementation
  boundary for the reasons identified above.

## Follow-Up After South Star 191

`South Star 191` promotes the narrow tellurium text slice:

- `Te` is admitted as bracket-only atom text;
- `[te]` is admitted as bracket-only aromatic atom text, not as bare `te`;
- `[te]1cccc1` is pinned as `aromatic_tellurium_text_tellurophene` under the
  same shared bracket-only aromatic element-text authority as selenium;
- mapped tellurium remains outside the first tellurium slice.
