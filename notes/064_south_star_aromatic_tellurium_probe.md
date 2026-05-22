# South Star Aromatic Tellurium Boundary Probe

Branch: `south-star`

Date: 2026-05-22

Task: `South Star 188: Probe aromatic tellurium text boundary`

## Question

Can tellurophene `[te]1cccc1` be admitted as the next narrow South Star
aromatic element-text slice, using the same principled boundary as bracket-only
aromatic selenium?

## Probe

Reusable probe:

`tmp/exploration/aromatic_text/002_probe_aromatic_tellurium_boundary.py`

The probe records RDKit parse/write behavior, sanitized atom facts, current
South Star support-gate categories, grammar conformance, atom-text obligations,
and the exact missing South Star policy tokens.

## RDKit Source Evidence

Relevant writer source:

- `/home/coder/repos/rdkit/Code/GraphMol/SmilesParse/SmilesWrite.cpp:38`
  defines the organic subset as atomic numbers
  `B,C,N,O,F,P,S,Cl,Br,I`.
- `/home/coder/repos/rdkit/Code/GraphMol/SmilesParse/SmilesWrite.cpp:96`
  brackets atoms outside that organic subset.
- `/home/coder/repos/rdkit/Code/GraphMol/SmilesParse/SmilesWrite.cpp:186`
  lowercases aromatic symbols beyond the organic subset for atomic numbers
  `5,6,7,8,14,15,16,33,34,52`; `52` is tellurium.

This puts tellurium in the same writer-lowercasing family as selenium, but
outside the bare organic-subset token family. The intended text is therefore
bracketed lowercase `[te]`, not bare `te`.

## Probe Results

| Case | Source | RDKit writer | Current gate | Grammar | Missing policy |
| --- | --- | --- | --- | --- | --- |
| tellurophene lowercase | `[te]1cccc1` | `[te]1cccc1` | `aromatic_ring_surface`, `unsupported_atom_text` | `unsupported_token` | `Te` bracket-only atom, `te` bracket-only aromatic atom |
| tellurophene capital | `[Te]1cccc1` | `[te]1cccc1` | `aromatic_ring_surface`, `unsupported_atom_text` | `unsupported_token` | same |
| mapped tellurophene | `[te:7]1cccc1` | `[te:7]1cccc1` | `aromatic_ring_surface`, `unsupported_atom_text` | `unsupported_token` | same plus map modifier |
| tellurium explicit H | `[teH]1cccc1` | parse fails | n/a | n/a | n/a |
| charged tellurium | `[te+]1cccc1` | parse fails | n/a | n/a | n/a |
| acyclic tellurium baseline | `[TeH]` | `[TeH]` | `unsupported_atom_text` | `unsupported_token` | `Te` bracket-only atom |
| selenium comparator | `[se]1cccc1` | `[se]1cccc1` | ok | ok | none |

## Interpretation

`[te]1cccc1` is a clean implementation candidate, but it is slightly broader
than the selenium slice because `Te` is not yet admitted as a general
bracket-only atom symbol. A narrow tellurium implementation should therefore
make two explicit policy moves:

- admit `Te` to bracket-only non-organic atom text;
- admit `te` to bracket-only aromatic atom text.

Those two moves are still text-policy moves, not traversal or stereo-model
moves. Tellurium is not treated as a metal by the current gate, RDKit parses
the five-membered aromatic ring, and RDKit writes both `[te]1cccc1` and
`[Te]1cccc1` as `[te]1cccc1`.

Mapped tellurium should not be bundled into the first tellurium slice. It
composes with atom-map text in principle, but it should follow the mapped
selenium modifier probe so modifier composition remains a named expansion.
Explicit-H and charged aromatic tellurium are not support questions in this
probe because RDKit does not parse the tested inputs.

## Recommendation

Open an implementation row for narrow tellurium text:

1. add `Te` to bracket-only atom text and representative bracket tokens;
2. add `te` to bracket-only aromatic atom text;
3. pin `[te]1cccc1` as a unified-reference aromatic monocycle fixture;
4. generalize the selenium-specific helper names only as far as needed, e.g.
   bracket-only aromatic element text, without admitting mapped/charged/explicit
   hydrogen variants;
5. keep `[te:7]1cccc1` for the separate mapped-aromatic modifier task.

## Follow-Up After South Star 191

`South Star 191` implements this recommendation narrowly:

- `Te` is a bracket-only non-organic atom-text symbol;
- `te` is a bracket-only aromatic atom-text symbol;
- `[te]1cccc1` is pinned as `aromatic_tellurium_text_tellurophene`;
- selenium and tellurium share a generic bracket-only aromatic element-text
  unified-reference proof path;
- mapped tellurium remains outside this first slice.
