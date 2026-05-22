# South Star Bracket Aromatic Element Non-Map Modifiers

Branch: `south-star`

Date: 2026-05-22

Task: `South Star 201: Probe non-map modifiers on bracket-only aromatic elements`

## Question

Which non-map modifiers on bracket-only aromatic elements are valid parsed
molecule facts, and which should stay outside the current South Star atom-text
policy?

## Probe

Reusable probe:

`tmp/exploration/aromatic_text/006_probe_bracket_only_aromatic_element_non_map_modifiers.py`

The probe compares RDKit parse/write behavior, current South Star support
gates, grammar conformance, atom-text obligations, and derived support sets for
isotope-modified selenium/tellurium aromatic monocycles.

## Results

| Modifier family | Examples | RDKit behavior | South Star grammar/atom text | Current gate |
| --- | --- | --- | --- | --- |
| isotope | `[15te]1cccc1`, `[15se]1cccc1` | parses and writes same modifier | grammar ok; renders `[15te]` / `[15se]` with `isotope_prefix` | `aromatic_ring_surface` |
| isotope + atom map | `[15te:7]1cccc1`, `[15se:7]1cccc1` | parses and writes same modifiers | grammar ok; renders isotope and map suffix obligations | `aromatic_ring_surface` |
| explicit H | `[teH]1cccc1`, `[seH]1cccc1` | parse fails | no South Star support question for these inputs | n/a |
| charge | `[te+]1cccc1`, `[se+]1cccc1` | parse fails | no South Star support question for these inputs | n/a |
| chiral marker input | `[te@]1cccc1`, `[se@]1cccc1` | parses but normalizes to `[te]` / `[se]`; chiral tag is unspecified | already the unmodified supported molecule after parsing | ok |

Derived isotope supports by substituting the bracket-only aromatic element atom
text in existing pinned support sets:

- `[te]` -> `[15te]`: 20 derived outputs, 0 parse failures, 0 grammar failures;
- `[se]` -> `[15se]`: 20 derived outputs, 0 parse failures, 0 grammar failures;
- `[te:7]` -> `[15te:7]`: 20 derived outputs, 0 parse failures, 0 grammar failures;
- `[se:7]` -> `[15se:7]`: 20 derived outputs, 0 parse failures, 0 grammar failures.

## Interpretation

The live atom-text frontier is isotope composition, not general "modified
bracket aromatic element" composition.

Isotope text is a parsed molecule fact and the existing atom-text renderer can
already express it. The current blocker is the support-gate predicate for
bracket-only aromatic element fields, which allows unmodified atoms and map-only
atoms but rejects isotope-bearing atoms. The shared monocycle traversal and
proof shape do not need to change.

Charge and explicit-H examples tested here fail RDKit parsing, so admitting
them would not be support broadening for this parsed-molecule layer. Chiral
marker inputs are normalized away by RDKit into unmodified aromatic element
facts; they are not an independent South Star support slice unless we decide to
track caller spelling rather than parsed molecule semantics.

## Recommendation

Open a narrow implementation row for isotope composition on bracket-only
aromatic element text:

1. Extend the support-gate predicate to allow isotope with optional atom map on
   bracket-only aromatic element text.
2. Pin at least `[15te]1cccc1` and `[15te:7]1cccc1`; include selenium isotope
   cases if the fixture size remains reviewable.
3. Reuse the existing bracket-only aromatic element proof path.
4. Keep charge and explicit-H out because RDKit does not parse the tested
   inputs.
5. Do not add a chiral modifier slice; RDKit drops the tested marker and the
   parsed molecule is already the unmodified supported case.
