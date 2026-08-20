# Marker Basis Policy Boundary

Branch: `stereo-constraint-model`

## Purpose

This note fixes the boundary exposed by the smallest stereo-gap witness after
commit `3aeca8b`. The key separation is not "RDKit vs non-RDKit" in general.
It is:

- parser-backed semantic validity: does RDKit parse a marker spelling back to
  the intended graph and double-bond stereo assignment?
- RDKit writer policy: among semantically valid spellings, which spelling does
  RDKit's writer actually emit for the supported writer regime?
- Grimace current support: which spellings the current runtime accepts today,
  including known red-gap behavior.

The public runtime target remains exact RDKit writer support. That target must
be reached through this two-layer boundary: first parser-backed semantic
validity, then named RDKit writer policy. A runtime shortcut that directly
admits one RDKit string without proving the parser-equivalence and
writer-policy layers would be another local special case.

## Smallest Witness

Fixture:
`tests/fixtures/rdkit_known_stereo_gaps/2026.03.1.json`

Case:
`github3967_part2_directional_ring_closure_canonical`

Direction-erased skeleton:

```text
C1=CCC=C2C3=CCC=CC=CC3C2C=C1
```

The fixture now pins four different sets for this same skeleton:

- source spelling: a 4-marker spelling accepted by RDKit's parser;
- RDKit emitted spelling: a 3-marker spelling emitted by RDKit's writer;
- parse-equivalent minimal bases: 18 minimal marker slot assignments that parse
  to the same target double-bond stereo assignment;
- Grimace current same-skeleton support: 2 accepted marker assignments.

The important observation is that the source spelling and RDKit emitted
spelling are both parse-equivalent, but Grimace's current same-skeleton
accepted assignments are not in the parse-equivalent minimal basis set. That
makes this witness more informative than a simple "missing RDKit string" case:
it exposes both a missing RDKit writer-policy spelling and current support
admitted on the same skeleton that does not satisfy the pinned parser-backed
basis check.

## Boundary Terms

`parse_equivalent_minimal_marker_slots`
: A fixture-pinned set of minimal slash/backslash marker assignments for one
direction-erased skeleton. Each assignment is sufficient for RDKit's parser to
recover the intended double-bond stereo signature, and no single marker can be
removed while preserving that signature. This is a parser-backed semantic
validity diagnostic, not an RDKit writer-support claim or an independent
chemistry oracle.

`boundary_layer_classes`
: A fixture-pinned set of labels that says which boundary surfaces are present
for a red case: RDKit expected output is parse-equivalent but missing from
support, current same-skeleton Grimace support contains parse-equivalent
outputs, current same-skeleton Grimace support contains parse-mismatched
outputs, or there is no current same-skeleton support.

`expected`
: The exact RDKit writer output for the pinned case. It must be
parser-equivalent, but parser equivalence alone does not prove RDKit writer
parity.

`expected_current_same_skeleton_marker_slots`
: The marker assignments Grimace currently accepts for the same skeleton while
the case remains red. These are current-behavior diagnostics, not expected
final behavior.

`gap_class`
: The broad current failure class. For marker-basis witnesses, this may need to
be refined after the semantic and writer-policy layers are both visible.

## Runtime Acceptance Rule

A future runtime-supported output should satisfy all relevant layers:

1. It belongs to the parser-equivalent space for the intended molecule and
   skeleton. At minimum, RDKit-parsed graph and double-bond stereo assignment
   must match the intended prepared molecule.
2. It satisfies named RDKit writer policy for traversal, rooting, fragment
   order, visible marker placement, marker movement, and no-marker
   obligations.
3. It is generated online through the support boundary. No post-hoc string
   repair or completed-string projection is allowed.

For the public RDKit-parity API, parser-equivalent but non-RDKit spellings are
not sufficient. For future semantic checks, they may be valuable diagnostics,
but those tests must be explicitly named as parser-backed semantic-equivalence
tests unless they use an independent oracle.

## Fact Ownership

Semantic facts:

- stereo double-bond endpoints and candidate carriers;
- coupled endpoint/component membership;
- token phase assignments that preserve intended stereo relations;
- parser-backed marker-basis validity for a direction-erased skeleton.

RDKit writer-policy facts:

- traversal order observations;
- selected visible marker slots;
- no-marker observations at candidate edges;
- ring-closure, branch, and later-slot marker placement;
- deferred marker obligations and legal discharge slots;
- any RDKit-specific suppression of otherwise parser-equivalent marker bases.

Current-support diagnostics:

- Grimace accepted same-skeleton marker assignments for red cases;
- row survivor counts and current rejected/accepted marker bases;
- shadow/procedural compatibility checks during migration.

## Consequences

- Do not treat `parse_equivalent_minimal_marker_slots` as the final support
  set. It is larger than RDKit writer support.
- Do not treat RDKit's emitted spelling as a law of chemistry. It is one
  writer-policy-selected spelling inside the parser-equivalent space.
- Do not treat current Grimace same-skeleton outputs as acceptable just because
  they share the direction-erased skeleton. They must parse to the intended
  stereo assignment.
- Before promoting marker placement into runtime support, the code should name
  whether each filter is semantic or RDKit writer-policy.

## Next Slice

The next implementation slice should classify the remaining known red stereo
gaps with this vocabulary:

- parse-equivalent RDKit spelling missing from support;
- current Grimace same-skeleton accepted spelling is parse-mismatched;
- RDKit writer-policy placement/movement missing;
- unsupported representation or fixture surface.

That classification should stay fixture-backed and RDKit-version-pinned.
