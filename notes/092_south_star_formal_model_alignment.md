# South Star Formal Model Alignment

Branch: `south-star`

Date: 2026-05-22

Context: follow-up to `notes/091_smiles_exact_support.tex`

## Purpose

Record the implementation posture after reviewing the formal exact-support
formulation.

The key conclusion from `091_smiles_exact_support.tex` is that, for a fixed
finite molecular graph and a bounded explicitly declared SMILES dialect, exact
semantic support can be modeled as:

```text
finite attributed graph-traversal grammar
+ finite-domain constraints over syntax slots
```

South Star should now align the implementation to that model directly, rather
than continue accumulating feature-local witnesses or exporting a provisional
API before the model is cleanly reflected in code.

## Decisions

### Annotation Policy

South Star should not hard-code one annotation policy as the whole concept.

The policy layer should be pluggable. The current
`maximal_eligible_carrier` policy remains the first concrete policy and the
default seed policy, but it is not the definition of South Star itself.

Implementation consequence:

- keep policy names explicit in result diagnostics and tests;
- keep policy-dependent support expectations tagged with the policy;
- avoid baking maximal-carrier assumptions into molecule facts, traversal
  events, or generic constraint records.

### No Accidental Stereo

No-accidental-stereo constraints are first-class South Star constraints.

Exact semantic support requires both:

- preserve every specified stereo assignment; and
- prevent emitted syntax from introducing stereo at unspecified potential
  stereo sites.

This should become a named constraint family, not an afterthought hidden in
parse-back tests.

Implementation consequence:

- identify potential stereo sites, not only specified stereo sites;
- derive typed obligations for unspecified potential sites;
- constrain marker/token assignments so unspecified sites parse as unspecified;
- keep parse-back tests as evidence, not as the defining filter.

### Refactor Now

The implementation should now refactor toward the formal model, even if that
causes churn.

The current seed passes its readiness gates, but readiness under the current
shape is not enough. The model should be made explicit:

- traversal skeletons;
- syntax slots;
- typed constraints;
- solver assignments;
- renderer inputs;
- support as the rendered image of satisfying assignments.

The goal is not to add more cases first. The goal is to make the one-truth
spine more exact and less feature-local.

### Public Export

Public export is off the table for now.

`MolToSmilesEnumS(mol) -> tuple[str, ...]` remains a useful proposed surface,
and the private dry-run wrapper remains useful contract evidence. But the
surface should not be exported while the implementation is being realigned to
the formal model.

The previous export-decision audit should therefore be read as: the old gates
no longer had an obvious internal fixture blocker, not as a recommendation to
export before the formal-model refactor.

## Duplicate Witnesses

The formal model distinguishes two questions:

1. What is the support set?
2. How many internal witnesses generate each support string?

The mathematically natural support definition is:

```text
support = image(render : satisfying_assignments -> strings)
```

Under this definition, duplicate witnesses are allowed internally. For example,
two different roots, branch orders, symmetric atom choices, or automorphism-
related traversal skeletons may render the same SMILES string. Deterministic
deduplication of rendered strings is therefore semantically valid: it computes
the image of `render`.

This is different from post-hoc parse filtering. Deduplication does not decide
whether a string is valid; validity is already decided by the constraints.
Deduplication only collapses multiple valid witnesses for the same string.

The alternative is to add a canonical-witness constraint:

```text
among all satisfying assignments that render the same string, keep one
canonical assignment
```

That can reduce duplicate generation and may matter for performance, but it is
not required for correctness. It is also nontrivial because "same rendered
string" can be caused by graph automorphisms, branch/traversal redundancies, or
policy-level normalization. Premature canonical-witness logic risks mixing
semantic membership with optimization and symmetry breaking.

Recommended posture:

- define support as the rendered image of satisfying assignments;
- keep deterministic output deduplication as the initial implementation;
- expose duplicate counts and deduplication ratios in diagnostics;
- add canonical-witness constraints later only if needed for performance or
  reviewability, and only as a separate symmetry-breaking layer.

## Immediate Refactor Direction

The next implementation work should align the current code to the formal
objects in `091_smiles_exact_support.tex`.

Suggested sequence:

1. Make traversal skeletons explicit records rather than implicit recursive
   control flow inside `enum_s.py`.
2. Make syntax slots explicit and typed: atom text, bond text, ring label,
   directional marker, tetrahedral token, fragment separator.
3. Move directional preservation constraints and no-accidental-stereo
   constraints into named constraint families.
4. Keep annotation policy as an injected layer over eligible slot/assignment
   facts.
5. Make rendering consume only skeleton events plus solved assignments.
6. Treat fixture and parse-back tests as witnesses against the model, not as
   generation authority.

## Non-Goals For The Next Slice

- Do not export `MolToSmilesEnumS`.
- Do not broaden the molecule domain just to add more examples.
- Do not chase RDKit writer parity inside the South Star semantic model.
- Do not implement canonical-witness constraints until the ordinary
  render-image model is clean and measured.

