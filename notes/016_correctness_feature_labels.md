# Correctness Feature Labels

The correctness coverage report currently answers two useful questions:

- how many checked-in cases exist by fixture family
- where the evidence came from: upstream RDKit, local probe, dataset-derived
  mining, random-writer observation, known RDKit gap, or RDKit quirk

That is provenance. It does not directly answer which writer-surface behavior
is covered.

Feature labels would add a small, tested vocabulary for that second question.
The goal is to make coverage holes visible without turning fixture files into a
metadata system.

## What a Feature Label Means

A feature label names a writer behavior that a case intentionally exercises.
Examples:

- `rooted`
- `all-roots`
- `disconnected`
- `nonfirst-fragment-root`
- `tetrahedral-stereo`
- `directional-bond-stereo`
- `kekule`
- `explicit-bonds`
- `explicit-hydrogens`
- `atom-map-ignored`
- `atom-map-preserved`
- `charges`
- `isotopes`
- `dative-bonds`
- `ring-closure-ordering`

The generated report could then include:

```text
Fixture cases by feature:
- rooted: 142
- disconnected: 18
- directional-bond-stereo: 27
- kekule: 6
```

Those counts are not a proof of completeness. They are a map of where evidence
exists and where it is thin.

## Why This Is Separate From Source Classification

Source classification answers "where did this case come from?"

Feature labels answer "what behavior does this case cover?"

Those should stay separate. A dataset-derived case may cover dative bonds, a
local probe may cover dative bonds, and an upstream RDKit regression may cover
dative bonds. Collapsing provenance and writer features into one field would
make the report less useful and harder to test.

## No-Regret Constraints

- Use a closed vocabulary in one helper module.
- Validate labels in fixture loaders.
- Require sorted, unique labels.
- Treat labels as optional at first.
- Do not infer labels from SMILES strings, case IDs, or source strings.
- Do not label every existing case just to fill a table.
- Do not make feature labels part of public runtime behavior.
- Do not use feature counts as release claims.

The first useful labels should be added only where the fixture intent is
already explicit: feature-matrix fixtures, selected serializer regressions, and
known stereo-gap fixtures.

## Minimal Implementation Plan

1. Define the vocabulary and validation helper.
2. Add loader tests for valid labels, unknown labels, duplicates, and unsorted
   labels.
3. Add optional `features` to a small number of existing cases whose intent is
   obvious.
4. Teach `report_correctness_coverage.py` to count feature labels.
5. Keep docs pointed at the generated report, not a manually maintained table.

Stop there. If labels start needing parameters, nesting, or auto-inference, the
design is getting too large for this slice.
