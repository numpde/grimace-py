# Correctness Coverage Expansion Plan

This note plans how to expand Grimace correctness coverage without turning the
suite into an unstructured pile of examples. The goal is broader evidence for
the current RDKit writer surface, with clear evidence type, stable fixture
provenance, and tests that fail for meaningful reasons.

## Baseline Snapshot

Snapshot from 2026-05-25. The main pinned RDKit parity lane exercises these
version-keyed fixture families for RDKit `2026.03.1`:

- `rdkit_exact_small_support`: 67 exact-support cases.
- `rdkit_rooted_random`: 1 rooted-random upstream RDKit case.
- `rdkit_serializer_regressions`: 130 exact support and inventory regression
  cases.
- `rdkit_writer_membership`: 51 deterministic RDKit writer-membership cases.

Adjacent RDKit-grounded evidence exists:

- `rdkit_known_stereo_gaps`: 16 known gap cases.
- `rdkit_known_quirks`: 1 isolated RDKit behavior observation.
- `rdkit_disconnected_sampling` and `rdkit_stereo_regressions`: compatibility
  and reusable regression fixtures outside the main pinned parity runner.
- `rdkit_upstream_serializer_coverage`: reviewed upstream serializer ledger,
  currently with no `unreviewed` or `needs-fixture` entries.

There are 19 checked-in cases explicitly derived from dataset mining: 4 older
regression cases, 9 deterministic writer-membership cases, and 6 exact-support
random-slice cases. This is still a small corpus, but it is now broad enough to
exercise multiple writer surfaces.

## Revised Principles

1. Expand by evidence type and classification, not by molecule count.
   A new case is useful only if it strengthens positive evidence, records a
   specific parity gap, isolates a known RDKit quirk, or closes an upstream
   coverage-triage question.

2. Keep evidence type explicit.
   Exact support equality is strongest. Deterministic RDKit writer membership
   is weaker but essential for large supports. Bounded random sampling is
   evidence, not proof. Known gaps are a separate classification, not a weak
   support claim.

3. Use mining only as candidate generation.
   The miner should find candidates. Promotion into checked-in fixtures requires
   human classification, a stable source string, and an executable test
   assertion.

4. Prefer small exact cases when possible.
   If a mined or upstream case has small saturated support, promote it to exact
   support/inventory equality rather than membership-only coverage.

5. Do not duplicate source of truth.
   Fixture families, typed loaders, and the upstream coverage ledger are the
   source of truth. Any coverage table or summary should be generated from
   those inputs, not maintained by hand.

6. Stratify by writer feature.
   Coverage should expose writer-surface holes: roots, fragments, stereo,
   dative bonds, atom maps, isotopes, charges, aromaticity, ring closures,
   explicit bonds, explicit hydrogens, and kekule output.

7. Keep known gaps visible.
   A known gap is not documentation-only. It should remain version-pinned and
   executable as parity debt, but it should not be mixed into passing parity
   claims.

## Non-Goals

- Do not make CI discover new molecules or new RDKit behavior dynamically.
  CI should run checked-in fixtures and generated summaries from checked-in
  inputs.
- Do not replace RDKit writer-parity claims with semantic equivalence claims.
  Semantic evidence is useful, but it belongs in a separately named layer.
- Do not maintain coverage counts by hand in user-facing docs.
- Do not add fixture metadata just because it is convenient for a report. Add
  metadata only when it becomes a tested contract.
- Do not put broad, expensive corpus scans in the normal parity lane.

## Positive Evidence Types

Use the strongest practical positive evidence for each passing case. This is
not a total order, but exact support evidence is generally stronger than
membership-only evidence:

1. `exact-support-and-inventory`: support and token inventory both match.
2. `exact-support`: Grimace support equals the pinned expected support.
3. `decoder-equivalence`: decoder and determinized decoder reach the same
   expected support when the support is small enough to enumerate through the
   decoder.
4. `writer-membership`: a deterministic RDKit writer output is in Grimace
   support.
5. `sampled-rdkit-membership`: bounded RDKit random outputs are in Grimace
   support.
6. `sampled-support-agreement`: repeated RDKit sampling appears to saturate the
   same support as Grimace for a small case. This is evidence only unless the
   support is otherwise exhaustively established.

Do not promote a case with weaker evidence if stronger bounded evidence is
cheap. For example, a tiny mined molecule should become an exact-support
fixture, not only a writer-membership fixture.

## Case Classification

Classification is separate from positive evidence. Use the vocabulary from
`docs/correctness-contracts.md` where it applies:

- `exact-rdkit-match`: the string is in RDKit's pinned writer support.
- `semantic-equivalent`: the string parses to the intended molecule/stereo
  assignment, but is not known to be in RDKit's pinned writer support.
- `rdkit-only`: RDKit emits the string and Grimace does not. While it fails,
  it belongs in a known-gap fixture, not in a passing parity family.
- `semantic-error`: the string does not parse to the intended molecule/stereo
  assignment.
- `rdkit-quirk`: pinned RDKit behavior is unusual enough to isolate from normal
  parity claims.
- `known-rdkit-gap`: Grimace does not yet mirror a pinned RDKit writer case.
- `out-of-scope`: an upstream RDKit claim does not map to Grimace's current
  public writer surface. This is a serializer-ledger status, not a molecular
  fixture claim.

The normal parity lane should contain passing `exact-rdkit-match` claims.
Known RDKit gaps should be executable in a separate diagnostic path or encoded
as explicit expected debt, so they do not silently become passing coverage.

Do not add explicit `claim_kind` or `classification` fields to every fixture
unless the current fixture-family boundary stops being enough. Derive meaning
from the owning fixture family first; add metadata only when it removes
ambiguity that tests cannot otherwise resolve.

## Expansion Paths

### 1. Dataset-Mined Writer Cases

Use `scripts/mine_rdkit_regressions.py` to scan larger and more diverse slices.
Promote only high-signal cases:

- `rdkit_only` results become known-gap fixtures while they fail.
- Small saturated results become exact-support fixtures.
- `grimace_only` results are useful diagnostics but are not RDKit parity
  failures by themselves.
- `uncertain` results stay in mining logs until they can be classified.

Promotion rule: one promoted case should explain a distinct failure mechanism
or writer feature. Avoid adding many near-duplicates from the same molecule
family.

### 2. Feature-Matrix Fixtures

Add hand-designed small cases that isolate serializer features:

- fixed root and all-roots behavior
- disconnected rooted fragments
- atom maps with `ignoreAtomMapNumbers`
- `allBondsExplicit`
- `allHsExplicit`
- `kekuleSmiles`
- aromatic charge normalization
- isotopes
- dative bond direction
- ring closure token ordering
- tetrahedral chirality
- directional double-bond stereo
- coupled stereo systems

Small feature cases should prefer exact support and token-inventory equality.
The current feature matrix is covered by exact small-support fixtures plus the
serializer-regression exact fixtures for surfaces that were already pinned from
upstream RDKit cases. Keep this as a map, not per-fixture metadata, until tests
need machine-readable feature labels:

- Connected roots: `cco_root1_nonstereo`,
  `feature_matrix_01_cco_all_roots_nonstereo`.
- Disconnected roots: `co_cco_root0_disconnected`,
  `co_cco_root3_disconnected`, and adjacent rooted-fragment cases.
- Atom maps: `ethyl_atom_maps_ignored`, `ethyl_atom_maps_kept`, and rooted
  variants.
- Writer flags: pyridine/benzene explicit-bond and kekule cases,
  `methane_all_hs_explicit`, plus dataset-derived flag cases.
- Isotopes and charges: `feature_matrix_02_isotopic_ammonium_all_roots`,
  `random_dataset_exact_06_disodium_carbonate_all_roots`.
- Dative bonds: `ammine_platinum_*_dative` and `ammine_copper_*_dative`.
- Ring closures: `feature_matrix_03_cyclopropyl_chloride_all_roots_nonstereo`
  and aromatic rooted ring cases.
- Stereo: `bromochlorofluoromethane_*`, `fluorochloroethene_*`,
  `fluorochloroimine_*`, serializer-regression stereo fixtures, and current
  known-gap fixtures for coupled directional stereo.

### 3. Upstream Serializer Ledger Tightening

The ledger already maps upstream RDKit source blocks to fixture links. Improve
the quality of those links before adding more upstream prose:

- Every in-scope `covered` entry should link to executable fixture cases.
- Each linked fixture should make its evidence type obvious from family and
  test path.
- Serializer-ledger `known-gap` entries should point to executable gap cases,
  not only notes.
- `out-of-scope` entries should explain the public-surface mismatch concisely.
The correctness coverage report now validates executable ledger links: every
`covered` and `known-gap` entry must link to existing fixture files and case
IDs, while `out-of-scope` entries must not link executable fixture cases. This
keeps link integrity checked without duplicating the fixture loaders' typed
molecular validation. A manual audit tightened remaining vague `out-of-scope`
notes so they name the public-surface mismatch.

### 4. Known Gap Suite

Keep known stereo gaps as first-class work items:

- Add missing minimized reproductions when a gap case is large.
- Group gaps by implementation family, not by upstream file alone.
- When a gap is fixed, promote its fixture to the appropriate passing parity
  family and update the ledger status.
- Keep diagnostic gap execution separate from the normal passing parity lane
  unless the test asserts explicit expected debt rather than pretending the gap
  is passing coverage.

### 5. Public Runtime Cross-Checks

For exact support fixtures, check more than `MolToSmilesEnum`:

- decoder reachability
- determinized decoder reachability
- exact token inventory
- token inventory superset containment
- prepared-molecule round trip when relevant
- deviation accepts every expected member and rejects a targeted non-member

Do this for small cases only. Large writer-membership cases should not grow
expensive decoder walks unless the case specifically targets decoder behavior.

## Parity Lane Decision

`make parity` should stay limited to exact-version passing RDKit writer-parity
claims with checked-in fixtures. The current runner loads the authoritative
pinned fixture families from `tests.helpers.pinned_rdkit_fixtures`:

- exact small support
- rooted random upstream cases
- serializer regressions
- writer-membership cases

RDKit-grounded tests that should stay outside `make parity`:

- `tests.rdkit_serialization.test_disconnected`: compatibility sampling over
  disconnected behavior, not an exact-version fixture corpus.
- `tests.rdkit_serialization.test_writer_flags`: dynamic high-draw support
  probe, not pinned fixture evidence.
- `tests.rdkit_serialization.test_known_quirks`: version-pinned RDKit behavior
  observations, but not passing writer-parity claims.
- `tests.run_known_stereo_gaps`: executable parity debt, intentionally separate
  from the passing parity lane.

This keeps `make parity` interpretable: a failure means a checked-in passing
RDKit writer-parity claim no longer holds for the installed pinned RDKit.

## Integrated Checklist

- [x] Baseline the current evidence from code.
  - [x] Add or run a small local report that counts cases by fixture family.
  - [x] Count dataset-derived cases from fixture `source` fields.
  - [x] Count coverage-ledger statuses.
  - [x] Once the report exists, avoid manually maintaining baseline counts in
        user-facing docs.

- [x] Define the fixture promotion rules in contributor docs.
  - [x] State that mining output is candidate data only.
  - [x] State positive evidence types.
  - [x] State case classifications separately from positive evidence.
  - [x] State when to choose exact support vs writer membership.
  - [x] State that `uncertain` mined cases are not promoted.

- [x] Add a generated pinned RDKit evidence summary.
  - [x] Generate from fixture JSON and the serializer coverage ledger.
  - [x] Include counts by fixture family.
  - [x] Include counts by source class: upstream, local probe, dataset-derived,
        random-writer observation, known RDKit gap, RDKit quirk.
  - [x] Fail on empty serializer ledgers and unknown serializer-ledger
        statuses.
  - [x] Do not include counts by writer feature until feature labels become a
        tested contract.
  - [x] Keep the generator as the source of the summary, not a manually edited
        table.

- [x] Improve source classification without adding fixture fat.
  - [x] First classify existing cases from `source` strings and fixture family.
  - [x] Fail on unclassified source strings instead of hiding them in an
        `other` bucket.
  - [x] Add explicit metadata only if string/family classification becomes
        ambiguous or untestable.
  - [x] Add loader tests for any new metadata before relying on it.

- [x] Add first deterministic dataset-mined writer-membership expansion.
  - [x] Run mining over sequential slices.
  - [x] Run separate scans for connected and disconnected molecules.
  - [x] Run separate scans for stereo and nonstereo surfaces.
  - [x] Run targeted scans for writer flags:
        `kekuleSmiles`, `allBondsExplicit`, `allHsExplicit`,
        `ignoreAtomMapNumbers`.
  - [x] Promote distinct clean deterministic membership cases across multiple
        writer surfaces.
  - [x] Avoid near-duplicate promotions from the same failure mechanism.

- [x] Continue dataset-mined writer-case expansion.
  - [x] Run random-slice mining.
  - [x] Promote small saturated random-slice cases into exact-support fixtures.
  - [x] Do not promote `rdkit_only` gap fixtures from this slice; no distinct
        deterministic `rdkit_only` candidates were found.

- [x] Promote small mined supports to exact-support fixtures.
  - [x] Identify mined cases with small support.
  - [x] Saturate with repeated RDKit sampling where applicable.
  - [x] Verify exact Grimace support and inventory.
  - [x] Add decoder and determinized decoder checks only when enumeration cost
        is bounded.

- [x] Add feature-matrix fixtures.
  - [x] Rooted connected nonstereo.
  - [x] All-roots connected nonstereo.
  - [x] Disconnected root in first fragment.
  - [x] Disconnected root in nonfirst fragment.
  - [x] Atom-map ignored and preserved surfaces.
  - [x] Explicit bonds.
  - [x] Explicit hydrogens.
  - [x] Kekule output.
  - [x] Isotopes and charges.
  - [x] Dative bond direction.
  - [x] Ring closure ordering.
  - [x] Tetrahedral chirality.
  - [x] Directional double-bond stereo.
  - [x] Coupled stereo minimizations, classified as exact support or
        `known-rdkit-gap` according to the current implementation.

- [x] Tighten upstream serializer coverage links.
  - [x] For each `covered` in-scope ledger entry, confirm fixture links point to
        executable cases.
  - [x] For each serializer-ledger `known-gap`, confirm there is an executable
        gap case.
  - [x] For each `out-of-scope`, keep the reason specific to Grimace's public
        surface.
  - [x] Keep `unreviewed` and `needs-fixture` at zero.

- [x] Decide what belongs in `make parity`.
  - [x] Review RDKit-grounded tests currently outside `tests.run_pinned_rdkit_parity`.
  - [x] Move exact-version parity claims into the parity runner when they are
        stable and not too slow.
  - [x] Keep broad compatibility or diagnostic tests outside parity when they
        are not exact-version pinned.

- [x] Add runtime cross-checks for small exact fixtures.
  - [x] `MolToSmilesEnum`.
  - [x] `MolToSmilesDecoder`.
  - [x] `MolToSmilesDeterminizedDecoder`.
  - [x] `MolToSmilesTokenInventory`.
  - [x] `MolToSmilesTokenInventorySuperset`.
  - [x] `MolToSmilesDeviation`.
  - [x] PreparedMol byte round trip where the public API supports it.

- [x] Keep the suite fast enough to matter.
  - [x] Put exhaustive small cases in normal parity.
  - [x] Put expensive mining and broad sampling behind explicit local scripts
        or perf-style lanes.
  - [x] Do not make CI depend on random discovery.
  - [x] Check in promoted fixtures, not raw mining logs.

## Next Meaningful Step

This checklist is complete for the current correctness-coverage slice. The next
meaningful work should be a new, explicitly scoped slice: either fix a known
stereo gap, add tested feature labels for generated reporting, or run another
mining campaign and promote only newly classified evidence.
