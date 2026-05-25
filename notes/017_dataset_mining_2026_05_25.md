# Dataset Mining Campaign, 2026-05-25

This campaign used `scripts/mine_rdkit_regressions.py` as candidate generation
only. Raw JSONL logs were written under:

```text
/home/ra/tmp/grimace-py-mining/2026-05-25-new-evidence/
```

The logs are not checked in. Checked-in fixtures are the promoted evidence.

## Scans

Four bounded deterministic scans were run against the bundled molecule
fixture:

- disconnected all-roots isomeric writer surface
- disconnected root-0 nonstereo with explicit bonds
- disconnected root-0 nonstereo with explicit hydrogens
- disconnected last-fragment-root nonstereo with explicit bonds

All completed checked cases in these scans were `clean`: RDKit's deterministic
writer output was contained in Grimace's exact support. The last-fragment-root
scan had one timeout; that molecule was not considered for promotion.

## Promotion rule

Most clean cases were not promoted. Ordinary salts and large supports add
little evidence beyond the existing fixtures.

Promoted cases had to be:

- dataset-derived
- exact-support bounded
- non-duplicative with existing fixture intent
- useful for a concrete writer surface

## Promoted evidence

Four cases were promoted to `rdkit_exact_small_support`:

- PubChem CID `24266`: disconnected all-roots fluorophosphate salt.
- PubChem CID `5935`: rooted disconnected explicit-H hydrochloride surface.
- PubChem CID `31239`: rooted disconnected explicit-bond aromatic pyridinium
  chloride surface.
- PubChem CID `92024769`: disconnected all-roots aluminosilicate fragments with
  aluminum and silicon bracket atoms.

These are exact-support cases, not membership-only cases.

No cases were promoted from the last-fragment-root explicit-bond scan. Its
small clean candidates overlapped existing disconnected-root evidence or the
newly promoted disconnected fragment surfaces.

## Follow-Up Explicit-Bond Scan

A later bounded sampled scan covered connected, all-roots, nonstereo
`allBondsExplicit=True` output:

```text
/home/ra/tmp/grimace-rdkit-mining-2026-05-25/connected_all_bonds_nonstereo_sampled.jsonl
```

The scan checked 60 molecules. It found 38 clean cases and 22 uncertain cases;
there were no `rdkit_only` regressions. Most clean cases duplicated existing
aliphatic, aromatic, or simple explicit-bond surfaces.

One case was promoted to `rdkit_exact_small_support`:

- PubChem CID `26042`: connected all-roots titanium dioxide with explicit
  bonds and bracketed metal/oxygen tokens.

## Follow-Up Dataset-Backed Writer-Surface Scans

A later promotion pass used three bounded scans:

```text
/home/ra/tmp/grimace-dataset-backed-expansion-2026-05-25/connected_isomeric_all_roots_deterministic.jsonl
/home/ra/tmp/grimace-dataset-backed-expansion-2026-05-25/connected_root0_all_hs_nonstereo_deterministic.jsonl
/home/ra/tmp/grimace-dataset-backed-expansion-2026-05-25/connected_root0_kekule_nonstereo_deterministic.jsonl
```

The scans checked 160, 120, and 120 molecules respectively. All checked cases
were `clean`.

Four bounded cases were promoted to `rdkit_exact_small_support`:

- PubChem CID `4173`: rooted explicit-H nitroimidazole with nitro charge
  tokens.
- PubChem CID `204`: rooted explicit-H urazole/urea carbonyl surface.
- PubChem CID `3385`: rooted kekule fluorouracil heterocycle.
- PubChem CID `5430`: rooted kekule fused benzimidazole/thiazole surface.

Five larger cases were promoted to `rdkit_writer_membership`:

- PubChem CID `5355130`: all-roots cinnamate ester with directional alkene.
- PubChem CID `5959`: stereo-rich chloramphenicol writer output.
- PubChem CID `4510`: all-roots trinitroglycerin nitrate surface.
- PubChem CID `54670067`: all-roots ascorbic-acid lactone/enediol surface.
- PubChem CID `6049`: all-roots EDTA polyacid/amine surface.

These promotions are deliberately feature-stratified. Clean candidates that
only repeated already-covered simple aromatic, aliphatic, or single-flag
surfaces were left in the raw logs.

An additional longer RDKit random-sampling check was run for these larger
candidate cases:

```text
/home/ra/tmp/grimace-long-rdkit-sampling-2026-05-25/long_sampling.jsonl
```

The five promoted cases saturated exactly against Grimace support under the
longer sampling budget. PubChem CID `444795` retinoic acid did not: the sampled
RDKit support and Grimace support both had 2000 strings, but differed by 220
strings in each direction. It was therefore left out of passing promoted
writer-membership evidence; it is a candidate for a separate minimized known
stereo/polyene gap.
