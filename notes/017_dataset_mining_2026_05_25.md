# Dataset Mining Campaign, 2026-05-25

This campaign used `scripts/mine_rdkit_regressions.py` as candidate generation
only. Raw JSONL logs were written under:

```text
/home/ra/tmp/grimace-py-mining/2026-05-25-new-evidence/
```

The logs are not checked in. Checked-in fixtures are the promoted evidence.

## Scans

Three bounded deterministic scans were run against the bundled molecule
fixture:

- disconnected all-roots isomeric writer surface
- disconnected root-0 nonstereo with explicit bonds
- disconnected root-0 nonstereo with explicit hydrogens

All checked cases in these scans were `clean`: RDKit's deterministic writer
output was contained in Grimace's exact support.

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
