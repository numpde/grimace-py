# Exploratory Scripts

This directory contains scratch investigations that are useful historical
context, but are not part of the supported test suite or runtime.

Scripts are grouped by topic and numbered chronologically within each topic.
Prefer adding new scripts as `NNN_short_description.py` instead of placing new
files directly under `tmp/`.

## `stereo_assignment/`

Bond-stereo and RDKit serializer-policy investigations. The current sequence
starts with local carrier-choice probes, then moves through RDKit sampled
parity checks, Z3 constraint modeling, smallest-witness search, and finally
RDKit bond-direction/traversal-policy probes.

The most current scripts are:

- `027_investigate_rdkit_bond_dir_policy.py`: compares semantic-local carrier
  constraints with RDKit's local non-stereo-double cleanup behavior.
- `028_investigate_rdkit_traversal_coupling.py`: inspects the reduced
  porphyrin 16-vs-12 discrepancy using RDKit output-order metadata.

