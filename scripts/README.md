# Scripts

Utility scripts for local development, validation, and release support.

## `mine_rdkit_regressions.py`

Local dataset miner for RDKit-derived writer regressions.

It scans the bundled `top_100000` fixture, computes the deterministic RDKit
string for a chosen public-surface mode, and checks whether that string is
contained in Grimace's exact support. Each molecule is evaluated in a
subprocess so slow or wedged cases can be skipped with a timeout.

It can also compare the public writer flags exposed by `MolToSmilesEnum(...)`,
including `kekuleSmiles`, `allBondsExplicit`, `allHsExplicit`, and
`ignoreAtomMapNumbers`.

Examples:

```bash
PYTHONPATH=python:. python3 scripts/mine_rdkit_regressions.py \
  --root none \
  --isomeric true \
  --connected connected \
  --max-atoms 30 \
  --limit 120
```

```bash
PYTHONPATH=python:. python3 scripts/mine_rdkit_regressions.py \
  --root last \
  --isomeric true \
  --connected connected \
  --start-after 444795 \
  --max-atoms 40
```

```bash
PYTHONPATH=python:. python3 scripts/mine_rdkit_regressions.py \
  --root zero \
  --isomeric false \
  --all-bonds-explicit true \
  --all-hs-explicit false \
  --connected connected \
  --max-atoms 30 \
  --limit 120
```
