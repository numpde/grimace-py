# Scripts

Utility scripts for local development, validation, and release support.

## `mine_rdkit_regressions.py`

Local dataset miner for RDKit-derived writer regressions.

It scans the bundled `top_100000` fixture and can either:

- compare the deterministic RDKit writer output against Grimace support
- sample RDKit random writer outputs until a simple plateau heuristic fires
  and classify the case as `clean`, `rdkit_only`, `grimace_only`, or
  `uncertain`

Each molecule is evaluated in a subprocess so slow or wedged cases can be
skipped with a timeout.

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

```bash
PYTHONPATH=python:. python3 scripts/mine_rdkit_regressions.py \
  --root none \
  --isomeric true \
  --rdkit-mode sampled \
  --draws-per-round 40 \
  --stagnation-rounds 5 \
  --max-draws 400 \
  --connected connected \
  --max-atoms 25 \
  --limit 80
```
