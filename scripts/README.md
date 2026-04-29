# Scripts

Utility scripts for local development, validation, and release support.

Examples below assume you are running them from the same Python environment
where `grimace` is installed or built, for example an activated `.venv`.
Using a different interpreter can silently compare against a different RDKit or
extension build than the one you intended to validate. These scripts are
strictly local validation tools; they are not a supported public API surface.

Performance and profiling diagnostics should live here or under `tests/perf/`,
not as `#[ignore]` probes inside the Rust test binary. `cargo test` is kept for
bounded correctness checks; heavier profiling runs should be explicit.

## Performance Tooling

- `tests/perf/test_readme_timings.py`
  - measures the public timing table and writes `docs/timings.tsv` plus
    `docs/timings.md`
- `record_perf_hotspots.py`
  - records focused whole-process `perf` hotspots and appends them to
    `notes/004_perf_history.jsonl`
  - optionally saves the full report under `notes/perf_reports/`

## `mine_rdkit_regressions.py`

Local dataset miner for RDKit-derived writer regressions.

It scans the bundled `top_100000` fixture and can either:

- compare the deterministic RDKit writer output against Grimace support
- sample RDKit random writer outputs until a simple plateau heuristic fires
  and classify the case as `clean`, `rdkit_only`, `grimace_only`, or
  `uncertain`

`grimace_only` is a confirmed status: the miner only reports it after an
initial plateaued miss survives a higher-budget confirmation pass across
additional RDKit RNG seeds. That avoids treating single-seed sampling artifacts
as real support gaps.

In practice, most large sampled cases end up as `uncertain`, not because they
contradict Grimace support, but because exact support grows much faster than a
reasonable RDKit draw budget. The highest-signal correctness failure remains
`rdkit_only`: RDKit emitted a string that Grimace cannot produce.

Each molecule is evaluated in a subprocess so slow or wedged cases can be
skipped with a timeout.

For long scans, `--jsonl-output` writes one JSON record per event and
`--resume-jsonl` resumes from the last recorded CID in that file. The resume
counter is cumulative: if the file already contains `170` checked cases and you
rerun with `--limit 200 --resume-jsonl`, the resumed run stops after the next
`30` checked cases.

It can also compare the public writer flags exposed by `MolToSmilesEnum(...)`,
including `kekuleSmiles`, `allBondsExplicit`, `allHsExplicit`, and
`ignoreAtomMapNumbers`.

Examples:

```bash
python scripts/mine_rdkit_regressions.py \
  --root none \
  --isomeric true \
  --connected connected \
  --max-atoms 30 \
  --limit 120
```

```bash
python scripts/mine_rdkit_regressions.py \
  --root last \
  --isomeric true \
  --connected connected \
  --start-after 444795 \
  --max-atoms 40
```

```bash
python scripts/mine_rdkit_regressions.py \
  --root zero \
  --isomeric false \
  --all-bonds-explicit true \
  --all-hs-explicit false \
  --connected connected \
  --max-atoms 30 \
  --limit 120
```

```bash
python scripts/mine_rdkit_regressions.py \
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

```bash
python scripts/mine_rdkit_regressions.py \
  --root none \
  --isomeric true \
  --rdkit-mode sampled \
  --connected connected \
  --max-atoms 30 \
  --limit 200 \
  --jsonl-output tmp/rdkit-scan.jsonl
```

```bash
python scripts/mine_rdkit_regressions.py \
  --root none \
  --isomeric true \
  --rdkit-mode sampled \
  --connected connected \
  --max-atoms 30 \
  --limit 400 \
  --jsonl-output tmp/rdkit-scan.jsonl \
  --resume-jsonl
```

If you intentionally want to run against an uninstalled source tree instead of
an installed or `maturin develop` build, prepend `PYTHONPATH=python:.`.

## `extract_rdkit_serializer_cases.py`

Parser-based extractor for the local RDKit serializer source fixture.

It reads `tests/fixtures/rdkit_upstream_serializer_sources/2026.03.1/` and
writes the upstream serializer coverage inventory:

- `tests/fixtures/rdkit_upstream_serializer_coverage/2026.03.1.json`

The extractor uses:

- `tree-sitter-cpp` for RDKit C++ Catch2 `TEST_CASE` and `SECTION` blocks
- Python `ast` for RDKit Python `test*` functions/methods
- `tree-sitter-java` for RDKit Java `test*` methods

It owns generated fields such as upstream file, line range, parser kind,
matched serializer terms, and snippet hash.  Reviewed coverage fields such as
`status`, `claim`, `grimace_links`, and `notes` are preserved when
regenerating.

Regenerate after changing the upstream source fixture:

```bash
python scripts/extract_rdkit_serializer_cases.py --write
```

Check that the committed inventory is current:

```bash
python scripts/extract_rdkit_serializer_cases.py --check
```

## `report_rdkit_serializer_coverage.py`

Summarizes the RDKit serializer coverage ledger before and during triage.

```bash
python scripts/report_rdkit_serializer_coverage.py
```

Show the first unreviewed entries:

```bash
python scripts/report_rdkit_serializer_coverage.py --status unreviewed
```

Once triage is expected to be complete, make remaining `unreviewed` or
`needs-fixture` entries fail explicitly:

```bash
python scripts/report_rdkit_serializer_coverage.py --fail-untriaged
```
