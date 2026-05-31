# Timing lane cleanup

This checklist defines a no-regret cleanup for the current timing and
PreparedMol zstd benchmark lanes. The goal is structural symmetry where the
work is genuinely the same, and explicit asymmetry where it is not.

## Target shape

- [x] Keep the PreparedMol zstd dictionary lane separate.
  - It creates a shipped runtime artifact.
  - It has no enum/support benchmark counterpart.
  - Its current conceptual name, `prepared-mol-zstd-dictionary`, is still
    accurate.
- [x] Rename the enum/support timing lane from the generic `perf` name to an
      explicit benchmark name.
  - Target public lane: `make timings-enum`.
  - Removed the temporary `make perf` alias before release.
- [x] Rename the PreparedMol zstd timing lane to the same timing namespace.
  - Target public lane: `make timings-prepared-mol-zstd`.
  - Removed the temporary `make prepared-mol-zstd-timings` alias before
    release.
- [x] Use parallel path names for checked-in timing artifacts.
  - Enum TSV: `docs/timings-enum.tsv`.
  - Enum plots: `docs/timings-enum-plots/`.
  - PreparedMol zstd TSV: `docs/timings-prepared-mol-zstd.tsv`.
  - PreparedMol zstd plots: `docs/timings-prepared-mol-zstd-plots/`.
- [x] Make `docs/timings.md` an overview page.
  - Link to enum/support timings.
  - Link to PreparedMol zstd timings.
  - Do not overload one benchmark page as the timing index.
- [x] Add one page per benchmark.
  - Enum/support page: `docs/timings-enum.md`.
  - PreparedMol zstd page: `docs/timings-prepared-mol-zstd.md`.

## Naming inventory

- [x] Replace vague names for enum/support timing internals.
  - Compose file: `compose/timings-enum.yml`.
  - Container directory: `containers/timings-enum/`.
  - Docker image: `grimace-py-timings-enum:local`.
- [x] Replace zstd timing internals with the same order.
  - `compose/prepared-mol-zstd-timings.yml` ->
    `compose/timings-prepared-mol-zstd.yml`.
  - `containers/prepared-mol-zstd-timings/` ->
    `containers/timings-prepared-mol-zstd/`.
  - Docker image `grimace-py-prepared-mol-zstd-timings:local` ->
    `grimace-py-timings-prepared-mol-zstd:local`.
- [x] Rename zstd timing scripts to match the lane.
  - `scripts/prepared_mol_zstd_timings_measure.py` ->
    `scripts/timings_prepared_mol_zstd_measure.py`.
  - `scripts/prepared_mol_zstd_timings_plot.py` ->
    `scripts/timings_prepared_mol_zstd_plot.py`.
- [x] Give enum/support timing scripts first-class names.
  - Extracted measurement/rendering from `tests/perf/test_readme_timings.py`.
  - Target measurement script: `scripts/timings_enum_measure.py`.
  - Target rendering script: `scripts/timings_enum_render.py` or
    `scripts/timings_enum_plot.py`, depending on whether it owns markdown,
    plots, or both.
  - Keep tests as tests; do not hide artifact generation inside a unittest
    module long-term.

## Shared timing metadata

- [x] Stop importing timing environment metadata from a test module in scripts.
  - Moved environment detection to `scripts/timing_environment.py`.
  - Moved timing history and perf-report helpers to `scripts/timing_history.py`.
  - Timing scripts no longer import `tests`.
- [x] Keep one source of truth for:
  - git commit
  - git change subject
  - git dirty flag
  - recorded UTC timestamp
  - platform
  - Python version
  - RDKit version
  - CPU model
  - visible CPU count
  - cgroup memory limit
- [x] Keep benchmark-specific environment additions local to each benchmark.
  - PreparedMol zstd adds `zstandard` and `zstd_library`.
  - Enum/support does not need zstd fields.
- [x] Rename `PERF_GIT_METADATA_ENV`.
  - Target: `TIMING_GIT_METADATA_ENV`.
  - Use it for all timing lanes.
  - Do not duplicate git shell snippets in Makefile recipes.
- [x] Keep the metadata fields repeated in TSV rows for grep-friendly,
      self-contained artifacts.
  - Validate single-valued metadata before plotting/rendering.
  - Do not put hostnames into public docs or TSVs unless there is a deliberate
    privacy decision.

## Enum/support benchmark extraction

- [x] Separate artifact generation from unittest mechanics.
  - Kept a thin opt-in test wrapper under `tests/perf/`.
  - Moved timing measurement to a script callable by the container entrypoint.
  - Moved markdown/table/plot rendering to script code.
- [x] Preserve current behavior while moving code.
  - Same molecules.
  - Same surfaces.
  - Same columns unless intentionally renamed.
  - Same plot semantics.
  - Same environment section in rendered docs.
- [x] Keep the history append policy explicit.
  - Existing enum/support lane appends to `notes/004_perf_history.jsonl`.
  - Decide whether to keep that file as historical perf/timing history or add a
    new `notes/timings-enum-history.jsonl`.
  - Do not rewrite old history as part of the first cleanup unless there is a
    clear payoff.
- [x] Keep focused hotspot profiling separate.
  - `scripts/record_perf_hotspots.py` and `notes/perf_reports/` are profiler
    diagnostics, not public timing lanes.
  - Do not rename them only to satisfy timing-lane symmetry.

## PreparedMol zstd timing page

- [x] Add a markdown page for the zstd timing benchmark.
  - Explain what is measured: per-molecule `PreparedMol` raw bytes compressed
    with zstd across levels 1 through 19.
  - Explain the two modes: no dictionary and shipped dictionary.
  - State sample size and selection policy.
  - State dictionary artifact, zstd dictionary ID, and manifest hash prefix.
  - State benchmark environment from the TSV.
  - Include both plots.
  - Link to the raw TSV.
- [x] Keep the page about benchmark interpretation, not implementation trivia.
  - Avoid repeating the full dictionary design note.
  - Link to the PreparedMol zstd note only if deeper rationale is useful.
- [x] Ensure plot captions are outside images.
  - The figure should not bake benchmark context into pixels when text belongs
    in docs.

## Makefile cleanup

- [x] Introduce explicit artifact variables for both timing lanes.
  - `TIMINGS_ENUM_ARTIFACT_FILES`.
  - `TIMINGS_ENUM_ARTIFACT_DIRS`.
  - `TIMINGS_PREPARED_MOL_ZSTD_ARTIFACT_FILES`.
  - `TIMINGS_PREPARED_MOL_ZSTD_ARTIFACT_DIRS`.
- [x] Rename artifact guards accordingly.
  - `TIMINGS_ENUM_ARTIFACTS_GUARD`.
  - `TIMINGS_PREPARED_MOL_ZSTD_ARTIFACTS_GUARD`.
- [x] Keep the guard behavior strict.
  - Refuse missing files.
  - Refuse symlinks or paths resolved outside the repository.
  - Bind only the exact writable artifacts needed by each lane.
- [x] Update `make help`.
  - Prefer `timings-enum`.
  - Prefer `timings-prepared-mol-zstd`.
  - Mention `perf` only as a temporary alias if it remains.
- [x] Keep CI lanes separate from timing lanes.
  - Timing lanes remain opt-in.
  - `make ci` should not run benchmark regeneration.

## Compose and container cleanup

- [x] Rename compose services to match lane names.
  - `timings-enum`.
  - `timings-prepared-mol-zstd`.
- [x] Keep both services strict.
  - non-root runtime user from Makefile
  - `network_mode: "none"` at runtime
  - `read_only: true`
  - `cap_drop: [ALL]`
  - `no-new-privileges:true`
  - bounded `pids_limit`
  - bounded `mem_limit`
  - only required `tmpfs`
- [x] Keep build-time dependency download in image builds only.
  - Runtime services should not need network.
  - Runtime services should not install packages.
- [x] Keep writable bind mounts narrow.
  - Enum/support timing lane writes only enum timing artifacts and history.
  - PreparedMol zstd timing lane writes only zstd timing artifacts.
- [x] Update posture tests for all renamed files, services, images, and mounts.

## Documentation cleanup

- [x] Update the docs navigation.
  - `Timings` should point to the overview.
  - Overview should link to both benchmark pages.
- [x] Update `docs/development/containerized.md`.
  - Explain the two timing lanes side by side.
  - Keep dictionary generation separate.
  - List output paths exactly once.
- [x] Update `scripts/README.md`.
  - Group enum/support timing scripts together.
  - Group PreparedMol zstd timing scripts together.
  - Keep dictionary generation separate.
- [x] Search docs for stale names.
  - `make perf`
  - `prepared-mol-zstd-timings`
  - `docs/timings.tsv`
  - `docs/timing-plots`
  - `compose/perf.yml`
  - `containers/perf`
  - Stale names remain only in this historical checklist and negative posture
    assertions.

## Compatibility policy

- [x] Remove the temporary `make perf` alias before release.
  - The branch has not shipped with `make timings-enum` yet.
  - Keeping both names would make the cleanup less explicit.
- [x] Remove the temporary `make prepared-mol-zstd-timings` alias before release.
  - The branch has not shipped with `make timings-prepared-mol-zstd` yet.
  - Keeping both names would preserve stale terminology.
- [x] Do not keep old artifact paths as duplicated generated files.
  - Move artifacts with `git mv`.
  - Update all docs and tests to the new paths.
  - Avoid checking in identical old and new TSVs.
- [x] Do not rename the dictionary artifact layout as part of timing cleanup.
  - Runtime package data paths are a separate compatibility surface.

## Verification checklist

- [x] Run fast local syntax checks after script moves.
  - `python -m py_compile ...`
- [x] Run script help smoke tests.
  - Enum/support timing scripts should load without benchmark execution.
  - PreparedMol zstd timing scripts should load without benchmark execution.
- [x] Validate compose configs.
  - `docker compose -f compose/timings-enum.yml config`
  - `docker compose -f compose/timings-prepared-mol-zstd.yml config`
  - `docker compose -f compose/prepared-mol-zstd-dictionary.yml config`
- [x] Run repository checks after each meaningful rename.
  - `make checks`
- [x] Run the renamed enum/support timing lane once after extraction.
  - `make timings-enum`
  - Confirm TSV, markdown, plots, and history update.
  - Confirm environment metadata is present and single-valued.
- [x] Run the renamed PreparedMol zstd timing lane once after path changes.
  - `make timings-prepared-mol-zstd`
  - Confirm TSV and plots update.
  - Confirm environment metadata is present and single-valued.
- [x] Build docs.
  - `make docs`
  - Confirm timing overview links to both benchmark pages.
  - Confirm no stale artifact links.
- [x] Run a final stale-name search.
  - Old names remain only in this historical checklist and negative posture
    assertions.
- [x] Inspect git diff before each commit.
  - File moves should be moves, not accidental delete/recreate churn where
    avoidable.
  - Regenerated artifacts should be expected and limited.

## Commit sequence

- [x] Commit 1: add this checklist.
- [x] Commit 2: extract shared timing environment and rename Makefile metadata
      helper.
- [x] Commit 3: rename PreparedMol zstd timing lane, compose file, container,
      scripts, and artifacts.
- [x] Commit 4: add PreparedMol zstd timing docs page and docs links.
- [x] Commit 5: extract enum/support timing scripts from unittest code.
- [x] Commit 6: rename enum/support lane, compose file, container, and
      artifacts.
- [x] Commit 7: make `docs/timings.md` the overview page and move current enum
      content to `docs/timings-enum.md`.
- [x] Commit 8: move timing history helpers out of `tests`.
- [x] Commit 9: remove temporary timing aliases.
- [x] Commit 10: make enum/support timing TSV metadata self-contained.
- [x] Commit 11: refresh PreparedMol zstd timing artifacts from a clean tree.
- [x] Commit 12: refresh enum/support timing artifacts from a clean tree.

## Done criteria

- [x] Timing lanes are named by what they measure.
- [x] Dictionary generation remains separate from benchmark timing.
- [x] No script imports benchmark environment metadata from `tests`.
- [x] Makefile has one git-metadata helper for all timing lanes.
- [x] Compose and container paths are parallel for both timing benchmarks.
- [x] Checked-in timing artifacts use parallel names.
- [x] Docs have a timing overview and one page per benchmark.
- [x] Stale names are limited to this historical note and negative posture
      assertions.
- [x] `make checks`, both timing lanes, and `make docs` pass.
