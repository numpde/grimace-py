#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_INPUT = REPO_ROOT / "docs" / "prepared-mol-zstd-timings.tsv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "docs" / "prepared-mol-zstd-timings-plots"

MODE_STYLE = {
    "no-dictionary": {
        "label": "No dictionary",
        "color": "#b45309",
        "marker": "o",
    },
    "dictionary": {
        "label": "Dictionary",
        "color": "#2563eb",
        "marker": "s",
    },
}


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, dialect="excel-tab"))


def _draw_scatter(
    rows: list[dict[str, str]],
    *,
    time_field: str,
    std_field: str,
    ylabel: str,
    note: str,
    log_time: bool,
    note_location: str,
    output: Path,
) -> None:
    from plox import Plox

    with Plox(
        {
            "figure.figsize": (7.2, 5.0),
            "figure.constrained_layout.use": True,
        }
    ) as px:
        ax = px.a
        for mode, style in MODE_STYLE.items():
            mode_rows = sorted(
                (row for row in rows if row["mode"] == mode),
                key=lambda row: int(row["level"]),
            )
            if not mode_rows:
                raise SystemExit(f"No {mode!r} rows in timing input")
            levels = [int(row["level"]) for row in mode_rows]
            x = [float(row["compression_ratio"]) * 100 for row in mode_rows]
            y = [float(row[time_field]) * 1_000 for row in mode_rows]
            yerr = [float(row[std_field]) * 1_000 for row in mode_rows]
            if log_time and any(value - error <= 0 for value, error in zip(y, yerr)):
                raise SystemExit(
                    f"{mode!r} {time_field} has a non-positive lower error bound; "
                    "log-scale additive error bars need fresh timing samples or a "
                    "different uncertainty statistic."
                )
            ax.plot(
                x,
                y,
                linestyle="--",
                linewidth=1.0,
                color=style["color"],
                alpha=0.7,
            )
            ax.errorbar(
                x,
                y,
                yerr=yerr,
                fmt=style["marker"],
                color=style["color"],
                ecolor=style["color"],
                elinewidth=0.8,
                capsize=3,
                markersize=6,
                label=style["label"],
            )
            for x_value, y_value, level in zip(x, y, levels):
                ax.annotate(
                    str(level),
                    (x_value, y_value),
                    xytext=(5, 4),
                    textcoords="offset points",
                    fontsize="small",
                    color=style["color"],
                )

        ax.set_xlabel("Compression ratio, %")
        ax.set_ylabel(ylabel)
        ax.set_xlim(3.5, 14.5)
        if log_time:
            ax.set_yscale("log")
        ax.grid(color="#dddddd", linewidth=0.8)
        ax.set_axisbelow(True)
        note_x, note_y, note_va = {
            "lower-left": (0.02, 0.02, "bottom"),
            "upper-left": (0.02, 0.98, "top"),
        }[note_location]
        ax.text(
            note_x,
            note_y,
            note,
            transform=ax.transAxes,
            fontsize="small",
            va=note_va,
            ha="left",
            bbox={
                "boxstyle": "round,pad=0.25",
                "facecolor": "white",
                "edgecolor": "#dddddd",
                "alpha": 0.9,
            },
        )
        ax.legend(frameon=True, edgecolor="#dddddd")
        px.f.savefig(output)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot PreparedMol zstd compression timing tradeoffs.",
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    rows = _read_rows(args.input)
    if not rows:
        raise SystemExit(f"No rows in {args.input}")
    sample_counts = {row["sample_count"] for row in rows}
    if len(sample_counts) != 1:
        raise SystemExit("Timing input must use one sample_count")
    sample_count = next(iter(sample_counts))
    for field in (
        "recorded_at_utc",
        "git_commit",
        "git_change",
        "git_dirty",
        "platform",
        "python",
        "rdkit",
        "zstandard",
        "zstd_library",
        "cpu_model",
        "visible_cpus",
        "cgroup_memory_limit_bytes",
        "container",
        "sample_policy",
        "sample_seed",
        "sample_source_rows_sha256",
        "sample_cids_sha256",
        "dictionary_artifact",
        "dictionary_id",
        "dictionary_sha256",
    ):
        values = {row[field] for row in rows}
        if len(values) != 1:
            raise SystemExit(f"Timing input must use one {field}")
    dictionary_artifact = rows[0]["dictionary_artifact"]
    dictionary_id = rows[0]["dictionary_id"]
    dictionary_sha = rows[0]["dictionary_sha256"][:12]
    note = (
        f"molecules: {sample_count}\n"
        f"dictionary: {dictionary_artifact}\n"
        f"id: {dictionary_id}; sha256: {dictionary_sha}..."
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for path in args.output_dir.glob("*.png"):
        path.unlink()

    compression_output = args.output_dir / "compression-ratio-vs-compression-time.png"
    decompression_output = (
        args.output_dir / "compression-ratio-vs-decompression-time.png"
    )
    _draw_scatter(
        rows,
        time_field="compression_mean_s",
        std_field="compression_std_s",
        ylabel=f"Compression time for {sample_count} payloads, ms",
        note=note,
        log_time=True,
        note_location="lower-left",
        output=compression_output,
    )
    _draw_scatter(
        rows,
        time_field="decompression_mean_s",
        std_field="decompression_std_s",
        ylabel=f"Decompression time for {sample_count} payloads, ms",
        note=note,
        log_time=False,
        note_location="upper-left",
        output=decompression_output,
    )
    print(f"Wrote {compression_output}")
    print(f"Wrote {decompression_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
