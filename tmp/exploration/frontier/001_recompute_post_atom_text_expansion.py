"""Recompute the South Star frontier after recent atom-text expansion slices."""

from __future__ import annotations

from collections import Counter, defaultdict

from tests.helpers.south_star_adversarial_corpus import (
    generate_south_star_adversarial_candidates,
    south_star_adversarial_triage,
)
from tests.helpers.south_star_exact_support import load_south_star_expanded_support_cases


def main() -> None:
    triages = tuple(
        south_star_adversarial_triage(candidate)
        for candidate in generate_south_star_adversarial_candidates()
    )
    supported = tuple(triage for triage in triages if triage.supported_by_gate)
    unsupported = tuple(triage for triage in triages if not triage.supported_by_gate)

    print("## Adversarial Corpus")
    print()
    print(f"- candidates: {len(triages)}")
    print(f"- supported: {len(supported)}")
    print(f"- unsupported: {len(unsupported)}")
    print()
    print("| candidate | source | unsupported categories | boundary targets |")
    print("| --- | --- | --- | --- |")
    for triage in unsupported:
        candidate = triage.candidate
        print(
            f"| `{candidate.candidate_id}` | `{candidate.source_smiles}` | "
            f"`{', '.join(triage.unsupported_categories)}` | "
            f"`{', '.join(candidate.boundary_targets)}` |"
        )

    print()
    print("Unsupported category counts:")
    for category, count in sorted(
        Counter(
            category
            for triage in unsupported
            for category in triage.unsupported_categories
        ).items()
    ):
        print(f"- `{category}`: {count}")

    print()
    print("Supported boundary-target counts:")
    supported_targets: dict[str, int] = defaultdict(int)
    for triage in supported:
        for target in triage.candidate.boundary_targets:
            supported_targets[target] += 1
    for target, count in sorted(supported_targets.items()):
        print(f"- `{target}`: {count}")

    print()
    print("## Expanded Fixture Feature Areas")
    print()
    feature_counts = Counter(
        case.feature_area for case in load_south_star_expanded_support_cases()
    )
    for feature_area, count in sorted(feature_counts.items()):
        print(f"- `{feature_area}`: {count}")


if __name__ == "__main__":
    main()
