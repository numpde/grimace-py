from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
from typing import TypeVar


T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class FirstOccurrenceOutputOrderPolicy:
    name: str = "first_occurrence_deduplication"

    def deduplicate(self, outputs: Iterable[T]) -> tuple[T, ...]:
        return tuple(dict.fromkeys(outputs))
