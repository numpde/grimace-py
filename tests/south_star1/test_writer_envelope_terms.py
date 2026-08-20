"""Focused safety tests for writer-envelope term memoization."""

from __future__ import annotations

from dataclasses import dataclass
import unittest

from grimace._south_star1.writer_envelope_terms import _canonical_json
from grimace._south_star1.writer_envelope_terms import _digest
from grimace._south_star1.writer_envelope_terms import _memoize_writer_envelope_terms
from grimace._south_star1.writer_envelope_terms import _TERM_CACHE
from grimace._south_star1.writer_envelope_terms import _term


@dataclass(frozen=True)
class FrozenEnvelope:
    value: str


@dataclass
class MutableEnvelope:
    value: object


class WriterEnvelopeTermsTest(unittest.TestCase):
    def test_memoized_and_plain_terms_have_identical_bytes_and_digests(self) -> None:
        values = (
            FrozenEnvelope("cursor"),
            FrozenEnvelope("branch"),
            FrozenEnvelope("terminal"),
        )
        plain = tuple(_term(value) for value in values)
        with _memoize_writer_envelope_terms():
            memoized = tuple(_term(value) for value in values)
        self.assertEqual(
            tuple(_canonical_json(value) for value in memoized),
            tuple(_canonical_json(value) for value in plain),
        )
        self.assertEqual(
            tuple(_digest(value) for value in memoized),
            tuple(_digest(value) for value in plain),
        )

    def test_mutable_values_are_not_returned_from_a_stale_cache(self) -> None:
        values = [
            ["before"],
            {"key": "before"},
            {"before"},
            MutableEnvelope("before"),
        ]
        with _memoize_writer_envelope_terms():
            for value in values:
                before = _term(value)
                if isinstance(value, list):
                    value[0] = "after"
                elif isinstance(value, dict):
                    value["key"] = "after"
                elif isinstance(value, set):
                    value.remove("before")
                    value.add("after")
                else:
                    value.value = "after"
                self.assertNotEqual(_term(value), before)

    def test_cycles_in_containers_and_dataclasses_are_rejected(self) -> None:
        values = []
        cycle_list = []
        cycle_list.append(cycle_list)
        cycle_mapping = {}
        cycle_mapping["self"] = cycle_mapping
        cycle_dataclass = MutableEnvelope(None)
        cycle_dataclass.value = cycle_dataclass
        values.extend((cycle_list, cycle_mapping, cycle_dataclass))
        for value in values:
            with self.subTest(type=type(value).__name__), self.assertRaisesRegex(
                ValueError, "cyclic_writer_envelope_term"
            ):
                _term(value)

    def test_repeated_frozen_dataclass_serialization_hits_the_cache(self) -> None:
        value = FrozenEnvelope("cached")
        with _memoize_writer_envelope_terms():
            first = _term(value)
            second = _term(value)
            cache = _TERM_CACHE.get()
            self.assertIsNotNone(cache)
            self.assertIs(cache[id(value)][0], value)
            self.assertIs(first, second)

    def test_exiting_context_clears_memoized_state(self) -> None:
        value = FrozenEnvelope("cleared")
        with _memoize_writer_envelope_terms():
            _term(value)
            self.assertIsNotNone(_TERM_CACHE.get())
        self.assertIsNone(_TERM_CACHE.get())


if __name__ == "__main__":
    unittest.main()
