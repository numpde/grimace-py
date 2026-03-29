"""Shared assertion helpers for support and parity tests."""

from __future__ import annotations

import unittest

from tests.helpers.tokenization import expected_next_tokens_from_support


def assert_support_valid(
    test_case: unittest.TestCase,
    prepared,
    root_idx: int,
    support,
    validator,
) -> None:
    test_case.assertEqual([], validator(prepared, root_idx, None, support))


def assert_prefix_options_match_outputs(
    test_case: unittest.TestCase,
    prefix: str,
    options,
    outputs,
    *,
    atom_tokens,
) -> None:
    expected = expected_next_tokens_from_support(outputs, prefix, atom_tokens=atom_tokens)
    test_case.assertEqual(expected, tuple(sorted(options)))
