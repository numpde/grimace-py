"""Shared assertion helpers for support and parity tests."""

from __future__ import annotations

import unittest


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
) -> None:
    test_case.assertTrue(options)
    test_case.assertTrue(
        any(output.startswith(prefix + token) for output in outputs for token in options)
    )
