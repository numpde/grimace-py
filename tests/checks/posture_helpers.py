import re
import unittest


def assert_before(test: unittest.TestCase, text: str, earlier: str, later: str) -> None:
    earlier_index = text.find(earlier)
    later_index = text.find(later)
    test.assertNotEqual(earlier_index, -1, earlier)
    test.assertNotEqual(later_index, -1, later)
    test.assertLess(earlier_index, later_index)


def line_count(text: str, pattern: str) -> int:
    return len(re.findall(rf"(?m)^{pattern}$", text))
