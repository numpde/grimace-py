from __future__ import annotations

import sys
import unittest

from tests.south_star2.test_pinned_stereo_evidence import PinnedStereoEvidenceTests


def main() -> int:
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(PinnedStereoEvidenceTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
