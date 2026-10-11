from __future__ import annotations

import sys
import unittest

from rdkit import rdBase

from tests.south_star2.test_pinned_stereo_evidence import PinnedStereoEvidenceTests


PINNED_RDKIT_VERSION = "2026.03.1"


def main() -> int:
    if rdBase.rdkitVersion != PINNED_RDKIT_VERSION:
        raise RuntimeError(
            "South Star 2 stereo evidence requires RDKit "
            f"{PINNED_RDKIT_VERSION}; installed version is {rdBase.rdkitVersion}"
        )
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(PinnedStereoEvidenceTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
