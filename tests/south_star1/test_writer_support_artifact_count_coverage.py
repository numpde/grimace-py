"""Count-DAG and support-image coverage contracts."""

import unittest

from tests.south_star1.writer_support_artifact_domain_methods import WriterSupportArtifactDomainMethods


def _selected(name: str) -> bool:
    return name.startswith((
        "test_count_dag_",
        "test_support_image_coverage_",
        "test_coverage_",
    ))


class WriterSupportArtifactCountCoverageTest(unittest.TestCase):
    pass


for _name, _method in vars(WriterSupportArtifactDomainMethods).items():
    if _selected(_name):
        setattr(WriterSupportArtifactCountCoverageTest, _name, _method)
