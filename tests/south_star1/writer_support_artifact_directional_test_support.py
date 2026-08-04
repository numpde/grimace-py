"""Directional support-artifact test helpers."""

from tests.south_star1.writer_support_artifact_domain_methods import *
from tests.south_star1.writer_support_artifact_fixtures import (
    directional_support_artifact_fixture,
)
from tests.south_star1.writer_support_artifact_fixtures import (
    shared_acyclic_directional_support_artifact_fixture,
)


def _directional_rooted_artifact():
    fixture = directional_support_artifact_fixture()
    return fixture.facts, fixture.runtime_options, fixture.artifact


def _shared_acyclic_directional_artifact():
    fixture = shared_acyclic_directional_support_artifact_fixture()
    return fixture.facts, fixture.runtime_options, fixture.artifact

__all__ = [name for name in globals() if not name.startswith("__")]
