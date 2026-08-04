"""Copy-on-read fixtures for rich support-artifact contract tests."""

from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache
from grimace._south_star1.facts import MoleculeFacts
from grimace._south_star1.policy import SmilesPolicy
from grimace._south_star1.prepared_runtime import SouthStarRuntimeOptions
from grimace._south_star1.rdkit_adapter import RdkitOrdinaryExtractionOptions
from grimace._south_star1.rdkit_adapter import ordinary_molecule_facts_from_smiles
from grimace._south_star1.writer_support_artifact_envelope import writer_support_artifact_envelope_for_prefix_read
from grimace._south_star1.writer_support_artifact_envelope import writer_support_artifact_envelope_for_snapshot
from grimace._south_star1.writer_snapshot_prefix_envelope import writer_snapshot_prefix_read_envelope_for_emitted_texts
from tests.south_star1.helpers import directional_facts
from tests.south_star1.helpers import shared_acyclic_directional_facts
from tests.south_star1.helpers import tetrahedral_facts
from tests.south_star1.helpers import two_atom_facts
from tests.south_star1.writer_test_context import initial_writer_snapshot
from tests.south_star1.writer_test_context import prepare_writer_facts
from tests.south_star1.writer_test_context import writer_runtime_options
from tests.south_star1.writer_test_fixtures import terminal_tetra_center_facts
from tests.south_star1.writer_test_fixtures import terminal_tetra_center_policy

@dataclass(frozen=True, slots=True)
class WriterSupportArtifactFixture:
    facts: MoleculeFacts
    runtime_options: SouthStarRuntimeOptions
    artifact: dict[str, object]


def _build_snapshot_artifact(
    facts: MoleculeFacts,
    *,
    rooted_at_atom: int = -1,
    policy: SmilesPolicy | None = None,
) -> WriterSupportArtifactFixture:
    options = writer_runtime_options(rooted_at_atom=rooted_at_atom)
    prepared = prepare_writer_facts(facts, policy=policy)
    artifact = writer_support_artifact_envelope_for_snapshot(
        prepared=prepared,
        snapshot=initial_writer_snapshot(prepared, options),
    )
    return WriterSupportArtifactFixture(facts, options, deepcopy(artifact))


def support_artifact_fixture(
    facts: MoleculeFacts,
    *,
    rooted_at_atom: int = -1,
    policy: SmilesPolicy | None = None,
) -> WriterSupportArtifactFixture:
    """Build one uncached fixture and return an owned artifact copy."""
    return _build_snapshot_artifact(
        facts,
        rooted_at_atom=rooted_at_atom,
        policy=policy,
    )


def rdkit_graph_facts(smiles: str) -> MoleculeFacts:
    return ordinary_molecule_facts_from_smiles(
        smiles,
        RdkitOrdinaryExtractionOptions(include_potential_sites=False),
    )


@lru_cache(maxsize=None)
def _cached_rdkit_support_artifact_fixture(
    smiles: str,
) -> WriterSupportArtifactFixture:
    return _build_snapshot_artifact(rdkit_graph_facts(smiles))


def _copy_fixture(
    fixture: WriterSupportArtifactFixture,
) -> WriterSupportArtifactFixture:
    return WriterSupportArtifactFixture(
        fixture.facts,
        fixture.runtime_options,
        deepcopy(fixture.artifact),
    )


def rdkit_support_artifact_fixture(smiles: str) -> WriterSupportArtifactFixture:
    return _copy_fixture(_cached_rdkit_support_artifact_fixture(smiles))


@lru_cache(maxsize=1)
def _cached_completed_prefix_support_artifact_fixture():
    facts = two_atom_facts()
    options = writer_runtime_options()
    prepared = prepare_writer_facts(facts)
    prefix = writer_snapshot_prefix_read_envelope_for_emitted_texts(
        prepared=prepared,
        snapshot=initial_writer_snapshot(prepared, options),
        emitted_texts=("C", "C"),
    )
    artifact = writer_support_artifact_envelope_for_prefix_read(
        prepared=prepared,
        prefix_read_envelope=prefix,
    )
    return WriterSupportArtifactFixture(facts, options, deepcopy(artifact))


def completed_prefix_support_artifact_fixture():
    return _copy_fixture(_cached_completed_prefix_support_artifact_fixture())


@lru_cache(maxsize=1)
def _cached_tetra_support_artifact_fixture():
    return _build_snapshot_artifact(tetrahedral_facts())


def tetra_support_artifact_fixture():
    return _copy_fixture(_cached_tetra_support_artifact_fixture())


@lru_cache(maxsize=1)
def _cached_directional_support_artifact_fixture():
    return _build_snapshot_artifact(directional_facts(), rooted_at_atom=2)


def directional_support_artifact_fixture():
    return _copy_fixture(_cached_directional_support_artifact_fixture())


@lru_cache(maxsize=1)
def _cached_shared_acyclic_directional_support_artifact_fixture():
    return _build_snapshot_artifact(
        shared_acyclic_directional_facts(),
        rooted_at_atom=0,
    )


def shared_acyclic_directional_support_artifact_fixture():
    return _copy_fixture(
        _cached_shared_acyclic_directional_support_artifact_fixture()
    )


def terminal_tetra_support_artifact_fixture():
    return support_artifact_fixture(
        terminal_tetra_center_facts(),
        rooted_at_atom=0,
        policy=terminal_tetra_center_policy(),
    )


def rdkit_support_artifact_verification(smiles: str):
    fixture = rdkit_support_artifact_fixture(smiles)
    from grimace._south_star1.writer_support_artifact_fact_verifier import verify_writer_support_artifact_for_facts
    return verify_writer_support_artifact_for_facts(facts=fixture.facts, runtime_options=fixture.runtime_options, artifact=fixture.artifact)
