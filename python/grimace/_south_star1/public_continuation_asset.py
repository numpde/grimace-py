"""Public composition boundary for certified continuation assets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import cast

from .errors import SouthStarError
from .errors import SouthStarErrorKind
from .policy import SerializationLanguageMode
from .prepared_runtime import SouthStarRuntimeOptions
from .prepared_runtime import SouthStarWriterSurface
from .prepared_runtime import prepare_south_star_mol_from_rdkit
from .prepared_runtime import runtime_root_atom_for_prepared
from .rdkit_adapter import RdkitOrdinaryExtractionOptions
from .ordinary_stereo_sites import OrdinaryStereoSiteOptions
from .rdkit_adapter import rdkit_molecule_has_specified_stereo
from .rdkit_adapter import require_rdkit_molecule
from .writer_continuation_asset import open_writer_continuation_core
from .writer_continuation_asset import write_writer_continuation_asset
from .writer_continuation_asset import verify_writer_continuation_asset_consistency
from .writer_continuation_asset import verify_writer_continuation_asset_for_prepared
from .writer_continuation_asset import writer_continuation_asset_runtime_options
from .writer_envelope_terms import _identity_envelope
from .writer_prepared_identity import writer_prepared_identity
from .writer_snapshot import capture_initial_writer_frontier_snapshot


@dataclass(frozen=True, slots=True)
class MolToSmilesContinuationAssetVerification:
    """Ephemeral proof report for one molecule-bound continuation asset."""

    accepted: bool
    manifest_digest: str
    raw_cursor_count: int
    edge_locator_count: int
    branch_locator_count: int
    branch_proof_count: int
    terminal_record_count: int
    terminal_locator_count: int
    terminal_proof_count: int
    live_replay_complete: bool
    semantically_replayed_operations: tuple[str, ...]
    checked_relation_families: tuple[str, ...]
    checked_obligation_families: tuple[str, ...]
    unchecked_obligation_families: tuple[str, ...]


def verify_mol_to_smiles_continuation_asset(
    mol: object,
    path: str | Path,
    *,
    expected_manifest_digest: str | None = None,
) -> MolToSmilesContinuationAssetVerification:
    """Recertify one transported asset against the exact supplied RDKit molecule."""

    asset = open_writer_continuation_core(path)
    if (
        expected_manifest_digest is not None
        and asset.manifest_digest != expected_manifest_digest
    ):
        raise SouthStarError(
            SouthStarErrorKind.SEMANTIC_MISMATCH,
            "continuation_asset_manifest_digest_mismatch",
        )

    structural = verify_writer_continuation_asset_consistency(asset.path)
    if not structural.accepted:
        raise SouthStarError(
            SouthStarErrorKind.INTERNAL_INVARIANT,
            structural.reason or "continuation_asset_structural_rejection",
        )

    runtime_options = writer_continuation_asset_runtime_options(asset)
    prepared = prepare_public_continuation_molecule(
        mol,
        writer_surface=SouthStarWriterSurface(),
        runtime_options=runtime_options,
    )
    expected_identity = _identity_envelope(
        writer_prepared_identity(prepared, runtime_options)
    )
    if expected_identity != asset.manifest["prepared_identity"]:
        raise SouthStarError(
            SouthStarErrorKind.SEMANTIC_MISMATCH,
            "continuation_asset_prepared_identity_mismatch",
        )

    semantic = verify_writer_continuation_asset_for_prepared(
        prepared=prepared,
        asset=asset,
    )
    if not semantic.accepted:
        raise SouthStarError(
            SouthStarErrorKind.SEMANTIC_MISMATCH,
            semantic.reason or "continuation_asset_semantic_rejection",
        )
    return MolToSmilesContinuationAssetVerification(
        accepted=True,
        manifest_digest=asset.manifest_digest,
        raw_cursor_count=semantic.raw_cursor_count,
        edge_locator_count=semantic.edge_locator_count,
        branch_locator_count=semantic.branch_locator_count,
        branch_proof_count=semantic.branch_proof_count,
        terminal_record_count=semantic.terminal_record_count,
        terminal_locator_count=semantic.terminal_locator_count,
        terminal_proof_count=semantic.terminal_proof_count,
        live_replay_complete=semantic.live_replay_complete,
        semantically_replayed_operations=semantic.semantically_replayed_operations,
        checked_relation_families=semantic.checked_relation_families,
        checked_obligation_families=semantic.checked_obligation_families,
        unchecked_obligation_families=semantic.unchecked_obligation_families,
    )


def build_mol_to_smiles_continuation_asset(
    mol: object,
    path: str | Path,
    *,
    isomeric_smiles: bool = True,
    kekule_smiles: bool = False,
    rooted_at_atom: int = -1,
    canonical: bool = False,
    all_bonds_explicit: bool = False,
    all_hs_explicit: bool = False,
    do_random: bool = True,
    ignore_atom_map_numbers: bool = False,
) -> str:
    """Build and atomically publish one semantically certified asset."""

    if not isinstance(rooted_at_atom, int) or rooted_at_atom < -1:
        raise SouthStarError(
            SouthStarErrorKind.INVALID_FACTS,
            f"invalid rootedAtAtom for certified continuation asset: {rooted_at_atom!r}",
        )

    writer_surface = SouthStarWriterSurface(
        isomeric_smiles=bool(isomeric_smiles),
        kekule_smiles=bool(kekule_smiles),
        all_bonds_explicit=bool(all_bonds_explicit),
        all_hs_explicit=bool(all_hs_explicit),
        ignore_atom_map_numbers=bool(ignore_atom_map_numbers),
    )
    runtime_options = SouthStarRuntimeOptions(
        rooted_at_atom=rooted_at_atom,
        canonical=bool(canonical),
        do_random=bool(do_random),
        serialization_language=SerializationLanguageMode.WRITER_SHAPED,
    )
    prepared = prepare_public_continuation_molecule(
        mol,
        writer_surface=writer_surface,
        runtime_options=runtime_options,
    )
    snapshot = capture_initial_writer_frontier_snapshot(
        prepared=prepared,
        runtime_options=runtime_options,
    )
    manifest = write_writer_continuation_asset(
        path=path,
        prepared=prepared,
        snapshot=snapshot,
    )
    return cast(str, manifest["digest"])


def prepare_public_continuation_molecule(
    mol: object,
    *,
    writer_surface: SouthStarWriterSurface,
    runtime_options: SouthStarRuntimeOptions,
):
    """Prepare one public RDKit molecule under the asset's fixed policy."""

    mol = require_rdkit_molecule(mol)
    has_specified_stereo = rdkit_molecule_has_specified_stereo(mol)
    if has_specified_stereo and not writer_surface.isomeric_smiles:
        raise SouthStarError(
            SouthStarErrorKind.UNSUPPORTED_POLICY,
            "isomericSmiles=False is unsupported for a molecule with specified stereo",
        )
    extraction_options = (
        RdkitOrdinaryExtractionOptions(
            include_potential_sites=True,
            stereo_site_discovery_mode="specified_closure",
            stereo_site_options=OrdinaryStereoSiteOptions(
                ligand_equivalence="exact_stereochemical_graph_automorphism",
            ),
        )
        if has_specified_stereo
        else RdkitOrdinaryExtractionOptions(
            include_potential_sites=False,
            extract_specified_tetrahedral=False,
            extract_specified_directional=False,
        )
    )
    prepared = prepare_south_star_mol_from_rdkit(
        mol,
        writer_surface=writer_surface,
        extraction_options=extraction_options,
    )
    # Root and runtime-mode validation belong to the same preparation boundary
    # for both publication and later molecule-bound proof sessions.
    runtime_root_atom_for_prepared(runtime_options, prepared=prepared)
    return prepared


__all__ = (
    "build_mol_to_smiles_continuation_asset",
    "MolToSmilesContinuationAssetVerification",
    "prepare_public_continuation_molecule",
    "verify_mol_to_smiles_continuation_asset",
)
