"""Public composition boundary for certified continuation assets."""

from __future__ import annotations

from pathlib import Path
from typing import cast

from .errors import SouthStarError
from .errors import SouthStarErrorKind
from .policy import SerializationLanguageMode
from .prepared_runtime import SouthStarRuntimeOptions
from .prepared_runtime import SouthStarWriterSurface
from .prepared_runtime import prepare_south_star_mol_from_rdkit
from .rdkit_adapter import RdkitOrdinaryExtractionOptions
from .rdkit_adapter import rdkit_molecule_has_specified_stereo
from .rdkit_adapter import require_rdkit_molecule
from .writer_continuation_asset import write_writer_continuation_asset
from .writer_snapshot import capture_initial_writer_frontier_snapshot


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

    mol = require_rdkit_molecule(mol)
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


__all__ = ("build_mol_to_smiles_continuation_asset",)
