"""Public runtime API for the Rust-backed SMILES next-token engine.

The public signatures keep RDKit-like defaults for surface compatibility, but
the supported runtime today is intentionally narrower: pass `canonical=False`
and `doRandom=True` explicitly.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence

import grimace._deviation as _deviation
import grimace._mol_to_smiles_options as _options
import grimace._runtime as _runtime
import grimace._sampling as _sampling
from grimace._prepared_mol import PreparedMol, PrepareMol

MolToSmilesChoice = _runtime.MolToSmilesChoice
SmilesDeviation = _deviation.SmilesDeviation
SmilesSample = _sampling.SmilesSample
SmilesSampleStep = _sampling.SmilesSampleStep


def _runtime_kwargs(option_values: Mapping[str, object]) -> dict[str, object]:
    return _options.coerce_public_options(
        _options.MOL_TO_SMILES_OPTIONS,
        option_values,
        context="MolToSmiles",
    )


def MolToSmilesEnum(
    mol: object,
    *,
    isomericSmiles: bool = True,
    kekuleSmiles: bool = False,
    rootedAtAtom: int = -1,
    canonical: bool = True,
    allBondsExplicit: bool = False,
    allHsExplicit: bool = False,
    doRandom: bool = False,
    ignoreAtomMapNumbers: bool = False,
) -> Iterator[str]:
    """Yield the exact rooted SMILES support for a molecule.

    Pass `canonical=False` and `doRandom=True` explicitly. The RDKit-like
    default signature is preserved for surface compatibility, not because the
    defaults are currently implemented.
    """

    return _runtime.mol_to_smiles_enum(
        mol,
        **_runtime_kwargs(locals()),
    )


def MolToSmilesTokenInventory(
    mol: object,
    *,
    isomericSmiles: bool = True,
    kekuleSmiles: bool = False,
    rootedAtAtom: int = -1,
    canonical: bool = True,
    allBondsExplicit: bool = False,
    allHsExplicit: bool = False,
    doRandom: bool = False,
    ignoreAtomMapNumbers: bool = False,
) -> tuple[str, ...]:
    """Return the token inventory for a molecule under the public runtime flags.

    Pass `canonical=False` and `doRandom=True` explicitly. The RDKit-like
    default signature is preserved for surface compatibility, not because the
    defaults are currently implemented.
    """

    return _runtime.mol_to_smiles_token_inventory(
        mol,
        **_runtime_kwargs(locals()),
    )


def MolToSmilesTokenInventorySuperset(
    mol: object,
    *,
    isomericSmiles: bool = True,
    kekuleSmiles: bool = False,
    rootedAtAtom: int = -1,
    canonical: bool = True,
    allBondsExplicit: bool = False,
    allHsExplicit: bool = False,
    doRandom: bool = False,
    ignoreAtomMapNumbers: bool = False,
) -> tuple[str, ...]:
    """Return a conservative token inventory superset without decoder walking.

    Pass `canonical=False` and `doRandom=True` explicitly. The RDKit-like
    default signature is preserved for surface compatibility, not because the
    defaults are currently implemented.
    """

    return _runtime.mol_to_smiles_token_inventory_superset(
        mol,
        **_runtime_kwargs(locals()),
    )


def MolToSmilesDeviation(
    mol: object,
    candidate: str | Sequence[str],
    *,
    isomericSmiles: bool = True,
    kekuleSmiles: bool = False,
    rootedAtAtom: int = -1,
    canonical: bool = True,
    allBondsExplicit: bool = False,
    allHsExplicit: bool = False,
    doRandom: bool = False,
    ignoreAtomMapNumbers: bool = False,
) -> SmilesDeviation | None:
    """Return the first candidate location outside the molecule's SMILES language.

    A sequence candidate is treated as atomic external tokens: each item must
    match one legal Grimace decoder token text.
    """

    return _deviation.mol_to_smiles_deviation(
        mol,
        candidate,
        **_runtime_kwargs(locals()),
    )


def MolToSmilesSample(
    mol: object,
    *,
    seed: int,
    decoder_view: str = "determinized",
    sampling_mode: str = "uniform_token",
    isomericSmiles: bool = True,
    kekuleSmiles: bool = False,
    rootedAtAtom: int = -1,
    canonical: bool = True,
    allBondsExplicit: bool = False,
    allHsExplicit: bool = False,
    doRandom: bool = False,
    ignoreAtomMapNumbers: bool = False,
) -> SmilesSample:
    """Draw one supported SMILES path and retain per-step token choices."""

    return _sampling.mol_to_smiles_sample(
        mol,
        seed=seed,
        decoder_view=decoder_view,
        sampling_mode=sampling_mode,
        **_runtime_kwargs(locals()),
    )


class _PublicDecoderBase(_runtime._PublicDecoderBase):
    __slots__ = ()

    def __init__(
        self,
        mol: object,
        *,
        isomericSmiles: bool = True,
        kekuleSmiles: bool = False,
        rootedAtAtom: int = -1,
        canonical: bool = True,
        allBondsExplicit: bool = False,
        allHsExplicit: bool = False,
        doRandom: bool = False,
        ignoreAtomMapNumbers: bool = False,
    ) -> None:
        super().__init__(
            mol,
            **_runtime_kwargs(locals()),
        )


class MolToSmilesDecoder(_runtime.MolToSmilesDecoder, _PublicDecoderBase):
    """Branch-preserving online decoder for the supported public runtime.

    Pass `canonical=False` and `doRandom=True` explicitly.
    """

    __slots__ = ()


class MolToSmilesDeterminizedDecoder(
    _runtime.MolToSmilesDeterminizedDecoder,
    _PublicDecoderBase,
):
    """Determinized online decoder for the supported public runtime.

    Pass `canonical=False` and `doRandom=True` explicitly.
    """

    __slots__ = ()


__all__ = [
    "MolToSmilesChoice",
    "MolToSmilesDecoder",
    "MolToSmilesDeterminizedDecoder",
    "MolToSmilesDeviation",
    "MolToSmilesEnum",
    "MolToSmilesSample",
    "MolToSmilesTokenInventory",
    "MolToSmilesTokenInventorySuperset",
    "PreparedMol",
    "PrepareMol",
    "SmilesDeviation",
    "SmilesSample",
    "SmilesSampleStep",
]
