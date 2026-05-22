"""Public runtime API for the Rust-backed SMILES next-token engine.

The public signatures keep RDKit-like defaults for surface compatibility, but
the supported runtime today is intentionally narrower: pass `canonical=False`
and `doRandom=True` explicitly.
"""

from __future__ import annotations

import importlib
from collections.abc import Iterator, Sequence
from typing import Any

import grimace._mol_to_smiles_options as _options
from grimace._prepared_mol import PreparedMol, PrepareMol

try:
    _RUNTIME = importlib.import_module("grimace._runtime")
except ImportError as exc:  # pragma: no cover - exercised only in broken installs
    _RUNTIME = None
    _CORE_IMPORT_ERROR = exc
else:  # pragma: no cover - exercised in environments with the extension available
    _CORE_IMPORT_ERROR = None
    _DEVIATION = importlib.import_module("grimace._deviation")


def _require_runtime() -> Any:
    if _RUNTIME is None:
        raise ImportError(
            "grimace requires the compiled Rust extension "
            "'grimace._core'. Build or install the package with the "
            "extension enabled."
        ) from _CORE_IMPORT_ERROR
    return _RUNTIME


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

    runtime = _require_runtime()
    option_values = locals()
    return runtime.mol_to_smiles_enum(
        mol,
        **_options.mol_to_smiles_internal_kwargs_from_public_values(
            _options.MOL_TO_SMILES_OPTIONS,
            option_values,
        ),
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

    runtime = _require_runtime()
    option_values = locals()
    return runtime.mol_to_smiles_token_inventory(
        mol,
        **_options.mol_to_smiles_internal_kwargs_from_public_values(
            _options.MOL_TO_SMILES_OPTIONS,
            option_values,
        ),
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

    runtime = _require_runtime()
    option_values = locals()
    return runtime.mol_to_smiles_token_inventory_superset(
        mol,
        **_options.mol_to_smiles_internal_kwargs_from_public_values(
            _options.MOL_TO_SMILES_OPTIONS,
            option_values,
        ),
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

    _require_runtime()
    option_values = locals()
    return _DEVIATION.mol_to_smiles_deviation(
        mol,
        candidate,
        **_options.mol_to_smiles_internal_kwargs_from_public_values(
            _options.MOL_TO_SMILES_OPTIONS,
            option_values,
        ),
    )


if _RUNTIME is not None:
    MolToSmilesChoice = _RUNTIME.MolToSmilesChoice
    SmilesDeviation = _DEVIATION.SmilesDeviation

    class _PublicDecoderBase(_RUNTIME._PublicDecoderBase):
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
            option_values = locals()
            super().__init__(
                mol,
                **_options.mol_to_smiles_internal_kwargs_from_public_values(
                    _options.MOL_TO_SMILES_OPTIONS,
                    option_values,
                ),
            )

        @property
        def _impl(self) -> "_PublicDecoderBase":
            return self


    class MolToSmilesDecoder(_RUNTIME.MolToSmilesDecoder, _PublicDecoderBase):
        """Branch-preserving online decoder for the supported public runtime.

        Pass `canonical=False` and `doRandom=True` explicitly.
        """

        __slots__ = ()


    class MolToSmilesDeterminizedDecoder(_RUNTIME.MolToSmilesDeterminizedDecoder, _PublicDecoderBase):
        """Determinized online decoder for the supported public runtime.

        Pass `canonical=False` and `doRandom=True` explicitly.
        """

        __slots__ = ()

else:
    class _ImportErrorRuntimeBase:
        __slots__ = ()

        def __init__(self, *args: object, **kwargs: object) -> None:
            _require_runtime()


    class MolToSmilesChoice:  # pragma: no cover - exercised only in broken installs
        __slots__ = ()

        def __init__(self, *args: object, **kwargs: object) -> None:
            _require_runtime()


    class SmilesDeviation:  # pragma: no cover - exercised only in broken installs
        __slots__ = ()

        def __init__(self, *args: object, **kwargs: object) -> None:
            _require_runtime()


    class MolToSmilesDecoder(_ImportErrorRuntimeBase):  # pragma: no cover - broken installs only
        pass


    class MolToSmilesDeterminizedDecoder(_ImportErrorRuntimeBase):  # pragma: no cover - broken installs only
        pass


__all__ = [
    "MolToSmilesChoice",
    "MolToSmilesDecoder",
    "MolToSmilesDeterminizedDecoder",
    "MolToSmilesDeviation",
    "MolToSmilesEnum",
    "MolToSmilesTokenInventory",
    "MolToSmilesTokenInventorySuperset",
    "PreparedMol",
    "PrepareMol",
    "SmilesDeviation",
]
