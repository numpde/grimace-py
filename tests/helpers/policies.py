"""Helpers for loading and modifying internal reference policies."""

from __future__ import annotations

from copy import deepcopy

from grimace._mol_to_smiles_options import MOL_TO_SMILES_OPTIONS
from grimace._reference.policy import ReferencePolicy


def _public_option_defaults(**overrides: object) -> dict[str, object]:
    return {
        **{spec.public_name: spec.default for spec in MOL_TO_SMILES_OPTIONS},
        **overrides,
    }


_DEFAULT_POLICY_DATA = {
    "policy_name": "test_rdkit_random",
    "sampling": _public_option_defaults(
        seed=0,
        draw_budget=500,
        canonical=False,
        doRandom=True,
    ),
    "identity_check": _public_option_defaults(
        parse_with_rdkit=True,
    ),
}

_CONNECTED_NONSTEREO_POLICY_DATA = {
    **_DEFAULT_POLICY_DATA,
    "policy_name": "test_rdkit_random_connected_nonstereo",
}


def load_default_policy() -> ReferencePolicy:
    return ReferencePolicy(data=deepcopy(_DEFAULT_POLICY_DATA))


def load_connected_nonstereo_policy() -> ReferencePolicy:
    return ReferencePolicy(data=deepcopy(_CONNECTED_NONSTEREO_POLICY_DATA))


def with_sampling_override(
    policy: ReferencePolicy,
    **sampling_overrides: object,
) -> ReferencePolicy:
    data = deepcopy(policy.data)
    sampling = dict(data["sampling"])
    sampling.update(sampling_overrides)
    data["sampling"] = sampling
    return ReferencePolicy(data=data)
