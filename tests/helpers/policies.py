"""Helpers for loading and modifying reference policies."""

from __future__ import annotations

from copy import deepcopy

from smiles_next_token.reference import (
    DEFAULT_RDKIT_RANDOM_CONNECTED_NONSTEREO_POLICY_PATH,
    DEFAULT_RDKIT_RANDOM_POLICY_PATH,
    ReferencePolicy,
)


def load_default_policy() -> ReferencePolicy:
    return ReferencePolicy.from_path(DEFAULT_RDKIT_RANDOM_POLICY_PATH)


def load_connected_nonstereo_policy() -> ReferencePolicy:
    return ReferencePolicy.from_path(DEFAULT_RDKIT_RANDOM_CONNECTED_NONSTEREO_POLICY_PATH)


def with_sampling_override(
    policy: ReferencePolicy,
    **sampling_overrides: object,
) -> ReferencePolicy:
    data = deepcopy(policy.data)
    sampling = dict(data["sampling"])
    sampling.update(sampling_overrides)
    data["sampling"] = sampling
    return ReferencePolicy(data=data)
