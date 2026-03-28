"""Compatibility exports for the legacy pure-Python rooted enumerators.

Prefer the top-level ``smiles_next_token`` runtime API for production callers
and ``smiles_next_token.rdkit_reference`` for oracle/reference workflows.
This module remains for compatibility with older imports and parity tests.
"""

from smiles_next_token.reference.rooted.connected_nonstereo import (
    enumerate_rooted_connected_nonstereo_smiles_support,
    validate_rooted_connected_nonstereo_smiles_support,
)
from smiles_next_token.reference.rooted.connected_stereo import (
    enumerate_rooted_connected_stereo_smiles_support,
    validate_rooted_connected_stereo_smiles_support,
)


enumerate_rooted_nonstereo_smiles_support = (
    enumerate_rooted_connected_nonstereo_smiles_support
)
validate_rooted_nonstereo_smiles_support = (
    validate_rooted_connected_nonstereo_smiles_support
)
enumerate_rooted_smiles_support = enumerate_rooted_connected_nonstereo_smiles_support
validate_rooted_smiles_support = validate_rooted_connected_nonstereo_smiles_support


__all__ = [
    "enumerate_rooted_connected_nonstereo_smiles_support",
    "enumerate_rooted_connected_stereo_smiles_support",
    "validate_rooted_connected_nonstereo_smiles_support",
    "validate_rooted_connected_stereo_smiles_support",
    "enumerate_rooted_nonstereo_smiles_support",
    "validate_rooted_nonstereo_smiles_support",
    "enumerate_rooted_smiles_support",
    "validate_rooted_smiles_support",
]
