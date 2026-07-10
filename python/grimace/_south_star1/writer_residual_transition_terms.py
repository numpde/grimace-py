"""Typed residual transition terms for offline writer artifact replay."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .ids import AtomId
from .ids import OccurrenceId
from .ids import SiteId
from .policy import TetraToken
from .residual_constraints import ResidualFactorKey
from .residual_constraints import ResidualPropagationResult
from .residual_constraints import ResidualStoreValueSnapshot
from .residual_constraints import TetraLocalParity
from .residual_constraints import VarId


class WriterResidualTransitionKind(Enum):
    TETRA_ATOM_TOKEN_RESTRICTION = "tetra_atom_token_restriction"
    TETRA_LOCAL_ORDER_FACTOR_CLOSURE = "tetra_local_order_factor_closure"


@dataclass(frozen=True, slots=True)
class TetraAtomTokenRestrictionTransitionTerm:
    kind: WriterResidualTransitionKind
    source_snapshot: ResidualStoreValueSnapshot
    source_snapshot_digest: str
    atom: AtomId
    site: SiteId
    token: TetraToken
    constraint_var: VarId
    constraint_value: TetraToken
    affected_variables: tuple[VarId, ...]
    affected_factor_keys: tuple[ResidualFactorKey, ...]
    propagation_result: ResidualPropagationResult
    projected_variables: tuple[VarId, ...]
    discharged_factor_keys: tuple[ResidualFactorKey, ...]
    successor_snapshot: ResidualStoreValueSnapshot
    successor_snapshot_digest: str


@dataclass(frozen=True, slots=True)
class TetraLocalOrderFactorClosureTransitionTerm:
    kind: WriterResidualTransitionKind
    source_snapshot: ResidualStoreValueSnapshot
    source_snapshot_digest: str
    atom: AtomId
    site: SiteId
    local_order: tuple[OccurrenceId, ...]
    reference_order: tuple[OccurrenceId, ...]
    target_parity: TetraLocalParity
    constraint_var: VarId
    constraint_value: TetraLocalParity
    affected_variables: tuple[VarId, ...]
    affected_factor_keys: tuple[ResidualFactorKey, ...]
    propagation_result: ResidualPropagationResult
    projected_variables: tuple[VarId, ...]
    discharged_factor_keys: tuple[ResidualFactorKey, ...]
    successor_snapshot: ResidualStoreValueSnapshot
    successor_snapshot_digest: str

