"""Typed residual transition terms for offline writer artifact replay."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .ids import AtomId
from .ids import BondId
from .ids import OccurrenceId
from .ids import SiteId
from .policy import TetraToken
from .policy import DirectionMark
from .residual_constraints import DirectionalNormalizedSign
from .residual_constraints import DirectionalSiteCarrierModel
from .residual_constraints import ResidualFactorKey
from .residual_constraints import ResidualPropagationResult
from .residual_constraints import ResidualStoreValueSnapshot
from .residual_constraints import TetraLocalParity
from .residual_constraints import VarId


class WriterResidualTransitionKind(Enum):
    TETRA_ATOM_TOKEN_RESTRICTION = "tetra_atom_token_restriction"
    TETRA_LOCAL_ORDER_FACTOR_CLOSURE = "tetra_local_order_factor_closure"
    DIRECTIONAL_CARRIER_MARK_RESTRICTION = "directional_carrier_mark_restriction"
    DIRECTIONAL_RING_ENDPOINT_PROJECTION = "directional_ring_endpoint_projection"


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


@dataclass(frozen=True, slots=True)
class DirectionalCarrierMarkRestrictionTransitionTerm:
    kind: WriterResidualTransitionKind
    source_snapshot: ResidualStoreValueSnapshot
    source_snapshot_digest: str
    bond: BondId
    parent: AtomId
    child: AtomId
    direction_mark: DirectionMark
    canonical_orientation: int
    carrier_models: tuple[DirectionalSiteCarrierModel, ...]
    restrictions: tuple[tuple[VarId, DirectionalNormalizedSign], ...]
    affected_variables: tuple[VarId, ...]
    affected_factor_keys: tuple[ResidualFactorKey, ...]
    propagation_result: ResidualPropagationResult
    discharged_factor_keys: tuple[ResidualFactorKey, ...]
    projected_variables: tuple[VarId, ...]
    successor_snapshot: ResidualStoreValueSnapshot
    successor_snapshot_digest: str


@dataclass(frozen=True, slots=True)
class DirectionalRingEndpointProjectionTransitionTerm:
    kind: WriterResidualTransitionKind
    source_snapshot: ResidualStoreValueSnapshot
    source_snapshot_digest: str
    bond: BondId
    endpoint_atom: AtomId
    partner_atom: AtomId
    ring_label_value: int
    ring_label_text: str
    endpoint_text: str
    bond_text: str
    direction_mark: DirectionMark
    carrier_model: DirectionalSiteCarrierModel
    compatible_second_endpoint_choices: tuple[tuple[str, DirectionMark], ...]
    domain_intersections: tuple[
        tuple[VarId, tuple[DirectionalNormalizedSign, ...]], ...
    ]
    affected_variables: tuple[VarId, ...]
    affected_factor_keys: tuple[ResidualFactorKey, ...]
    propagation_result: ResidualPropagationResult
    projected_variables: tuple[VarId, ...]
    discharged_factor_keys: tuple[ResidualFactorKey, ...]
    successor_snapshot: ResidualStoreValueSnapshot
    successor_snapshot_digest: str
