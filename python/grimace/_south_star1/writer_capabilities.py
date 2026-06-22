"""Writer execution capability vocabulary."""

from __future__ import annotations

from enum import Enum


class _WriterExecutionCapabilityKind(Enum):
    TREE_CHILD_ENTRY = "tree_child_entry"
    CYCLIC_TREE_ENTRY = "cyclic_tree_entry"
    TREE_BOND_SLOT = "tree_bond_slot"
    VISIBLE_TREE_BOND_TEXT = "visible_tree_bond_text"

    CLOSURE_ENDPOINT_OPEN = "closure_endpoint_open"
    CLOSURE_ENDPOINT_PAIR = "closure_endpoint_pair"
    CONCURRENT_CLOSURE_ENDPOINT_OPEN = "concurrent_closure_endpoint_open"
    VISIBLE_CLOSURE_BOND_TEXT = "visible_closure_bond_text"

    TETRA_TOKEN_RESTRICTION = "tetra_token_restriction"
    TETRA_LOCAL_ORDER_RESTRICTION = "tetra_local_order_restriction"
    TETRA_RING_ENDPOINT_ORDER_OCCURRENCE = (
        "tetra_ring_endpoint_order_occurrence"
    )

    DIRECTIONAL_CARRIER_RESTRICTION = "directional_carrier_restriction"
    DIRECTIONAL_SITE_COMPATIBILITY = "directional_site_compatibility"
    DIRECTIONAL_RING_PAIR_COMPATIBILITY = (
        "directional_ring_pair_compatibility"
    )

    RESIDUAL_PROPAGATION = "residual_propagation"
    RESIDUAL_FACTOR_DISCHARGE = "residual_factor_discharge"


_PUBLIC_SUPPORTED_WRITER_EXECUTION_CAPABILITIES = frozenset(
    {
        _WriterExecutionCapabilityKind.TREE_CHILD_ENTRY,
        _WriterExecutionCapabilityKind.CYCLIC_TREE_ENTRY,
        _WriterExecutionCapabilityKind.TREE_BOND_SLOT,
        _WriterExecutionCapabilityKind.VISIBLE_TREE_BOND_TEXT,
        _WriterExecutionCapabilityKind.CLOSURE_ENDPOINT_OPEN,
        _WriterExecutionCapabilityKind.CLOSURE_ENDPOINT_PAIR,
        _WriterExecutionCapabilityKind.CONCURRENT_CLOSURE_ENDPOINT_OPEN,
        _WriterExecutionCapabilityKind.VISIBLE_CLOSURE_BOND_TEXT,
        _WriterExecutionCapabilityKind.TETRA_TOKEN_RESTRICTION,
        _WriterExecutionCapabilityKind.TETRA_LOCAL_ORDER_RESTRICTION,
        _WriterExecutionCapabilityKind.TETRA_RING_ENDPOINT_ORDER_OCCURRENCE,
        _WriterExecutionCapabilityKind.DIRECTIONAL_CARRIER_RESTRICTION,
        _WriterExecutionCapabilityKind.DIRECTIONAL_SITE_COMPATIBILITY,
        _WriterExecutionCapabilityKind.RESIDUAL_PROPAGATION,
        _WriterExecutionCapabilityKind.RESIDUAL_FACTOR_DISCHARGE,
    }
)


__all__ = (
    "_PUBLIC_SUPPORTED_WRITER_EXECUTION_CAPABILITIES",
    "_WriterExecutionCapabilityKind",
)
