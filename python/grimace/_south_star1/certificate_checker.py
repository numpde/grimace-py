"""Standalone replay entry points for South Star certificates."""

from __future__ import annotations

from .certificates import WitnessCertificate
from .certificates import validate_witness_certificate
from .constraints import TraversalAssignment


def replay_witness_certificate(
    *,
    facts,
    skeleton,
    slots,
    assignment: TraversalAssignment,
    policy,
    semantics,
    certificate: WitnessCertificate,
) -> None:
    """Replay a witness certificate against the finite South Star model."""

    validate_witness_certificate(
        facts=facts,
        skeleton=skeleton,
        slots=slots,
        assignment=assignment,
        policy=policy,
        semantics=semantics,
        certificate=certificate,
    )


__all__ = ("replay_witness_certificate",)
