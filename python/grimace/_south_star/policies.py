from __future__ import annotations

from dataclasses import dataclass

from grimace._south_star.annotation_policy import AnnotationPolicy
from grimace._south_star.annotation_policy import MaximalEligibleCarrierAnnotationPolicy
from grimace._south_star.fragments import AllFragmentOrderPolicy
from grimace._south_star.output_order import FirstOccurrenceOutputOrderPolicy


@dataclass(frozen=True, slots=True)
class SouthStarPolicySet:
    annotation_policy: AnnotationPolicy
    fragment_order_policy: AllFragmentOrderPolicy
    output_order_policy: FirstOccurrenceOutputOrderPolicy


DEFAULT_SOUTH_STAR_POLICY_SET = SouthStarPolicySet(
    annotation_policy=MaximalEligibleCarrierAnnotationPolicy(),
    fragment_order_policy=AllFragmentOrderPolicy(),
    output_order_policy=FirstOccurrenceOutputOrderPolicy(),
)
