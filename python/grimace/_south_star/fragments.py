from __future__ import annotations

from dataclasses import dataclass
from functools import reduce
from itertools import permutations
from itertools import product
from operator import mul


@dataclass(frozen=True, slots=True)
class SouthStarFragmentSupport:
    fragment_id: str
    outputs: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.fragment_id:
            raise ValueError("fragment id must be nonempty")
        if not self.outputs:
            raise ValueError(f"fragment {self.fragment_id!r} must define outputs")
        if tuple(dict.fromkeys(self.outputs)) != self.outputs:
            raise ValueError(f"fragment {self.fragment_id!r} has duplicate outputs")
        for output in self.outputs:
            if not output:
                raise ValueError(f"fragment {self.fragment_id!r} has empty output")
            if "." in output:
                raise ValueError(
                    f"fragment {self.fragment_id!r} output {output!r} is already "
                    "disconnected"
                )


@dataclass(frozen=True, slots=True)
class AllFragmentOrderPolicy:
    name: str = "all_fragment_orders"

    def fragment_orders(self, fragment_count: int) -> tuple[tuple[int, ...], ...]:
        if fragment_count <= 0:
            raise ValueError("fragment count must be positive")
        return tuple(permutations(range(fragment_count)))


@dataclass(frozen=True, slots=True)
class SouthStarDisconnectedCompositionResult:
    outputs: tuple[str, ...]
    fragment_count: int
    fragment_output_counts: tuple[int, ...]
    fragment_order_policy: str
    fragment_order_count: int
    estimated_product_size: int


def compose_disconnected_fragment_supports(
    fragments: tuple[SouthStarFragmentSupport, ...],
    *,
    fragment_order_policy: AllFragmentOrderPolicy | None = None,
) -> SouthStarDisconnectedCompositionResult:
    if not fragments:
        raise ValueError("disconnected composition requires at least one fragment")
    fragment_order_policy = fragment_order_policy or AllFragmentOrderPolicy()
    orders = fragment_order_policy.fragment_orders(len(fragments))
    outputs = tuple(
        dict.fromkeys(
            ".".join(output_group)
            for order in orders
            for output_group in product(
                *(fragments[fragment_idx].outputs for fragment_idx in order)
            )
        )
    )
    fragment_output_counts = tuple(len(fragment.outputs) for fragment in fragments)
    return SouthStarDisconnectedCompositionResult(
        outputs=outputs,
        fragment_count=len(fragments),
        fragment_output_counts=fragment_output_counts,
        fragment_order_policy=fragment_order_policy.name,
        fragment_order_count=len(orders),
        estimated_product_size=len(orders)
        * reduce(mul, fragment_output_counts, 1),
    )
