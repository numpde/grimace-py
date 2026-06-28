"""Named constructor for the writer-shaped online decoder route.

The generic online factories still exist for legacy exhaustive runtimes.  This
module gives WRITER_SHAPED a single obvious construction target: prepared input
plus writer-shaped runtime options produce a decoder backed by the live writer
runtime, not by any legacy online VM knobs.
"""

from __future__ import annotations

from .online_continuation import OnlineDecoderExecutionMode
from .online_decoder_api import SouthStarOnlineDecoder
from .online_decisions import FrontierCompactionMode
from .policy import SerializationLanguageMode
from .prepared_runtime import SouthStarPreparedMol
from .prepared_runtime import SouthStarRuntimeOptions
from .prepared_runtime import component_root_domains_for_prepared
from .prepared_runtime import require_writer_shaped_runtime_options
from .prepared_runtime import runtime_root_atom_for_prepared


_DEFAULT_WRITER_RUNTIME_OPTIONS = SouthStarRuntimeOptions(
    serialization_language=SerializationLanguageMode.WRITER_SHAPED,
)


def make_writer_shaped_online_decoder(
    *,
    prepared: SouthStarPreparedMol,
    runtime_options: SouthStarRuntimeOptions = _DEFAULT_WRITER_RUNTIME_OPTIONS,
    include_eos: bool = False,
) -> SouthStarOnlineDecoder:
    """Construct the online decoder for the live writer-shaped runtime.

    WRITER_SHAPED is intentionally prepared-only here.  Preparation supplies the
    structural molecule state; support is still enforced later by checked live
    writer frontier operations.
    """

    require_writer_shaped_runtime_options(runtime_options)
    rooted_at_atom = runtime_root_atom_for_prepared(
        runtime_options,
        prepared=prepared,
    )
    root_domains = tuple(
        atoms
        for _, atoms in component_root_domains_for_prepared(
            prepared=prepared,
            rooted_at_atom=rooted_at_atom,
        )
    )

    # These legacy facade fields are deliberately fixed: the writer runtime is
    # already the bounded live state engine, and exposes one checked choice per
    # emitted token text.  Cached/residual continuation modes do not apply.
    return SouthStarOnlineDecoder(
        prepared=prepared,
        facts=prepared.facts,
        policy=prepared.policy,
        semantics=prepared.semantics,
        templates=prepared.stereo_template_bundle(),
        rooted_at_atom=rooted_at_atom,
        graph_index=prepared.graph_index,
        component_root_domains=root_domains,
        runtime_options=runtime_options,
        branch_mode="determinized",
        compaction_mode=FrontierCompactionMode.TRAVERSAL_ONLY,
        include_eos=include_eos,
        execution_mode=OnlineDecoderExecutionMode.PREFIX_REPLAY,
    )


__all__ = ("make_writer_shaped_online_decoder",)
