"""Internal runtime bridge between RDKit input and the Rust core."""

from __future__ import annotations

import importlib
from collections.abc import Hashable, Iterator, Sequence
from dataclasses import dataclass, replace
from itertools import product
from numbers import Integral
from typing import Protocol, TypeAlias, cast

import grimace._prepared_mol as _prepared_mol_module

_core = importlib.import_module("grimace._core")
from grimace._prepared_mol import (
    PreparedMol,
    _prepared_mol_fragments,
    _prepared_mol_writer_flag_values,
)
from grimace._reference.prepared_graph import (
    CONNECTED_NONSTEREO_SURFACE,
    CONNECTED_STEREO_SURFACE,
    PREPARED_SMILES_GRAPH_SCHEMA_VERSION,
    PreparedSmilesGraph as ReferencePreparedSmilesGraph,
    prepare_smiles_graph_from_mol_to_smiles_kwargs,
    prepared_stereo_atom_token,
    ring_label_text,
)

DecoderCacheKey: TypeAlias = tuple[object, ...]


class _BaseDecoderState(Protocol):
    def prefix(self) -> str: ...
    def is_terminal(self) -> bool: ...
    def copy(self) -> object: ...
    def cache_key(self) -> Hashable: ...


class _CoreDecoderState(_BaseDecoderState, Protocol):
    def choice_successors(self) -> Sequence[tuple[str, object]]: ...
    def grouped_successors(self) -> Sequence[tuple[str, object]]: ...


class _AdapterDecoderState(_BaseDecoderState, Protocol):
    def choice_successor_states(self) -> tuple[tuple[str, object], ...]: ...
    def grouped_successor_states(self) -> tuple[tuple[str, object], ...]: ...


@dataclass(frozen=True, slots=True)
class MolToSmilesFlags:
    isomeric_smiles: bool = True
    kekule_smiles: bool = False
    rooted_at_atom: int = -1
    canonical: bool = True
    all_bonds_explicit: bool = False
    all_hs_explicit: bool = False
    do_random: bool = False
    ignore_atom_map_numbers: bool = False

    def with_rooted_at_atom(self, rooted_at_atom: int) -> "MolToSmilesFlags":
        return replace(self, rooted_at_atom=rooted_at_atom)


def _prepared_bond_dirs(prepared: object) -> tuple[str, ...]:
    return tuple(str(value) for value in getattr(prepared, "bond_dirs", ()))


def _requires_stereo_runtime_surface(
    mol_or_prepared: object,
    *,
    flags: MolToSmilesFlags,
) -> bool:
    if flags.isomeric_smiles:
        return True
    if not flags.all_bonds_explicit:
        return False
    if _prepared_mol_module._is_rdkit_mol(mol_or_prepared):
        return _prepared_mol_module._rdkit_mol_requires_stereo_surface(mol_or_prepared)
    if getattr(mol_or_prepared, "surface_kind", None) != CONNECTED_STEREO_SURFACE:
        return False
    return any(bond_dir != "NONE" for bond_dir in _prepared_bond_dirs(mol_or_prepared))


def _runtime_surface_kind(
    mol_or_prepared: object,
    *,
    flags: MolToSmilesFlags,
) -> str:
    # The runtime is mostly keyed by isomeric_smiles, but explicit single-bond
    # directions still require the stereo surface even when isomeric_smiles=False.
    if _requires_stereo_runtime_surface(mol_or_prepared, flags=flags):
        return CONNECTED_STEREO_SURFACE
    return CONNECTED_NONSTEREO_SURFACE


@dataclass(frozen=True, slots=True)
class _FragmentPlan:
    fragment: object
    rooted_at_atom: int | None


def _validate_surface_kind(
    prepared: object,
    *,
    surface_kind: str,
) -> None:
    if prepared.surface_kind != surface_kind:
        raise ValueError(
            f"PreparedSmilesGraph surface_kind={prepared.surface_kind!r} does not match "
            f"the requested surface_kind={surface_kind!r}"
        )


def _validate_writer_flags(
    prepared: object,
    flags: MolToSmilesFlags,
) -> None:
    if isinstance(prepared, _core.PreparedSmilesGraph):
        prepared_data = prepared.to_dict()
        actual = (
            bool(prepared_data["writer_do_isomeric_smiles"]),
            bool(prepared_data["writer_kekule_smiles"]),
            bool(prepared_data["writer_all_bonds_explicit"]),
            bool(prepared_data["writer_all_hs_explicit"]),
            bool(prepared_data["writer_ignore_atom_map_numbers"]),
        )
    else:
        actual = (
            prepared.writer_do_isomeric_smiles,
            prepared.writer_kekule_smiles,
            prepared.writer_all_bonds_explicit,
            prepared.writer_all_hs_explicit,
            prepared.writer_ignore_atom_map_numbers,
        )
    expected = (
        bool(flags.isomeric_smiles),
        bool(flags.kekule_smiles),
        bool(flags.all_bonds_explicit),
        bool(flags.all_hs_explicit),
        bool(flags.ignore_atom_map_numbers),
    )
    if actual != expected:
        raise ValueError(
            "PreparedSmilesGraph writer flags do not match the requested public runtime options"
        )


def _validate_prepared_mol_writer_flags(
    prepared: PreparedMol,
    flags: MolToSmilesFlags,
) -> None:
    actual = _prepared_mol_writer_flag_values(prepared)
    expected = (
        bool(flags.isomeric_smiles),
        bool(flags.kekule_smiles),
        bool(flags.all_bonds_explicit),
        bool(flags.all_hs_explicit),
        bool(flags.ignore_atom_map_numbers),
    )
    if actual != expected:
        raise ValueError(
            "PreparedMol writer flags do not match the requested public runtime options"
        )


def _prepare_runtime_input(
    mol_or_prepared: object,
    *,
    flags: MolToSmilesFlags,
) -> object:
    if isinstance(mol_or_prepared, PreparedMol):
        _validate_prepared_mol_writer_flags(mol_or_prepared, flags)
        return mol_or_prepared
    if _prepared_mol_module._is_rdkit_mol(mol_or_prepared):
        return _prepared_mol_module.PrepareMol(
            mol_or_prepared,
            isomericSmiles=flags.isomeric_smiles,
            kekuleSmiles=flags.kekule_smiles,
            allBondsExplicit=flags.all_bonds_explicit,
            allHsExplicit=flags.all_hs_explicit,
            ignoreAtomMapNumbers=flags.ignore_atom_map_numbers,
        )
    return mol_or_prepared


def _validate_supported_flags(flags: MolToSmilesFlags) -> None:
    for name, value in (
        ("isomericSmiles", flags.isomeric_smiles),
        ("kekuleSmiles", flags.kekule_smiles),
        ("canonical", flags.canonical),
        ("allBondsExplicit", flags.all_bonds_explicit),
        ("allHsExplicit", flags.all_hs_explicit),
        ("doRandom", flags.do_random),
        ("ignoreAtomMapNumbers", flags.ignore_atom_map_numbers),
    ):
        if value is not None and not isinstance(value, Integral):
            raise NotImplementedError(
                f"MolToSmiles runtime requires {name} to follow RDKit's Python binding "
                "and be a bool, int, or None"
            )
    if not isinstance(flags.rooted_at_atom, Integral):
        raise NotImplementedError(
            "MolToSmiles runtime requires rootedAtAtom to follow RDKit's Python binding "
            "and be an integer"
        )
    if bool(flags.canonical):
        raise NotImplementedError(
            "MolToSmiles runtime currently supports only canonical=False and "
            "doRandom=True; the public signatures keep RDKit-like defaults for "
            "surface compatibility, so pass those two flags explicitly."
        )
    if not bool(flags.do_random):
        raise NotImplementedError(
            "MolToSmiles runtime currently supports only canonical=False and "
            "doRandom=True; the public signatures keep RDKit-like defaults for "
            "surface compatibility, so pass those two flags explicitly."
        )


def _ensure_singly_connected_molecule(mol: object) -> None:
    if mol.GetNumAtoms() == 0:
        return
    if _prepared_mol_module._rdkit_mol_fragment_count(mol) != 1:
        raise NotImplementedError(
            "MolToSmiles runtime currently supports only singly-connected molecules"
        )


def _as_disconnected_prepared_mol(mol_or_prepared: object) -> PreparedMol | None:
    if isinstance(mol_or_prepared, PreparedMol):
        if len(_prepared_mol_fragments(mol_or_prepared)) > 1:
            return mol_or_prepared
    return None


def _fragment_plans_for_prepared_mol(
    prepared: PreparedMol,
    *,
    rooted_at_atom: int | None,
) -> tuple[_FragmentPlan, ...]:
    fragments = _prepared_mol_fragments(prepared)
    if rooted_at_atom is None:
        return tuple(
            _FragmentPlan(fragment.prepared_graph, None)
            for fragment in fragments
        )
    if len(fragments) == 1 and len(fragments[0].atom_indices) == 0:
        if rooted_at_atom == 0:
            return (_FragmentPlan(fragments[0].prepared_graph, 0),)
        raise IndexError("root_idx out of range")

    global_to_local: dict[int, tuple[int, int]] = {}
    for fragment_idx, fragment in enumerate(fragments):
        for local_idx, global_idx in enumerate(fragment.atom_indices):
            global_to_local[global_idx] = (fragment_idx, local_idx)

    if rooted_at_atom not in global_to_local:
        raise IndexError("root_idx out of range")

    rooted_fragment_idx, rooted_local_idx = global_to_local[rooted_at_atom]
    plans: list[_FragmentPlan] = []
    for fragment_idx, fragment in enumerate(fragments):
        if fragment_idx == rooted_fragment_idx:
            plans.append(_FragmentPlan(fragment.prepared_graph, rooted_local_idx))
        else:
            plans.append(_FragmentPlan(fragment.prepared_graph, None))
    return tuple(plans)


def _connected_prepared_mol_plan(
    prepared: PreparedMol,
    *,
    rooted_at_atom: int,
) -> _FragmentPlan:
    rooted_at_atom_or_none = None if rooted_at_atom < 0 else rooted_at_atom
    plans = _fragment_plans_for_prepared_mol(
        prepared,
        rooted_at_atom=rooted_at_atom_or_none,
    )
    if len(plans) != 1:
        raise NotImplementedError(
            "connected PreparedMol runtime requires one prepared fragment"
        )
    plan = plans[0]
    if plan.rooted_at_atom is None:
        return _FragmentPlan(plan.fragment, -1)
    return plan


def _atom_count(mol_or_prepared: object) -> int:
    if _prepared_mol_module._is_rdkit_mol(mol_or_prepared):
        return mol_or_prepared.GetNumAtoms()
    if isinstance(mol_or_prepared, PreparedMol):
        return sum(
            len(fragment.atom_indices)
            for fragment in _prepared_mol_fragments(mol_or_prepared)
        )
    if isinstance(mol_or_prepared, ReferencePreparedSmilesGraph):
        return mol_or_prepared.atom_count
    if isinstance(mol_or_prepared, _core.PreparedSmilesGraph):
        return cast(int, mol_or_prepared.to_dict()["atom_count"])
    raise TypeError(f"Unsupported molecule/prepared type: {type(mol_or_prepared)!r}")


def _connected_fragment_support(
    fragment_mol: object,
    *,
    flags: MolToSmilesFlags,
    rooted_at_atom: int | None,
) -> set[str]:
    if rooted_at_atom is not None:
        return mol_to_smiles_support(
            fragment_mol,
            isomeric_smiles=flags.isomeric_smiles,
            kekule_smiles=flags.kekule_smiles,
            rooted_at_atom=rooted_at_atom,
            canonical=flags.canonical,
            all_bonds_explicit=flags.all_bonds_explicit,
            all_hs_explicit=flags.all_hs_explicit,
            do_random=flags.do_random,
            ignore_atom_map_numbers=flags.ignore_atom_map_numbers,
        )

    # For all-roots support, build or unwrap the prepared graph once and reuse
    # it across roots instead of reparsing the same fragment each time.
    atom_count = _atom_count(fragment_mol)
    if _prepared_mol_module._is_rdkit_mol(fragment_mol):
        prepared_or_fragment = prepare_smiles_graph(fragment_mol, flags=flags)
    elif isinstance(fragment_mol, PreparedMol):
        prepared_or_fragment = _connected_prepared_mol_plan(
            fragment_mol,
            rooted_at_atom=-1,
        ).fragment
    else:
        prepared_or_fragment = fragment_mol

    support: set[str] = set()
    local_root_indices = (0,) if atom_count == 0 else range(atom_count)
    for local_root_idx in local_root_indices:
        support.update(
            mol_to_smiles_support(
                prepared_or_fragment,
                isomeric_smiles=flags.isomeric_smiles,
                kekule_smiles=flags.kekule_smiles,
                rooted_at_atom=local_root_idx,
                canonical=flags.canonical,
                all_bonds_explicit=flags.all_bonds_explicit,
                all_hs_explicit=flags.all_hs_explicit,
                do_random=flags.do_random,
                ignore_atom_map_numbers=flags.ignore_atom_map_numbers,
            )
        )
    return support


def _fragmented_prepared_mol_to_smiles_support(
    prepared: PreparedMol,
    *,
    flags: MolToSmilesFlags,
) -> set[str]:
    fragment_supports: list[tuple[str, ...]] = []
    rooted_at_atom = None if flags.rooted_at_atom < 0 else flags.rooted_at_atom

    for plan in _fragment_plans_for_prepared_mol(
        prepared,
        rooted_at_atom=rooted_at_atom,
    ):
        support = _connected_fragment_support(
            plan.fragment,
            flags=flags,
            rooted_at_atom=plan.rooted_at_atom,
        )
        fragment_supports.append(tuple(sorted(support)))

    return {".".join(parts) for parts in product(*fragment_supports)}


class MolToSmilesChoice:
    __slots__ = ("text", "next_state")

    def __init__(self, text: str, next_state: object) -> None:
        self.text = text
        self.next_state = next_state


def _new_public_decoder(decoder_type: type, state_impl: object) -> object:
    decoder = decoder_type.__new__(decoder_type)
    decoder._state = state_impl
    decoder._choices_cache = None
    return decoder


class _CoreStateAdapter:
    __slots__ = ("_decoder",)

    def __init__(self, decoder: object) -> None:
        self._decoder = decoder

    def choice_successor_states(self) -> tuple[tuple[str, object], ...]:
        successors = self._decoder.choice_successors()
        if not successors:
            return ()
        successor_states = [None] * len(successors)
        for idx, (text, next_decoder) in enumerate(successors):
            next_state = _CoreStateAdapter.__new__(_CoreStateAdapter)
            next_state._decoder = next_decoder
            successor_states[idx] = (text, next_state)
        return tuple(successor_states)

    def choices(self) -> tuple[MolToSmilesChoice, ...]:
        choices: list[MolToSmilesChoice] = []
        for text, next_state in self.choice_successor_states():
            choices.append(MolToSmilesChoice(text=text, next_state=next_state))
        return tuple(choices)

    def prefix(self) -> str:
        return self._decoder.prefix()

    def is_terminal(self) -> bool:
        return bool(self._decoder.is_terminal())

    def copy(self) -> "_CoreStateAdapter":
        return type(self)(self._decoder.copy())

    def cache_key(self) -> DecoderCacheKey:
        return ("core", self._decoder.cache_key())

    def grouped_successor_states(self) -> tuple[tuple[str, object], ...]:
        successors = self._decoder.grouped_successors()
        if not successors:
            return ()
        successor_states = [None] * len(successors)
        for idx, (text, next_decoder) in enumerate(successors):
            next_state = _CoreStateAdapter.__new__(_CoreStateAdapter)
            next_state._decoder = next_decoder
            successor_states[idx] = (text, next_state)
        return tuple(successor_states)

    def _advance_token(self, text: str) -> "_CoreStateAdapter":
        next_decoder = self._decoder.copy()
        next_decoder.advance_token(text)
        return type(self)(next_decoder)


class _MergedStateAdapter:
    __slots__ = ("_states",)

    def __init__(self, states: tuple[object, ...]) -> None:
        if not states:
            raise ValueError("Merged decoder state requires at least one branch")
        self._states = states

    def choice_successor_states(self) -> tuple[tuple[str, object], ...]:
        successor_states: list[tuple[str, object]] = []
        for state in self._states:
            if _state_is_terminal(state):
                continue
            successor_states.extend(_choice_successor_states(state))
        return tuple(successor_states)

    def choices(self) -> tuple[MolToSmilesChoice, ...]:
        choices: list[MolToSmilesChoice] = []
        for text, next_state in self.choice_successor_states():
            choices.append(MolToSmilesChoice(text=text, next_state=next_state))
        return tuple(choices)

    def prefix(self) -> str:
        prefix = self._states[0].prefix()
        for state in self._states[1:]:
            if state.prefix() != prefix:
                raise RuntimeError("Merged decoder states diverged on prefix")
        return prefix

    def is_terminal(self) -> bool:
        return all(_state_is_terminal(state) for state in self._states)

    def copy(self) -> "_MergedStateAdapter":
        return type(self)(tuple(state.copy() for state in self._states))

    def cache_key(self) -> DecoderCacheKey:
        return (
            "merged",
            tuple(sorted((_state_cache_key(state) for state in self._states), key=repr)),
        )

    def grouped_successor_states(self) -> tuple[tuple[str, object], ...]:
        grouped: dict[str, list[object]] = {}
        for state in self._states:
            for text, successor in _grouped_successor_states(state):
                grouped.setdefault(text, []).append(successor)
        return tuple(
            (text, _merge_choice_successor_states(tuple(successors)))
            for text, successors in grouped.items()
        )


class _DisconnectedStateAdapter:
    __slots__ = ("_fragment_states", "_fragment_idx", "_completed_prefix")

    def __init__(
        self,
        fragment_states: tuple[object, ...],
        *,
        fragment_idx: int = 0,
        completed_prefix: str = "",
    ) -> None:
        if not fragment_states:
            raise ValueError("Disconnected decoder requires at least one fragment state")
        self._fragment_states = fragment_states
        self._fragment_idx = fragment_idx
        self._completed_prefix = completed_prefix

    def _active_state(self) -> object:
        return self._fragment_states[self._fragment_idx]

    def choice_successor_states(self) -> tuple[tuple[str, object], ...]:
        active = self._active_state()
        if not _state_is_terminal(active):
            successor_states: list[tuple[str, object]] = []
            adapter_type = type(self)
            for text, successor in _choice_successor_states(active):
                successor_states.append(
                    (
                        text,
                        adapter_type(
                            self._fragment_states[: self._fragment_idx]
                            + (successor,)
                            + self._fragment_states[self._fragment_idx + 1 :],
                            fragment_idx=self._fragment_idx,
                            completed_prefix=self._completed_prefix,
                        ),
                    )
                )
            return tuple(successor_states)
        if self._fragment_idx + 1 < len(self._fragment_states):
            return (
                (
                    ".",
                    type(self)(
                        self._fragment_states,
                        fragment_idx=self._fragment_idx + 1,
                        completed_prefix=f"{self._completed_prefix}{active.prefix()}.",
                    ),
                ),
            )
        return ()

    def choices(self) -> tuple[MolToSmilesChoice, ...]:
        choices: list[MolToSmilesChoice] = []
        for text, next_state in self.choice_successor_states():
            choices.append(MolToSmilesChoice(text=text, next_state=next_state))
        return tuple(choices)

    def prefix(self) -> str:
        return f"{self._completed_prefix}{self._active_state().prefix()}"

    def is_terminal(self) -> bool:
        active = self._active_state()
        return _state_is_terminal(active) and self._fragment_idx + 1 == len(self._fragment_states)

    def copy(self) -> "_DisconnectedStateAdapter":
        return type(self)(
            tuple(state.copy() for state in self._fragment_states),
            fragment_idx=self._fragment_idx,
            completed_prefix=self._completed_prefix,
        )

    def cache_key(self) -> DecoderCacheKey:
        return (
            "disconnected",
            self._fragment_idx,
            self._completed_prefix,
            tuple(_state_cache_key(state) for state in self._fragment_states),
        )

    def grouped_successor_states(self) -> tuple[tuple[str, object], ...]:
        active = self._active_state()
        if not _state_is_terminal(active):
            return tuple(
                (
                    text,
                    type(self)(
                        self._fragment_states[: self._fragment_idx]
                        + (successor,)
                        + self._fragment_states[self._fragment_idx + 1 :],
                        fragment_idx=self._fragment_idx,
                        completed_prefix=self._completed_prefix,
                    ),
                )
                for text, successor in _grouped_successor_states(active)
            )
        if self._fragment_idx + 1 < len(self._fragment_states):
            return (
                (
                    ".",
                    type(self)(
                        self._fragment_states,
                        fragment_idx=self._fragment_idx + 1,
                        completed_prefix=f"{self._completed_prefix}{active.prefix()}.",
                    ),
                ),
            )
        return ()


def _merge_choice_successor_states(states: tuple[object, ...]) -> object:
    flattened: list[object] = []
    for state in states:
        if isinstance(state, _MergedStateAdapter):
            flattened.extend(state._states)
        else:
            flattened.append(state)
    if len(flattened) == 1:
        return flattened[0]
    return _MergedStateAdapter(tuple(flattened))


def _choice_successor_states(state: _AdapterDecoderState | _CoreDecoderState) -> tuple[tuple[str, object], ...]:
    if isinstance(state, (_CoreStateAdapter, _MergedStateAdapter, _DisconnectedStateAdapter)):
        return state.choice_successor_states()
    successors = state.choice_successors()
    if not successors:
        return ()
    return tuple(successors)


def _grouped_successor_states(state: _AdapterDecoderState | _CoreDecoderState) -> tuple[tuple[str, object], ...]:
    if isinstance(state, (_CoreStateAdapter, _MergedStateAdapter, _DisconnectedStateAdapter)):
        return state.grouped_successor_states()
    successors = state.grouped_successors()
    if not successors:
        return ()
    return tuple(successors)


def _state_is_terminal(state: _BaseDecoderState) -> bool:
    return bool(state.is_terminal())


def _determinized_choice_successors(
    state: _AdapterDecoderState | _CoreDecoderState,
) -> tuple[tuple[str, object], ...]:
    """Return one successor per token text by merging same-text branches."""
    return _grouped_successor_states(state)


def _state_cache_key(state: _BaseDecoderState) -> DecoderCacheKey:
    key = state.cache_key()
    if isinstance(key, tuple):
        return cast(DecoderCacheKey, key)
    return ("raw", key)


def _reachable_terminal_prefixes(
    state: _AdapterDecoderState | _CoreDecoderState,
    *,
    memo: dict[DecoderCacheKey, frozenset[str]] | None = None,
) -> frozenset[str]:
    """Return every terminal prefix reachable from one internal decoder state."""
    if memo is None:
        memo = {}

    key = _state_cache_key(state)
    cached = memo.get(key)
    if cached is not None:
        return cached

    if state.is_terminal():
        terminal = frozenset({state.prefix()})
        memo[key] = terminal
        return terminal

    outputs: set[str] = set()
    for _, successor in _choice_successor_states(state):
        outputs.update(_reachable_terminal_prefixes(successor, memo=memo))
    resolved = frozenset(outputs)
    memo[key] = resolved
    return resolved


def prepare_smiles_graph(
    mol_or_prepared: object,
    *,
    flags: MolToSmilesFlags,
) -> _core.PreparedSmilesGraph:
    surface_kind = _runtime_surface_kind(mol_or_prepared, flags=flags)
    if isinstance(mol_or_prepared, _core.PreparedSmilesGraph):
        _validate_surface_kind(mol_or_prepared, surface_kind=surface_kind)
        _validate_writer_flags(mol_or_prepared, flags)
        return mol_or_prepared

    if isinstance(mol_or_prepared, ReferencePreparedSmilesGraph):
        _validate_surface_kind(mol_or_prepared, surface_kind=surface_kind)
        _validate_writer_flags(mol_or_prepared, flags)
        return _core.PreparedSmilesGraph(mol_or_prepared)

    _ensure_singly_connected_molecule(mol_or_prepared)
    reference_prepared = prepare_smiles_graph_from_mol_to_smiles_kwargs(
        mol_or_prepared,
        surface_kind=surface_kind,
        isomeric_smiles=flags.isomeric_smiles,
        kekule_smiles=flags.kekule_smiles,
        all_bonds_explicit=flags.all_bonds_explicit,
        all_hs_explicit=flags.all_hs_explicit,
        ignore_atom_map_numbers=flags.ignore_atom_map_numbers,
    )
    return _core.PreparedSmilesGraph(reference_prepared)


make_prepared_graph = prepare_smiles_graph


def _make_flags(
    *,
    isomeric_smiles: bool = True,
    kekule_smiles: bool = False,
    rooted_at_atom: int = -1,
    canonical: bool = True,
    all_bonds_explicit: bool = False,
    all_hs_explicit: bool = False,
    do_random: bool = False,
    ignore_atom_map_numbers: bool = False,
) -> MolToSmilesFlags:
    def _normalize_bool_like(value: object) -> object:
        if value is None:
            return False
        if isinstance(value, Integral):
            return bool(value)
        return value

    def _normalize_root(value: object) -> object:
        if isinstance(value, Integral):
            return int(value)
        return value

    return MolToSmilesFlags(
        isomeric_smiles=_normalize_bool_like(isomeric_smiles),
        kekule_smiles=_normalize_bool_like(kekule_smiles),
        rooted_at_atom=_normalize_root(rooted_at_atom),
        canonical=_normalize_bool_like(canonical),
        all_bonds_explicit=_normalize_bool_like(all_bonds_explicit),
        all_hs_explicit=_normalize_bool_like(all_hs_explicit),
        do_random=_normalize_bool_like(do_random),
        ignore_atom_map_numbers=_normalize_bool_like(ignore_atom_map_numbers),
    )


def _instantiate_core_object(
    mol_or_prepared: object,
    flags: MolToSmilesFlags,
    *,
    stereo_type: type,
    nonstereo_type: type,
) -> object:
    if isinstance(mol_or_prepared, PreparedMol):
        plan = _connected_prepared_mol_plan(
            mol_or_prepared,
            rooted_at_atom=flags.rooted_at_atom,
        )
        mol_or_prepared = plan.fragment
        flags = flags.with_rooted_at_atom(cast(int, plan.rooted_at_atom))

    prepared = prepare_smiles_graph(mol_or_prepared, flags=flags)
    core_type = stereo_type if prepared.surface_kind == CONNECTED_STEREO_SURFACE else nonstereo_type
    return core_type(prepared, flags.rooted_at_atom)


def _make_walker(
    mol_or_prepared: object,
    flags: MolToSmilesFlags,
) -> object:
    return _instantiate_core_object(
        mol_or_prepared,
        flags,
        stereo_type=_core.RootedConnectedStereoWalker,
        nonstereo_type=_core.RootedConnectedNonStereoWalker,
    )


def _make_decoder(
    mol_or_prepared: object,
    flags: MolToSmilesFlags,
) -> object:
    return _instantiate_core_object(
        mol_or_prepared,
        flags,
        stereo_type=_core.RootedConnectedStereoDecoder,
        nonstereo_type=_core.RootedConnectedNonStereoDecoder,
    )


def _make_connected_state_adapter(
    mol_or_prepared: object,
    flags: MolToSmilesFlags,
) -> _CoreStateAdapter:
    return _CoreStateAdapter(_make_decoder(mol_or_prepared, flags))


def _make_fragment_state_adapter(
    fragment_mol: object,
    *,
    flags: MolToSmilesFlags,
    rooted_at_atom: int | None,
) -> object:
    if rooted_at_atom is not None:
        return _make_connected_state_adapter(fragment_mol, flags.with_rooted_at_atom(rooted_at_atom))

    atom_count = _atom_count(fragment_mol)
    if atom_count == 0:
        return _make_connected_state_adapter(fragment_mol, flags.with_rooted_at_atom(0))

    fragment_for_preparation = (
        _connected_prepared_mol_plan(fragment_mol, rooted_at_atom=-1).fragment
        if isinstance(fragment_mol, PreparedMol)
        else fragment_mol
    )
    prepared_fragment = prepare_smiles_graph(
        fragment_for_preparation,
        flags=flags.with_rooted_at_atom(0),
    )
    if prepared_fragment.surface_kind == CONNECTED_NONSTEREO_SURFACE:
        return _make_connected_state_adapter(prepared_fragment, flags.with_rooted_at_atom(-1))

    states = tuple(
        _make_decoder(prepared_fragment, flags.with_rooted_at_atom(local_root_idx))
        for local_root_idx in range(atom_count)
    )
    if len(states) == 1:
        return states[0]
    return _MergedStateAdapter(states)


def _make_disconnected_decoder(
    prepared: PreparedMol,
    flags: MolToSmilesFlags,
) -> _DisconnectedStateAdapter:
    rooted_at_atom = None if flags.rooted_at_atom < 0 else flags.rooted_at_atom
    fragment_states = tuple(
        _make_fragment_state_adapter(
            plan.fragment,
            flags=flags,
            rooted_at_atom=plan.rooted_at_atom,
        )
        for plan in _fragment_plans_for_prepared_mol(
            prepared,
            rooted_at_atom=rooted_at_atom,
        )
    )
    return _DisconnectedStateAdapter(fragment_states)


def _make_decoder_state_impl(
    mol_or_prepared: object,
    *,
    flags: MolToSmilesFlags,
) -> object:
    disconnected = _as_disconnected_prepared_mol(mol_or_prepared)
    if disconnected is not None:
        return _make_disconnected_decoder(disconnected, flags)
    if flags.rooted_at_atom < 0:
        return _make_fragment_state_adapter(
            mol_or_prepared,
            flags=flags,
            rooted_at_atom=None,
        )
    return _make_connected_state_adapter(mol_or_prepared, flags)


class _PublicDecoderBase:
    __slots__ = ("_state", "_choices_cache")
    _prefer_choice_cache_for_terminal = True

    def __init__(
        self,
        mol_or_prepared: object,
        *,
        isomeric_smiles: bool = True,
        kekule_smiles: bool = False,
        rooted_at_atom: int = -1,
        canonical: bool = True,
        all_bonds_explicit: bool = False,
        all_hs_explicit: bool = False,
        do_random: bool = False,
        ignore_atom_map_numbers: bool = False,
    ) -> None:
        flags = _make_flags(
            isomeric_smiles=isomeric_smiles,
            kekule_smiles=kekule_smiles,
            rooted_at_atom=rooted_at_atom,
            canonical=canonical,
            all_bonds_explicit=all_bonds_explicit,
            all_hs_explicit=all_hs_explicit,
            do_random=do_random,
            ignore_atom_map_numbers=ignore_atom_map_numbers,
        )
        _validate_supported_flags(flags)
        self._state = _make_decoder_state_impl(
            _prepare_runtime_input(mol_or_prepared, flags=flags),
            flags=flags,
        )
        self._choices_cache = None

    @classmethod
    def _from_parts(
        cls,
        state_impl: object,
    ) -> "_PublicDecoderBase":
        return _new_public_decoder(cls, state_impl)

    @property
    def prefix(self) -> str:
        return self._state.prefix()

    @property
    def is_terminal(self) -> bool:
        if self._choices_cache is not None:
            return not self._choices_cache
        if type(self)._prefer_choice_cache_for_terminal:
            return not self.next_choices
        return self._state.is_terminal()

    def copy(self) -> "_PublicDecoderBase":
        return type(self)._from_parts(self._state.copy())

    @property
    def next_choices(self) -> tuple[MolToSmilesChoice, ...]:
        if self._choices_cache is None:
            self._choices_cache = self.choices()
        return self._choices_cache


class MolToSmilesDecoder(_PublicDecoderBase):
    def choices(self) -> tuple[MolToSmilesChoice, ...]:
        successors = _choice_successor_states(self._state)
        if not successors:
            return ()
        decoder_type = type(self)
        decoder_new = decoder_type.__new__
        choice_new = MolToSmilesChoice.__new__
        choices = [None] * len(successors)
        for idx, (text, successor) in enumerate(successors):
            next_state = decoder_new(decoder_type)
            next_state._state = successor
            next_state._choices_cache = None
            choice = choice_new(MolToSmilesChoice)
            choice.text = text
            choice.next_state = next_state
            choices[idx] = choice
        return tuple(choices)


class MolToSmilesDeterminizedDecoder(_PublicDecoderBase):
    def choices(self) -> tuple[MolToSmilesChoice, ...]:
        successors = _determinized_choice_successors(self._state)
        if not successors:
            return ()
        decoder_type = type(self)
        decoder_new = decoder_type.__new__
        choice_new = MolToSmilesChoice.__new__
        choices = [None] * len(successors)
        for idx, (text, successor) in enumerate(successors):
            next_state = decoder_new(decoder_type)
            next_state._state = successor
            next_state._choices_cache = None
            choice = choice_new(MolToSmilesChoice)
            choice.text = text
            choice.next_state = next_state
            choices[idx] = choice
        return tuple(choices)


def _token_inventory_root_indices(
    mol_or_prepared: object,
    *,
    rooted_at_atom: int,
) -> tuple[int, ...]:
    atom_count = _atom_count(mol_or_prepared)
    if atom_count == 0:
        return (0,)
    if rooted_at_atom < 0:
        return (-1,)
    return (rooted_at_atom,)


def _exact_token_inventory_from_decoder(
    mol_or_prepared: object,
    *,
    flags: MolToSmilesFlags,
) -> tuple[str, ...]:
    inventory: set[str] = set()
    visited_state_keys: set[DecoderCacheKey] = set()

    for root_idx in _token_inventory_root_indices(
        mol_or_prepared,
        rooted_at_atom=flags.rooted_at_atom,
    ):
        stack = [
            _make_decoder_state_impl(
                mol_or_prepared,
                flags=flags.with_rooted_at_atom(root_idx),
            )
        ]

        while stack:
            state = stack.pop()
            state_key = _state_cache_key(state)
            if state_key in visited_state_keys:
                continue
            visited_state_keys.add(state_key)
            grouped_successors = _determinized_choice_successors(state)
            inventory.update(text for text, _ in grouped_successors)
            stack.extend(successor for _, successor in grouped_successors)

    return tuple(sorted(inventory))


def _stereo_atom_token_superset_variants(
    prepared: ReferencePreparedSmilesGraph,
    atom_idx: int,
) -> tuple[str, ...]:
    if not prepared.writer_do_isomeric_smiles:
        return ()

    chiral_tag = prepared.atom_chiral_tags[atom_idx]
    if chiral_tag not in {"CHI_TETRAHEDRAL_CCW", "CHI_TETRAHEDRAL_CW"}:
        return ()

    return tuple(
        prepared_stereo_atom_token(
            prepared,
            atom_idx,
            stereo_mark=stereo_mark,
        )
        for stereo_mark in ("@", "@@")
    )


def _branch_tokens_may_be_reachable(
    prepared: ReferencePreparedSmilesGraph,
    *,
    rooted_at_atom: int,
) -> bool:
    if rooted_at_atom < 0:
        return any(len(row) >= 2 for row in prepared.neighbors)
    return any(
        len(row) >= (2 if atom_idx == rooted_at_atom else 3)
        for atom_idx, row in enumerate(prepared.neighbors)
    )


def _prepared_token_inventory_superset(
    prepared: ReferencePreparedSmilesGraph,
    *,
    rooted_at_atom: int,
) -> tuple[str, ...]:
    atom_count = prepared.atom_count
    if atom_count > 0 and rooted_at_atom >= atom_count:
        raise IndexError("root_idx out of range")

    tokens = set(prepared.atom_tokens)

    for bond_token_row in prepared.neighbor_bond_tokens:
        tokens.update(token for token in bond_token_row if token)

    if any(bond_dir != "NONE" for bond_dir in prepared.bond_dirs):
        tokens.update(("/", "\\"))

    if _branch_tokens_may_be_reachable(prepared, rooted_at_atom=rooted_at_atom):
        tokens.update(("(", ")"))

    # A connected graph can need no more distinct ring labels than its cycle rank.
    cycle_rank = max(0, prepared.bond_count - atom_count + 1) if atom_count else 0
    tokens.update(ring_label_text(label) for label in range(1, cycle_rank + 1))

    if prepared.surface_kind == CONNECTED_STEREO_SURFACE:
        for atom_idx in range(atom_count):
            tokens.update(_stereo_atom_token_superset_variants(prepared, atom_idx))

    return tuple(sorted(tokens))


def _prepare_reference_graph_for_static_inventory(
    mol_or_prepared: object,
    *,
    flags: MolToSmilesFlags,
) -> ReferencePreparedSmilesGraph:
    surface_kind = _runtime_surface_kind(mol_or_prepared, flags=flags)
    if isinstance(mol_or_prepared, ReferencePreparedSmilesGraph):
        _validate_surface_kind(mol_or_prepared, surface_kind=surface_kind)
        _validate_writer_flags(mol_or_prepared, flags)
        return mol_or_prepared

    if isinstance(mol_or_prepared, _core.PreparedSmilesGraph):
        _validate_surface_kind(mol_or_prepared, surface_kind=surface_kind)
        _validate_writer_flags(mol_or_prepared, flags)
        return ReferencePreparedSmilesGraph.from_dict(mol_or_prepared.to_dict())

    _ensure_singly_connected_molecule(mol_or_prepared)
    return prepare_smiles_graph_from_mol_to_smiles_kwargs(
        mol_or_prepared,
        surface_kind=surface_kind,
        isomeric_smiles=flags.isomeric_smiles,
        kekule_smiles=flags.kekule_smiles,
        all_bonds_explicit=flags.all_bonds_explicit,
        all_hs_explicit=flags.all_hs_explicit,
        ignore_atom_map_numbers=flags.ignore_atom_map_numbers,
    )


def _connected_mol_to_smiles_token_inventory_superset(
    mol_or_prepared: object,
    *,
    flags: MolToSmilesFlags,
) -> tuple[str, ...]:
    rooted_at_atom = flags.rooted_at_atom
    if isinstance(mol_or_prepared, PreparedMol):
        plan = _connected_prepared_mol_plan(
            mol_or_prepared,
            rooted_at_atom=flags.rooted_at_atom,
        )
        rooted_at_atom = cast(int, plan.rooted_at_atom)
        mol_or_prepared = plan.fragment
        flags = flags.with_rooted_at_atom(rooted_at_atom)

    prepared = _prepare_reference_graph_for_static_inventory(
        mol_or_prepared,
        flags=flags,
    )
    return _prepared_token_inventory_superset(
        prepared,
        rooted_at_atom=rooted_at_atom,
    )


def _fragmented_prepared_mol_to_smiles_token_inventory_superset(
    prepared: PreparedMol,
    *,
    flags: MolToSmilesFlags,
) -> tuple[str, ...]:
    rooted_at_atom = None if flags.rooted_at_atom < 0 else flags.rooted_at_atom
    inventory: set[str] = set()
    fragment_plans = _fragment_plans_for_prepared_mol(
        prepared,
        rooted_at_atom=rooted_at_atom,
    )

    for plan in fragment_plans:
        fragment_root = -1 if plan.rooted_at_atom is None else plan.rooted_at_atom
        inventory.update(
            _connected_mol_to_smiles_token_inventory_superset(
                plan.fragment,
                flags=flags.with_rooted_at_atom(fragment_root),
            )
        )

    if len(fragment_plans) > 1:
        inventory.add(".")

    return tuple(sorted(inventory))


def mol_to_smiles_enum(
    mol_or_prepared: object,
    *,
    isomeric_smiles: bool = True,
    kekule_smiles: bool = False,
    rooted_at_atom: int = -1,
    canonical: bool = True,
    all_bonds_explicit: bool = False,
    all_hs_explicit: bool = False,
    do_random: bool = False,
    ignore_atom_map_numbers: bool = False,
) -> Iterator[str]:
    flags = _make_flags(
        isomeric_smiles=isomeric_smiles,
        kekule_smiles=kekule_smiles,
        rooted_at_atom=rooted_at_atom,
        canonical=canonical,
        all_bonds_explicit=all_bonds_explicit,
        all_hs_explicit=all_hs_explicit,
        do_random=do_random,
        ignore_atom_map_numbers=ignore_atom_map_numbers,
    )
    _validate_supported_flags(flags)
    mol_or_prepared = _prepare_runtime_input(mol_or_prepared, flags=flags)
    disconnected = _as_disconnected_prepared_mol(mol_or_prepared)
    if disconnected is not None:
        return iter(
            sorted(
                _fragmented_prepared_mol_to_smiles_support(
                    disconnected,
                    flags=flags,
                )
            )
        )
    if flags.rooted_at_atom < 0:
        return iter(
            sorted(
                _connected_fragment_support(
                    mol_or_prepared,
                    flags=flags,
                    rooted_at_atom=None,
                )
            )
        )
    walker = _make_walker(mol_or_prepared, flags)
    return iter(walker.enumerate_support())


def mol_to_smiles_support(
    mol_or_prepared: object,
    *,
    isomeric_smiles: bool = True,
    kekule_smiles: bool = False,
    rooted_at_atom: int = -1,
    canonical: bool = True,
    all_bonds_explicit: bool = False,
    all_hs_explicit: bool = False,
    do_random: bool = False,
    ignore_atom_map_numbers: bool = False,
) -> set[str]:
    return set(
        mol_to_smiles_enum(
            mol_or_prepared,
            isomeric_smiles=isomeric_smiles,
            kekule_smiles=kekule_smiles,
            rooted_at_atom=rooted_at_atom,
            canonical=canonical,
            all_bonds_explicit=all_bonds_explicit,
            all_hs_explicit=all_hs_explicit,
            do_random=do_random,
            ignore_atom_map_numbers=ignore_atom_map_numbers,
        )
    )


def mol_to_smiles_token_inventory(
    mol_or_prepared: object,
    *,
    isomeric_smiles: bool = True,
    kekule_smiles: bool = False,
    rooted_at_atom: int = -1,
    canonical: bool = True,
    all_bonds_explicit: bool = False,
    all_hs_explicit: bool = False,
    do_random: bool = False,
    ignore_atom_map_numbers: bool = False,
) -> tuple[str, ...]:
    """Return the exact decoder token inventory under the public runtime flags."""

    flags = _make_flags(
        isomeric_smiles=isomeric_smiles,
        kekule_smiles=kekule_smiles,
        rooted_at_atom=rooted_at_atom,
        canonical=canonical,
        all_bonds_explicit=all_bonds_explicit,
        all_hs_explicit=all_hs_explicit,
        do_random=do_random,
        ignore_atom_map_numbers=ignore_atom_map_numbers,
    )
    _validate_supported_flags(flags)
    mol_or_prepared = _prepare_runtime_input(mol_or_prepared, flags=flags)
    return _exact_token_inventory_from_decoder(
        mol_or_prepared,
        flags=flags,
    )


def mol_to_smiles_token_inventory_superset(
    mol_or_prepared: object,
    *,
    isomeric_smiles: bool = True,
    kekule_smiles: bool = False,
    rooted_at_atom: int = -1,
    canonical: bool = True,
    all_bonds_explicit: bool = False,
    all_hs_explicit: bool = False,
    do_random: bool = False,
    ignore_atom_map_numbers: bool = False,
) -> tuple[str, ...]:
    """Return a conservative static superset of reachable decoder tokens."""

    flags = _make_flags(
        isomeric_smiles=isomeric_smiles,
        kekule_smiles=kekule_smiles,
        rooted_at_atom=rooted_at_atom,
        canonical=canonical,
        all_bonds_explicit=all_bonds_explicit,
        all_hs_explicit=all_hs_explicit,
        do_random=do_random,
        ignore_atom_map_numbers=ignore_atom_map_numbers,
    )
    _validate_supported_flags(flags)
    mol_or_prepared = _prepare_runtime_input(mol_or_prepared, flags=flags)
    disconnected = _as_disconnected_prepared_mol(mol_or_prepared)
    if disconnected is not None:
        return _fragmented_prepared_mol_to_smiles_token_inventory_superset(
            disconnected,
            flags=flags,
        )
    return _connected_mol_to_smiles_token_inventory_superset(
        mol_or_prepared,
        flags=flags,
    )


def enumerate_rooted_connected_nonstereo_smiles_support(
    mol_or_prepared: object,
    root_idx: int,
) -> set[str]:
    return mol_to_smiles_support(
        mol_or_prepared,
        isomeric_smiles=False,
        rooted_at_atom=root_idx,
        canonical=False,
        do_random=True,
    )


def enumerate_rooted_connected_stereo_smiles_support(
    mol_or_prepared: object,
    root_idx: int,
) -> set[str]:
    return mol_to_smiles_support(
        mol_or_prepared,
        isomeric_smiles=True,
        rooted_at_atom=root_idx,
        canonical=False,
        do_random=True,
    )


def make_nonstereo_walker(
    mol_or_prepared: object,
    root_idx: int,
) -> _core.RootedConnectedNonStereoWalker:
    flags = _make_flags(
        isomeric_smiles=False,
        rooted_at_atom=root_idx,
        canonical=False,
        do_random=True,
    )
    return cast(
        _core.RootedConnectedNonStereoWalker,
        _instantiate_core_object(
            mol_or_prepared,
            flags,
            stereo_type=_core.RootedConnectedStereoWalker,
            nonstereo_type=_core.RootedConnectedNonStereoWalker,
        ),
    )


def make_stereo_walker(
    mol_or_prepared: object,
    root_idx: int,
) -> _core.RootedConnectedStereoWalker:
    flags = _make_flags(
        isomeric_smiles=True,
        rooted_at_atom=root_idx,
        canonical=False,
        do_random=True,
    )
    return cast(
        _core.RootedConnectedStereoWalker,
        _instantiate_core_object(
            mol_or_prepared,
            flags,
            stereo_type=_core.RootedConnectedStereoWalker,
            nonstereo_type=_core.RootedConnectedNonStereoWalker,
        ),
    )


def prepared_smiles_graph_schema_version() -> int:
    core_version = _core.prepared_smiles_graph_schema_version()
    if core_version != PREPARED_SMILES_GRAPH_SCHEMA_VERSION:
        raise RuntimeError(
            "Python RDKit bridge and Rust core disagree on prepared graph schema "
            f"version: python={PREPARED_SMILES_GRAPH_SCHEMA_VERSION}, core={core_version}"
        )
    return core_version


__all__ = [
    "CONNECTED_NONSTEREO_SURFACE",
    "CONNECTED_STEREO_SURFACE",
    "MolToSmilesDecoder",
    "MolToSmilesDeterminizedDecoder",
    "MolToSmilesFlags",
    "PREPARED_SMILES_GRAPH_SCHEMA_VERSION",
    "enumerate_rooted_connected_nonstereo_smiles_support",
    "enumerate_rooted_connected_stereo_smiles_support",
    "make_nonstereo_walker",
    "make_prepared_graph",
    "make_stereo_walker",
    "mol_to_smiles_enum",
    "mol_to_smiles_support",
    "prepare_smiles_graph",
    "prepared_smiles_graph_schema_version",
]
