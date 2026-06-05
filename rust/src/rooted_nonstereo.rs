use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use pyo3::exceptions::{PyIndexError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyAny;
use rustc_hash::FxHashSet;

use crate::frontier::{
    branch_choice_texts, decoder_choices_from_token_successors, dedup_frontier,
    extend_decoder_choices_from_token_successors, frontier_prefix as shared_frontier_prefix,
    group_decoder_choices, take_branch_choice_successors_or_err, take_first_successor_or_err,
    take_only_successor_or_err, take_token_successors_or_err, take_token_support_successors_or_err,
    token_support_from_choices, DecoderChoice, GroupedTransition,
};
use crate::prepared_graph::PreparedSmilesGraphData;
use crate::smiles_shared::{add_pending, ring_label_text, take_pending_for_atom};

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum BondTokenCode {
    Elided,
    Aromatic,
    Single,
    Double,
    Triple,
    DativeForward,
    DativeBackward,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum LiteralToken {
    BranchOpen,
    BranchClose,
    Bond(BondTokenCode),
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum EmittedToken {
    Literal(LiteralToken),
    RingLabel(usize),
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct PendingRing {
    label: usize,
    bond_token: BondTokenCode,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct AfterAtomAction {
    atom_idx: usize,
    closures_here: Vec<PendingRing>,
    neighbor_groups: Vec<Vec<usize>>,
    opening_count: usize,
    ring_action_count: usize,
    linear_child_idx: Option<usize>,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum Action {
    EmitToken(EmittedToken),
    EnterAtom(usize),
    AfterAtom(AfterAtomAction),
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct AtomBitSet {
    atom_count: usize,
    words: Vec<u64>,
}

impl AtomBitSet {
    const WORD_BITS: usize = u64::BITS as usize;

    fn new(atom_count: usize) -> Self {
        Self {
            atom_count,
            words: vec![0; atom_count.div_ceil(Self::WORD_BITS)],
        }
    }

    fn len(&self) -> usize {
        self.atom_count
    }

    fn bit_mask(atom_idx: usize) -> u64 {
        1_u64 << (Self::WORD_BITS - 1 - (atom_idx % Self::WORD_BITS))
    }

    fn contains(&self, atom_idx: usize) -> bool {
        debug_assert!(atom_idx < self.atom_count);
        let word_idx = atom_idx / Self::WORD_BITS;
        (self.words[word_idx] & Self::bit_mask(atom_idx)) != 0
    }

    fn insert(&mut self, atom_idx: usize) {
        debug_assert!(atom_idx < self.atom_count);
        let word_idx = atom_idx / Self::WORD_BITS;
        self.words[word_idx] |= Self::bit_mask(atom_idx);
    }
}

impl Ord for AtomBitSet {
    fn cmp(&self, other: &Self) -> Ordering {
        if self.atom_count == other.atom_count {
            return self.words.cmp(&other.words);
        }
        let common_atom_count = self.atom_count.min(other.atom_count);
        for atom_idx in 0..common_atom_count {
            match self.contains(atom_idx).cmp(&other.contains(atom_idx)) {
                Ordering::Equal => {}
                ordering => return ordering,
            }
        }
        self.atom_count.cmp(&other.atom_count)
    }
}

impl PartialOrd for AtomBitSet {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct RootedConnectedNonStereoWalkerStateData {
    prefix: String,
    visited: AtomBitSet,
    visited_count: usize,
    pending: Vec<(usize, Vec<PendingRing>)>,
    free_labels: Vec<usize>,
    next_label: usize,
    action_stack: Vec<Action>,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum RingAction {
    Close(usize),
    Open(usize),
}

fn bond_token_code(token: &str) -> BondTokenCode {
    match token {
        "" => BondTokenCode::Elided,
        ":" => BondTokenCode::Aromatic,
        "-" => BondTokenCode::Single,
        "=" => BondTokenCode::Double,
        "#" => BondTokenCode::Triple,
        "->" => BondTokenCode::DativeForward,
        "<-" => BondTokenCode::DativeBackward,
        _ => unreachable!(
            "PreparedSmilesGraph connected_nonstereo validation should reject unsupported bond token: {token:?}"
        ),
    }
}

fn bond_token_text(token: BondTokenCode) -> &'static str {
    match token {
        BondTokenCode::Elided => "",
        BondTokenCode::Aromatic => ":",
        BondTokenCode::Single => "-",
        BondTokenCode::Double => "=",
        BondTokenCode::Triple => "#",
        BondTokenCode::DativeForward => "->",
        BondTokenCode::DativeBackward => "<-",
    }
}

fn emitted_token_string(token: &EmittedToken) -> String {
    match token {
        EmittedToken::Literal(LiteralToken::BranchOpen) => "(".to_owned(),
        EmittedToken::Literal(LiteralToken::BranchClose) => ")".to_owned(),
        EmittedToken::Literal(LiteralToken::Bond(bond)) => bond_token_text(*bond).to_owned(),
        EmittedToken::RingLabel(label) => ring_label_text(*label),
    }
}

fn append_emitted_token(prefix: &mut String, token: &EmittedToken) {
    match token {
        EmittedToken::Literal(LiteralToken::BranchOpen) => prefix.push('('),
        EmittedToken::Literal(LiteralToken::BranchClose) => prefix.push(')'),
        EmittedToken::Literal(LiteralToken::Bond(bond)) => prefix.push_str(bond_token_text(*bond)),
        EmittedToken::RingLabel(label) => prefix.push_str(&ring_label_text(*label)),
    }
}

fn insert_sorted(labels: &mut Vec<usize>, label: usize) {
    match labels.binary_search(&label) {
        Ok(offset) | Err(offset) => labels.insert(offset, label),
    }
}

fn allocate_label(free_labels: &mut Vec<usize>, next_label: &mut usize) -> usize {
    if !free_labels.is_empty() {
        free_labels.remove(0)
    } else {
        let label = *next_label;
        *next_label += 1;
        label
    }
}

fn validate_root_idx(graph: &PreparedSmilesGraphData, root_idx: isize) -> PyResult<usize> {
    if graph.atom_count() == 0 {
        return Ok(0);
    }
    if root_idx < 0 || root_idx as usize >= graph.atom_count() {
        return Err(PyIndexError::new_err("root_idx out of range"));
    }
    Ok(root_idx as usize)
}

fn initial_frontier_for_root_spec(
    graph: &PreparedSmilesGraphData,
    root_idx: isize,
) -> PyResult<Vec<RootedConnectedNonStereoWalkerStateData>> {
    if graph.atom_count() == 0 {
        return Ok(vec![initial_state_for_root(graph, 0)]);
    }
    if root_idx < 0 {
        return Ok((0..graph.atom_count())
            .map(|atom_idx| initial_state_for_root(graph, atom_idx))
            .collect());
    }
    Ok(vec![initial_state_for_root(
        graph,
        validate_root_idx(graph, root_idx)?,
    )])
}

fn validate_state_shape(
    graph: &PreparedSmilesGraphData,
    state: &RootedConnectedNonStereoWalkerStateData,
) -> PyResult<()> {
    if state.visited.len() != graph.atom_count()
        || state
            .pending
            .iter()
            .any(|(atom_idx, _rings)| *atom_idx >= graph.atom_count())
    {
        return Err(PyValueError::new_err(
            "walker state is not compatible with this PreparedSmilesGraph",
        ));
    }
    Ok(())
}

fn ordered_neighbor_groups(
    graph: &PreparedSmilesGraphData,
    atom_idx: usize,
    visited: &AtomBitSet,
) -> Vec<Vec<usize>> {
    let mut seeds = graph
        .neighbors_of(atom_idx)
        .iter()
        .copied()
        .filter(|&neighbor_idx| !visited.contains(neighbor_idx))
        .collect::<Vec<_>>();
    if seeds.is_empty() {
        return Vec::new();
    }
    seeds.sort_unstable();
    if seeds.len() == 1 {
        return vec![seeds];
    }
    let mut groups_with_mins = Vec::<(usize, Vec<usize>)>::new();
    let mut seen = vec![false; graph.atom_count()];
    let mut stack = Vec::new();
    for &seed in &seeds {
        if seen[seed] {
            continue;
        }
        seen[seed] = true;
        stack.push(seed);
        let mut component_min = seed;
        let mut group = vec![seed];

        while let Some(current) = stack.pop() {
            if current < component_min {
                component_min = current;
            }
            for &neighbor_idx in graph.neighbors_of(current) {
                if neighbor_idx == atom_idx || visited.contains(neighbor_idx) || seen[neighbor_idx]
                {
                    continue;
                }
                seen[neighbor_idx] = true;
                if seeds.binary_search(&neighbor_idx).is_ok() {
                    group.push(neighbor_idx);
                }
                stack.push(neighbor_idx);
            }
        }

        group.sort_unstable();
        groups_with_mins.push((component_min, group));
    }

    groups_with_mins.sort_by_key(|(component_min, _)| *component_min);
    groups_with_mins
        .into_iter()
        .map(|(_component_min, group)| group)
        .collect()
}

fn for_each_cartesian_choice<F>(groups: &[Vec<usize>], f: &mut F)
where
    F: FnMut(&[usize]),
{
    fn recurse<F>(groups: &[Vec<usize>], group_idx: usize, current: &mut Vec<usize>, f: &mut F)
    where
        F: FnMut(&[usize]),
    {
        if group_idx == groups.len() {
            f(current);
            return;
        }

        for &choice in &groups[group_idx] {
            current.push(choice);
            recurse(groups, group_idx + 1, current, f);
            current.pop();
        }
    }

    let mut current = Vec::with_capacity(groups.len());
    recurse(groups, 0, &mut current, f);
}

#[cfg(test)]
fn for_each_permutation_py_order<T: Clone, F>(items: &[T], f: &mut F)
where
    F: FnMut(&[T]),
{
    fn recurse<T: Clone, F>(items: &[T], used: &mut [bool], current: &mut Vec<T>, f: &mut F)
    where
        F: FnMut(&[T]),
    {
        if current.len() == items.len() {
            f(current);
            return;
        }
        for idx in 0..items.len() {
            if used[idx] {
                continue;
            }
            used[idx] = true;
            current.push(items[idx].clone());
            recurse(items, used, current, f);
            current.pop();
            used[idx] = false;
        }
    }

    let mut used = vec![false; items.len()];
    let mut current = Vec::with_capacity(items.len());
    recurse(items, &mut used, &mut current, f);
}

fn for_each_permutation_py_order_copy<T: Copy, F>(items: &[T], f: &mut F)
where
    F: FnMut(&[T]),
{
    match items.len() {
        0 => f(&[]),
        1 => f(items),
        2 => {
            let order = [items[0], items[1]];
            f(&order);
            let order = [items[1], items[0]];
            f(&order);
        }
        3 => {
            let order = [items[0], items[1], items[2]];
            f(&order);
            let order = [items[0], items[2], items[1]];
            f(&order);
            let order = [items[1], items[0], items[2]];
            f(&order);
            let order = [items[1], items[2], items[0]];
            f(&order);
            let order = [items[2], items[0], items[1]];
            f(&order);
            let order = [items[2], items[1], items[0]];
            f(&order);
        }
        _ => {
            fn recurse<T: Copy, F>(items: &[T], used: &mut [bool], current: &mut Vec<T>, f: &mut F)
            where
                F: FnMut(&[T]),
            {
                if current.len() == items.len() {
                    f(current);
                    return;
                }
                for idx in 0..items.len() {
                    if used[idx] {
                        continue;
                    }
                    used[idx] = true;
                    current.push(items[idx]);
                    recurse(items, used, current, f);
                    current.pop();
                    used[idx] = false;
                }
            }

            let mut used = vec![false; items.len()];
            let mut current = Vec::with_capacity(items.len());
            recurse(items, &mut used, &mut current, f);
        }
    }
}

#[cfg(test)]
fn permutations_py_order<T: Clone>(items: &[T]) -> Vec<Vec<T>> {
    let mut out = Vec::new();
    for_each_permutation_py_order(items, &mut |perm| out.push(perm.to_vec()));
    out
}

#[cfg(test)]
fn permutations_py_order_copy<T: Copy>(items: &[T]) -> Vec<Vec<T>> {
    let mut out = Vec::new();
    for_each_permutation_py_order_copy(items, &mut |perm| out.push(perm.to_vec()));
    out
}

fn make_after_atom_action(
    atom_idx: usize,
    closures_here: Vec<PendingRing>,
    neighbor_groups: Vec<Vec<usize>>,
) -> AfterAtomAction {
    let opening_count = neighbor_groups
        .iter()
        .map(|group| group.len().saturating_sub(1))
        .sum::<usize>();
    let ring_action_count = closures_here.len() + opening_count;
    let linear_child_idx =
        if ring_action_count == 0 && neighbor_groups.len() == 1 && neighbor_groups[0].len() == 1 {
            Some(neighbor_groups[0][0])
        } else {
            None
        };
    AfterAtomAction {
        atom_idx,
        closures_here,
        neighbor_groups,
        opening_count,
        ring_action_count,
        linear_child_idx,
    }
}

fn initial_state_for_root(
    graph: &PreparedSmilesGraphData,
    root_idx: usize,
) -> RootedConnectedNonStereoWalkerStateData {
    let mut action_stack = Vec::new();
    if graph.atom_count() > 0 {
        action_stack.push(Action::EnterAtom(root_idx));
    }
    RootedConnectedNonStereoWalkerStateData {
        prefix: String::new(),
        visited: AtomBitSet::new(graph.atom_count()),
        visited_count: 0,
        pending: Vec::new(),
        free_labels: Vec::new(),
        next_label: 1,
        action_stack,
    }
}

fn normalize_state(state: &mut RootedConnectedNonStereoWalkerStateData) {
    while let Some(Action::AfterAtom(action)) = state.action_stack.last() {
        if action.ring_action_count > 0 || !action.neighbor_groups.is_empty() {
            return;
        }
        state.action_stack.pop();
    }
}

fn is_terminal_state(state: &RootedConnectedNonStereoWalkerStateData) -> bool {
    let mut normalized = state.clone();
    normalize_state(&mut normalized);
    normalized.action_stack.is_empty()
}

fn next_open_label(state: &RootedConnectedNonStereoWalkerStateData) -> usize {
    state
        .free_labels
        .first()
        .copied()
        .unwrap_or(state.next_label)
}

fn edge_prefix_or_atom(
    graph: &PreparedSmilesGraphData,
    parent_idx: usize,
    child_idx: usize,
) -> String {
    let token = adjacent_bond_token(graph, parent_idx, child_idx, "edge prefix or atom");
    if token.is_empty() {
        graph.atom_token(child_idx).to_owned()
    } else {
        token.to_owned()
    }
}

fn adjacent_bond_token<'a>(
    graph: &'a PreparedSmilesGraphData,
    parent_idx: usize,
    child_idx: usize,
    context: &str,
) -> &'a str {
    match graph.bond_token(parent_idx, child_idx) {
        Some(token) => token,
        None => unreachable!(
            "nonstereo walker expected adjacent atoms in {context}: {parent_idx}, {child_idx}"
        ),
    }
}

fn opening_targets_from_choices(
    neighbor_groups: &[Vec<usize>],
    chosen_children: &[usize],
) -> Vec<usize> {
    let mut opening_targets = Vec::new();
    for (group_index, group) in neighbor_groups.iter().enumerate() {
        let chosen_child = chosen_children[group_index];
        for &neighbor_idx in group {
            if neighbor_idx != chosen_child {
                opening_targets.push(neighbor_idx);
            }
        }
    }
    opening_targets
}

fn push_child_actions(
    graph: &PreparedSmilesGraphData,
    action_stack: &mut Vec<Action>,
    parent_idx: usize,
    child_order: &[usize],
    first_branch_open_consumed: bool,
) {
    if child_order.is_empty() {
        return;
    }

    let branch_children = &child_order[..child_order.len() - 1];
    let main_child = child_order[child_order.len() - 1];

    let main_prefix = adjacent_bond_token(graph, parent_idx, main_child, "main child action");
    action_stack.push(Action::EnterAtom(main_child));
    if !main_prefix.is_empty() {
        action_stack.push(Action::EmitToken(EmittedToken::Literal(
            LiteralToken::Bond(bond_token_code(main_prefix)),
        )));
    }

    if branch_children.is_empty() {
        return;
    }

    for &child_idx in branch_children[1..].iter().rev() {
        action_stack.push(Action::EmitToken(EmittedToken::Literal(
            LiteralToken::BranchClose,
        )));
        action_stack.push(Action::EnterAtom(child_idx));
        let edge_prefix = adjacent_bond_token(graph, parent_idx, child_idx, "branch child action");
        if !edge_prefix.is_empty() {
            action_stack.push(Action::EmitToken(EmittedToken::Literal(
                LiteralToken::Bond(bond_token_code(edge_prefix)),
            )));
        }
        action_stack.push(Action::EmitToken(EmittedToken::Literal(
            LiteralToken::BranchOpen,
        )));
    }

    let first_branch_child = branch_children[0];
    action_stack.push(Action::EmitToken(EmittedToken::Literal(
        LiteralToken::BranchClose,
    )));
    action_stack.push(Action::EnterAtom(first_branch_child));
    let first_edge_prefix = adjacent_bond_token(
        graph,
        parent_idx,
        first_branch_child,
        "first branch child action",
    );
    if !first_edge_prefix.is_empty() {
        action_stack.push(Action::EmitToken(EmittedToken::Literal(
            LiteralToken::Bond(bond_token_code(first_edge_prefix)),
        )));
    }
    if !first_branch_open_consumed {
        action_stack.push(Action::EmitToken(EmittedToken::Literal(
            LiteralToken::BranchOpen,
        )));
    }
}

fn consume_enter_atom(
    graph: &PreparedSmilesGraphData,
    state: &mut RootedConnectedNonStereoWalkerStateData,
    atom_idx: usize,
) {
    debug_assert!(!state.visited.contains(atom_idx));
    state.visited.insert(atom_idx);
    state.visited_count += 1;
    let closures_here = take_pending_for_atom(&mut state.pending, atom_idx);
    let neighbor_groups = ordered_neighbor_groups(graph, atom_idx, &state.visited);
    state
        .action_stack
        .push(Action::AfterAtom(make_after_atom_action(
            atom_idx,
            closures_here,
            neighbor_groups,
        )));
}

fn apply_exact_ring_plan(
    graph: &PreparedSmilesGraphData,
    state: &mut RootedConnectedNonStereoWalkerStateData,
    action: &AfterAtomAction,
    ring_action_order: &[RingAction],
    child_order: &[usize],
) {
    let mut current_pending = state.pending.clone();
    let mut current_free = state.free_labels.clone();
    let mut current_next = state.next_label;
    let mut freed_labels = Vec::new();
    let mut emitted_suffix_tokens = Vec::<EmittedToken>::new();

    for (index, ring_action) in ring_action_order.iter().enumerate() {
        let is_first = index == 0;
        match *ring_action {
            RingAction::Close(closure_idx) => {
                let closure = &action.closures_here[closure_idx];
                if closure.bond_token != BondTokenCode::Elided {
                    if !is_first {
                        emitted_suffix_tokens.push(EmittedToken::Literal(LiteralToken::Bond(
                            closure.bond_token,
                        )));
                    }
                    emitted_suffix_tokens.push(EmittedToken::RingLabel(closure.label));
                } else if !is_first {
                    emitted_suffix_tokens.push(EmittedToken::RingLabel(closure.label));
                }
                freed_labels.push(closure.label);
            }
            RingAction::Open(target_idx) => {
                let label = allocate_label(&mut current_free, &mut current_next);
                add_pending(
                    &mut current_pending,
                    target_idx,
                    PendingRing {
                        label,
                        bond_token: bond_token_code(adjacent_bond_token(
                            graph,
                            action.atom_idx,
                            target_idx,
                            "exact ring opening",
                        )),
                    },
                );
                if !is_first {
                    emitted_suffix_tokens.push(EmittedToken::RingLabel(label));
                }
            }
        }
    }

    for label in freed_labels {
        insert_sorted(&mut current_free, label);
    }

    state.pending = current_pending;
    state.free_labels = current_free;
    state.next_label = current_next;
    push_child_actions(
        graph,
        &mut state.action_stack,
        action.atom_idx,
        child_order,
        false,
    );
    for token in emitted_suffix_tokens.into_iter().rev() {
        state.action_stack.push(Action::EmitToken(token));
    }
}

#[derive(Copy, Clone, Debug)]
enum LabelAllocUndo {
    Reused(usize),
    New(usize),
}

struct ExactRingPlanUndo {
    pending_opened: Vec<(usize, PendingRing)>,
    freed_labels: Vec<usize>,
    allocation_undos: Vec<LabelAllocUndo>,
    action_stack_len: usize,
}

fn remove_pending(
    pending: &mut Vec<(usize, Vec<PendingRing>)>,
    target_atom: usize,
    ring: &PendingRing,
) {
    let offset = match pending.binary_search_by_key(&target_atom, |(atom_idx, _)| *atom_idx) {
        Ok(offset) => offset,
        Err(_) => unreachable!("pending target should exist: {target_atom}"),
    };
    let rings = &mut pending[offset].1;
    let ring_offset = match rings.binary_search_by(|candidate| {
        (candidate.label, candidate.bond_token).cmp(&(ring.label, ring.bond_token))
    }) {
        Ok(offset) => offset,
        Err(_) => unreachable!(
            "pending ring should exist: target_atom={target_atom}, label={}",
            ring.label
        ),
    };
    rings.remove(ring_offset);
    if rings.is_empty() {
        pending.remove(offset);
    }
}

fn remove_sorted_label(labels: &mut Vec<usize>, label: usize) {
    let offset = match labels.binary_search(&label) {
        Ok(offset) => offset,
        Err(_) => unreachable!("label should exist in sorted free label pool: {label}"),
    };
    labels.remove(offset);
}

fn allocate_label_with_undo(
    free_labels: &mut Vec<usize>,
    next_label: &mut usize,
) -> (usize, LabelAllocUndo) {
    if !free_labels.is_empty() {
        let label = free_labels.remove(0);
        (label, LabelAllocUndo::Reused(label))
    } else {
        let label = *next_label;
        *next_label += 1;
        (label, LabelAllocUndo::New(label))
    }
}

fn undo_label_allocations(
    free_labels: &mut Vec<usize>,
    next_label: &mut usize,
    undos: &[LabelAllocUndo],
) {
    for undo in undos.iter().rev() {
        match *undo {
            LabelAllocUndo::Reused(label) => insert_sorted(free_labels, label),
            LabelAllocUndo::New(label) => {
                debug_assert_eq!(*next_label, label + 1);
                *next_label = label;
            }
        }
    }
}

fn apply_exact_ring_plan_in_place(
    graph: &PreparedSmilesGraphData,
    state: &mut RootedConnectedNonStereoWalkerStateData,
    action: &AfterAtomAction,
    ring_action_order: &[RingAction],
    child_order: &[usize],
) -> ExactRingPlanUndo {
    let action_stack_len = state.action_stack.len();
    let mut pending_opened = Vec::<(usize, PendingRing)>::new();
    let mut freed_labels = Vec::<usize>::new();
    let mut allocation_undos = Vec::<LabelAllocUndo>::new();
    let mut emitted_suffix_tokens = Vec::<EmittedToken>::new();

    for (index, ring_action) in ring_action_order.iter().enumerate() {
        let is_first = index == 0;
        match *ring_action {
            RingAction::Close(closure_idx) => {
                let closure = &action.closures_here[closure_idx];
                if closure.bond_token != BondTokenCode::Elided {
                    if !is_first {
                        emitted_suffix_tokens.push(EmittedToken::Literal(LiteralToken::Bond(
                            closure.bond_token,
                        )));
                    }
                    emitted_suffix_tokens.push(EmittedToken::RingLabel(closure.label));
                } else if !is_first {
                    emitted_suffix_tokens.push(EmittedToken::RingLabel(closure.label));
                }
                freed_labels.push(closure.label);
            }
            RingAction::Open(target_idx) => {
                let (label, undo) =
                    allocate_label_with_undo(&mut state.free_labels, &mut state.next_label);
                allocation_undos.push(undo);
                let ring = PendingRing {
                    label,
                    bond_token: bond_token_code(adjacent_bond_token(
                        graph,
                        action.atom_idx,
                        target_idx,
                        "in-place exact ring opening",
                    )),
                };
                add_pending(&mut state.pending, target_idx, ring);
                pending_opened.push((target_idx, ring));
                if !is_first {
                    emitted_suffix_tokens.push(EmittedToken::RingLabel(label));
                }
            }
        }
    }

    for &label in &freed_labels {
        insert_sorted(&mut state.free_labels, label);
    }

    push_child_actions(
        graph,
        &mut state.action_stack,
        action.atom_idx,
        child_order,
        false,
    );
    for token in emitted_suffix_tokens.into_iter().rev() {
        state.action_stack.push(Action::EmitToken(token));
    }

    ExactRingPlanUndo {
        pending_opened,
        freed_labels,
        allocation_undos,
        action_stack_len,
    }
}

fn undo_exact_ring_plan_in_place(
    state: &mut RootedConnectedNonStereoWalkerStateData,
    undo: &ExactRingPlanUndo,
) {
    state.action_stack.truncate(undo.action_stack_len);
    for &label in undo.freed_labels.iter().rev() {
        remove_sorted_label(&mut state.free_labels, label);
    }
    for (target_idx, ring) in undo.pending_opened.iter().rev() {
        remove_pending(&mut state.pending, *target_idx, ring);
    }
    undo_label_allocations(
        &mut state.free_labels,
        &mut state.next_label,
        &undo.allocation_undos,
    );
}

fn next_token_support_for_state(
    graph: &PreparedSmilesGraphData,
    state: &RootedConnectedNonStereoWalkerStateData,
) -> Vec<String> {
    let mut normalized = state.clone();
    normalize_state(&mut normalized);
    let action = match normalized.action_stack.last() {
        Some(action) => action,
        None => return Vec::new(),
    };

    match action {
        Action::EmitToken(token) => vec![emitted_token_string(token)],
        Action::EnterAtom(atom_idx) => vec![graph.atom_token(*atom_idx).to_owned()],
        Action::AfterAtom(action) => {
            if action.ring_action_count > 0 {
                let mut tokens = BTreeSet::new();
                for closure in &action.closures_here {
                    if closure.bond_token == BondTokenCode::Elided {
                        tokens.insert(ring_label_text(closure.label));
                    } else {
                        tokens.insert(bond_token_text(closure.bond_token).to_owned());
                    }
                }
                if action.opening_count > 0 {
                    tokens.insert(ring_label_text(next_open_label(&normalized)));
                }
                tokens.into_iter().collect()
            } else if let Some(child_idx) = action.linear_child_idx {
                vec![edge_prefix_or_atom(graph, action.atom_idx, child_idx)]
            } else if action.neighbor_groups.is_empty() {
                Vec::new()
            } else {
                vec!["(".to_owned()]
            }
        }
    }
}

fn successors_by_token(
    graph: &PreparedSmilesGraphData,
    state: &RootedConnectedNonStereoWalkerStateData,
) -> BTreeMap<String, Vec<RootedConnectedNonStereoWalkerStateData>> {
    let mut normalized = state.clone();
    normalize_state(&mut normalized);
    let action = match normalized.action_stack.last().cloned() {
        Some(action) => action,
        None => return BTreeMap::new(),
    };

    match action {
        Action::EmitToken(token) => {
            let mut successor = normalized;
            successor.action_stack.pop();
            append_emitted_token(&mut successor.prefix, &token);
            normalize_state(&mut successor);
            BTreeMap::from([(emitted_token_string(&token), vec![successor])])
        }
        Action::EnterAtom(atom_idx) => {
            let mut successor = normalized;
            successor.action_stack.pop();
            successor.prefix.push_str(graph.atom_token(atom_idx));
            consume_enter_atom(graph, &mut successor, atom_idx);
            normalize_state(&mut successor);
            BTreeMap::from([(graph.atom_token(atom_idx).to_owned(), vec![successor])])
        }
        Action::AfterAtom(action) => {
            let mut successors =
                BTreeMap::<String, Vec<RootedConnectedNonStereoWalkerStateData>>::new();

            if action.ring_action_count > 0 {
                for_each_cartesian_choice(&action.neighbor_groups, &mut |chosen_children| {
                    let opening_targets =
                        opening_targets_from_choices(&action.neighbor_groups, chosen_children);
                    let mut ring_actions = Vec::new();
                    for closure_idx in 0..action.closures_here.len() {
                        ring_actions.push(RingAction::Close(closure_idx));
                    }
                    for &target_idx in &opening_targets {
                        ring_actions.push(RingAction::Open(target_idx));
                    }

                    for_each_permutation_py_order_copy(&ring_actions, &mut |ring_action_order| {
                        let first_token = match ring_action_order[0] {
                            RingAction::Close(closure_idx) => {
                                let closure = &action.closures_here[closure_idx];
                                if closure.bond_token == BondTokenCode::Elided {
                                    ring_label_text(closure.label)
                                } else {
                                    bond_token_text(closure.bond_token).to_owned()
                                }
                            }
                            RingAction::Open(_) => ring_label_text(next_open_label(&normalized)),
                        };

                        for_each_permutation_py_order_copy(chosen_children, &mut |child_order| {
                            let mut successor = normalized.clone();
                            successor.action_stack.pop();
                            successor.prefix.push_str(&first_token);
                            apply_exact_ring_plan(
                                graph,
                                &mut successor,
                                &action,
                                ring_action_order,
                                child_order,
                            );
                            normalize_state(&mut successor);
                            successors
                                .entry(first_token.clone())
                                .or_default()
                                .push(successor);
                        });
                    });
                });
                successors
            } else if let Some(child_idx) = action.linear_child_idx {
                let token = edge_prefix_or_atom(graph, action.atom_idx, child_idx);
                let mut successor = normalized;
                successor.action_stack.pop();
                successor.prefix.push_str(&token);
                let edge_prefix =
                    adjacent_bond_token(graph, action.atom_idx, child_idx, "linear successor");
                if edge_prefix.is_empty() {
                    consume_enter_atom(graph, &mut successor, child_idx);
                } else {
                    successor.action_stack.push(Action::EnterAtom(child_idx));
                }
                normalize_state(&mut successor);
                BTreeMap::from([(token, vec![successor])])
            } else if action.neighbor_groups.is_empty() {
                BTreeMap::new()
            } else {
                let child_order_seed = action
                    .neighbor_groups
                    .iter()
                    .map(|group| group[0])
                    .collect::<Vec<_>>();
                for_each_permutation_py_order_copy(&child_order_seed, &mut |child_order| {
                    let mut successor = normalized.clone();
                    successor.action_stack.pop();
                    successor.prefix.push('(');
                    push_child_actions(
                        graph,
                        &mut successor.action_stack,
                        action.atom_idx,
                        child_order,
                        true,
                    );
                    normalize_state(&mut successor);
                    successors
                        .entry("(".to_owned())
                        .or_default()
                        .push(successor);
                });
                successors
            }
        }
    }
}

fn for_each_exact_successor_by_token_owned<F>(
    graph: &PreparedSmilesGraphData,
    mut state: RootedConnectedNonStereoWalkerStateData,
    mut emit: F,
) where
    F: FnMut(String, RootedConnectedNonStereoWalkerStateData),
{
    normalize_state(&mut state);
    let action = match state.action_stack.pop() {
        Some(action) => action,
        None => return,
    };

    match action {
        Action::EmitToken(token) => {
            append_emitted_token(&mut state.prefix, &token);
            normalize_state(&mut state);
            emit(emitted_token_string(&token), state);
        }
        Action::EnterAtom(atom_idx) => {
            let token = graph.atom_token(atom_idx).to_owned();
            state.prefix.push_str(graph.atom_token(atom_idx));
            consume_enter_atom(graph, &mut state, atom_idx);
            normalize_state(&mut state);
            emit(token, state);
        }
        Action::AfterAtom(action) => {
            if action.ring_action_count > 0 {
                for_each_cartesian_choice(&action.neighbor_groups, &mut |chosen_children| {
                    let opening_targets =
                        opening_targets_from_choices(&action.neighbor_groups, chosen_children);
                    let mut ring_actions = Vec::new();
                    for closure_idx in 0..action.closures_here.len() {
                        ring_actions.push(RingAction::Close(closure_idx));
                    }
                    for &target_idx in &opening_targets {
                        ring_actions.push(RingAction::Open(target_idx));
                    }

                    for_each_permutation_py_order_copy(&ring_actions, &mut |ring_action_order| {
                        let first_token = match ring_action_order[0] {
                            RingAction::Close(closure_idx) => {
                                let closure = &action.closures_here[closure_idx];
                                if closure.bond_token == BondTokenCode::Elided {
                                    ring_label_text(closure.label)
                                } else {
                                    bond_token_text(closure.bond_token).to_owned()
                                }
                            }
                            RingAction::Open(_) => ring_label_text(next_open_label(&state)),
                        };

                        for_each_permutation_py_order_copy(chosen_children, &mut |child_order| {
                            let prefix_len = state.prefix.len();
                            state.prefix.push_str(&first_token);
                            let undo = apply_exact_ring_plan_in_place(
                                graph,
                                &mut state,
                                &action,
                                ring_action_order,
                                child_order,
                            );
                            normalize_state(&mut state);
                            emit(first_token.clone(), state.clone());
                            undo_exact_ring_plan_in_place(&mut state, &undo);
                            state.prefix.truncate(prefix_len);
                        });
                    });
                });
            } else if let Some(child_idx) = action.linear_child_idx {
                let token = edge_prefix_or_atom(graph, action.atom_idx, child_idx);
                state.prefix.push_str(&token);
                let edge_prefix = adjacent_bond_token(
                    graph,
                    action.atom_idx,
                    child_idx,
                    "exact linear successor",
                );
                if edge_prefix.is_empty() {
                    consume_enter_atom(graph, &mut state, child_idx);
                } else {
                    state.action_stack.push(Action::EnterAtom(child_idx));
                }
                normalize_state(&mut state);
                emit(token, state);
            } else if action.neighbor_groups.is_empty() {
            } else {
                let child_order_seed = action
                    .neighbor_groups
                    .iter()
                    .map(|group| group[0])
                    .collect::<Vec<_>>();
                for_each_permutation_py_order_copy(&child_order_seed, &mut |child_order| {
                    let prefix_len = state.prefix.len();
                    let action_stack_len = state.action_stack.len();
                    state.prefix.push('(');
                    push_child_actions(
                        graph,
                        &mut state.action_stack,
                        action.atom_idx,
                        child_order,
                        true,
                    );
                    normalize_state(&mut state);
                    emit("(".to_owned(), state.clone());
                    state.action_stack.truncate(action_stack_len);
                    state.prefix.truncate(prefix_len);
                });
            }
        }
    }
}

fn advance_token_state(
    graph: &PreparedSmilesGraphData,
    state: &RootedConnectedNonStereoWalkerStateData,
    chosen_token: &str,
) -> PyResult<RootedConnectedNonStereoWalkerStateData> {
    let successors = successors_by_token(graph, state);
    let candidates = take_token_successors_or_err(successors, chosen_token)?;
    take_first_successor_or_err(candidates, "token advance")
}

fn choices_for_state(
    graph: &PreparedSmilesGraphData,
    state: &RootedConnectedNonStereoWalkerStateData,
) -> Vec<DecoderChoice<RootedConnectedNonStereoWalkerStateData>> {
    decoder_choices_from_token_successors(successors_by_token(graph, state))
}

fn next_choice_texts_for_state(
    graph: &PreparedSmilesGraphData,
    state: &RootedConnectedNonStereoWalkerStateData,
) -> Vec<String> {
    branch_choice_texts(&choices_for_state(graph, state))
}

fn advance_choice_state(
    graph: &PreparedSmilesGraphData,
    state: &RootedConnectedNonStereoWalkerStateData,
    chosen_idx: usize,
) -> PyResult<RootedConnectedNonStereoWalkerStateData> {
    let choices = choices_for_state(graph, state);
    take_only_successor_or_err(
        take_branch_choice_successors_or_err(choices, chosen_idx)?,
        "choice advance",
    )
}

fn frontier_next_token_support(
    graph: &PreparedSmilesGraphData,
    frontier: &[RootedConnectedNonStereoWalkerStateData],
) -> Vec<String> {
    frontier_grouped_transitions(graph, frontier)
        .into_iter()
        .map(|transition| transition.text)
        .collect()
}

fn frontier_choices(
    graph: &PreparedSmilesGraphData,
    frontier: &[RootedConnectedNonStereoWalkerStateData],
) -> Vec<DecoderChoice<RootedConnectedNonStereoWalkerStateData>> {
    let mut choices = Vec::new();
    for state in frontier {
        extend_decoder_choices_from_token_successors(
            &mut choices,
            successors_by_token(graph, state),
        );
    }
    choices
}

fn frontier_grouped_transitions(
    graph: &PreparedSmilesGraphData,
    frontier: &[RootedConnectedNonStereoWalkerStateData],
) -> Vec<GroupedTransition<RootedConnectedNonStereoWalkerStateData>> {
    group_decoder_choices(frontier_choices(graph, frontier), dedup_frontier)
}

fn frontier_prefix(frontier: &[RootedConnectedNonStereoWalkerStateData]) -> String {
    shared_frontier_prefix(frontier, |state| state.prefix.as_str())
}

fn exact_state_structural_cmp(
    left: &RootedConnectedNonStereoWalkerStateData,
    right: &RootedConnectedNonStereoWalkerStateData,
) -> std::cmp::Ordering {
    (
        &left.visited,
        left.visited_count,
        &left.pending,
        &left.free_labels,
        left.next_label,
        &left.action_stack,
    )
        .cmp(&(
            &right.visited,
            right.visited_count,
            &right.pending,
            &right.free_labels,
            right.next_label,
            &right.action_stack,
        ))
}

fn exact_state_same_structure(
    left: &RootedConnectedNonStereoWalkerStateData,
    right: &RootedConnectedNonStereoWalkerStateData,
) -> bool {
    left.visited_count == right.visited_count
        && left.next_label == right.next_label
        && left.visited == right.visited
        && left.pending == right.pending
        && left.free_labels == right.free_labels
        && left.action_stack == right.action_stack
}

fn dedup_exact_frontier(
    mut states: Vec<RootedConnectedNonStereoWalkerStateData>,
) -> Vec<RootedConnectedNonStereoWalkerStateData> {
    debug_assert!(
        states
            .first()
            .map(|first| states.iter().all(|state| state.prefix == first.prefix))
            .unwrap_or(true),
        "exact frontier token buckets must stay prefix-homogeneous"
    );
    states.sort_unstable_by(exact_state_structural_cmp);
    states.dedup_by(|left, right| exact_state_same_structure(left, right));
    states
}

fn frontier_is_terminal(
    graph: &PreparedSmilesGraphData,
    frontier: &[RootedConnectedNonStereoWalkerStateData],
) -> bool {
    frontier_next_token_support(graph, frontier).is_empty()
}

fn exact_frontier_successors(
    graph: &PreparedSmilesGraphData,
    frontier: Vec<RootedConnectedNonStereoWalkerStateData>,
) -> Vec<GroupedTransition<RootedConnectedNonStereoWalkerStateData>> {
    let mut choices = Vec::new();
    for state in frontier {
        for_each_exact_successor_by_token_owned(graph, state, |token, successor| {
            choices.push(DecoderChoice::single(token, successor));
        });
    }
    group_decoder_choices(choices, dedup_exact_frontier)
}

fn enumerate_support_from_frontier(
    graph: &PreparedSmilesGraphData,
    frontier: Vec<RootedConnectedNonStereoWalkerStateData>,
    out: &mut FxHashSet<String>,
) {
    let prefix = frontier_prefix(&frontier);
    let transitions = exact_frontier_successors(graph, frontier);
    if transitions.is_empty() {
        out.insert(prefix);
        return;
    }

    for transition in transitions {
        enumerate_support_from_frontier(graph, transition.successors, out);
    }
}

#[cfg(test)]
fn enumerate_support_from_state(
    graph: &PreparedSmilesGraphData,
    state: RootedConnectedNonStereoWalkerStateData,
    out: &mut BTreeSet<String>,
) {
    let successors = successors_by_token(graph, &state);
    if successors.is_empty() {
        let mut terminal = state;
        normalize_state(&mut terminal);
        if terminal.action_stack.is_empty()
            && terminal.visited_count == graph.atom_count()
            && terminal.pending.is_empty()
        {
            out.insert(terminal.prefix);
        }
        return;
    }

    for successor_group in successors.into_values() {
        for successor in successor_group {
            enumerate_support_from_state(graph, successor, out);
        }
    }
}

pub(crate) fn enumerate_rooted_connected_nonstereo_smiles_support(
    graph: &PreparedSmilesGraphData,
    root_idx: isize,
) -> PyResult<Vec<String>> {
    if graph.atom_count() == 0 {
        return Ok(vec![String::new()]);
    }
    let root_idx = validate_root_idx(graph, root_idx)?;
    let initial_state = initial_state_for_root(graph, root_idx);
    let mut results = FxHashSet::default();
    enumerate_support_from_frontier(graph, vec![initial_state], &mut results);
    let mut results = results.into_iter().collect::<Vec<_>>();
    results.sort_unstable();
    Ok(results)
}

#[pyclass(
    name = "RootedConnectedNonStereoWalkerState",
    module = "grimace._core",
    frozen
)]
pub struct PyRootedConnectedNonStereoWalkerState {
    data: RootedConnectedNonStereoWalkerStateData,
}

#[pymethods]
impl PyRootedConnectedNonStereoWalkerState {
    #[getter]
    fn prefix(&self) -> String {
        self.data.prefix.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "RootedConnectedNonStereoWalkerState(prefix={:?}, visited_count={}, pending_sites={}, next_label={}, stack_depth={})",
            self.data.prefix,
            self.data.visited_count,
            self.data.pending.len(),
            self.data.next_label,
            self.data.action_stack.len(),
        )
    }
}

#[pyclass(
    name = "RootedConnectedNonStereoWalker",
    module = "grimace._core",
    frozen
)]
pub struct PyRootedConnectedNonStereoWalker {
    graph: PreparedSmilesGraphData,
    root_idx: usize,
}

#[pymethods]
impl PyRootedConnectedNonStereoWalker {
    #[new]
    fn new(graph: &Bound<'_, PyAny>, root_idx: isize) -> PyResult<Self> {
        let graph = PreparedSmilesGraphData::from_any(graph)?;
        let root_idx = validate_root_idx(&graph, root_idx)?;
        Ok(Self { graph, root_idx })
    }

    #[getter]
    fn root_idx(&self) -> usize {
        self.root_idx
    }

    fn initial_state(&self) -> PyRootedConnectedNonStereoWalkerState {
        PyRootedConnectedNonStereoWalkerState {
            data: initial_state_for_root(&self.graph, self.root_idx),
        }
    }

    fn next_token_support(
        &self,
        state: &PyRootedConnectedNonStereoWalkerState,
    ) -> PyResult<Vec<String>> {
        validate_state_shape(&self.graph, &state.data)?;
        Ok(next_token_support_for_state(&self.graph, &state.data))
    }

    fn next_choice_texts(
        &self,
        state: &PyRootedConnectedNonStereoWalkerState,
    ) -> PyResult<Vec<String>> {
        validate_state_shape(&self.graph, &state.data)?;
        Ok(next_choice_texts_for_state(&self.graph, &state.data))
    }

    fn advance_token(
        &self,
        state: &PyRootedConnectedNonStereoWalkerState,
        chosen_token: &str,
    ) -> PyResult<PyRootedConnectedNonStereoWalkerState> {
        validate_state_shape(&self.graph, &state.data)?;
        Ok(PyRootedConnectedNonStereoWalkerState {
            data: advance_token_state(&self.graph, &state.data, chosen_token)?,
        })
    }

    fn advance_choice(
        &self,
        state: &PyRootedConnectedNonStereoWalkerState,
        chosen_idx: usize,
    ) -> PyResult<PyRootedConnectedNonStereoWalkerState> {
        validate_state_shape(&self.graph, &state.data)?;
        Ok(PyRootedConnectedNonStereoWalkerState {
            data: advance_choice_state(&self.graph, &state.data, chosen_idx)?,
        })
    }

    fn is_terminal(&self, state: &PyRootedConnectedNonStereoWalkerState) -> PyResult<bool> {
        validate_state_shape(&self.graph, &state.data)?;
        Ok(is_terminal_state(&state.data))
    }

    fn enumerate_support(&self) -> PyResult<Vec<String>> {
        enumerate_rooted_connected_nonstereo_smiles_support(&self.graph, self.root_idx as isize)
    }

    fn __repr__(&self) -> String {
        format!(
            "RootedConnectedNonStereoWalker(root_idx={}, atom_count={}, policy_name={:?}, policy_digest={:?})",
            self.root_idx,
            self.graph.atom_count(),
            self.graph.policy_name,
            self.graph.policy_digest,
        )
    }
}

#[pyclass(
    skip_from_py_object,
    name = "RootedConnectedNonStereoDecoder",
    module = "grimace._core"
)]
#[derive(Clone)]
pub struct PyRootedConnectedNonStereoDecoder {
    graph: Arc<PreparedSmilesGraphData>,
    frontier: Vec<RootedConnectedNonStereoWalkerStateData>,
    cached_choices: Option<Vec<DecoderChoice<RootedConnectedNonStereoWalkerStateData>>>,
}

impl PyRootedConnectedNonStereoDecoder {
    fn from_frontier(
        graph: Arc<PreparedSmilesGraphData>,
        frontier: Vec<RootedConnectedNonStereoWalkerStateData>,
    ) -> Self {
        Self {
            graph,
            frontier,
            cached_choices: None,
        }
    }

    fn cached_frontier_choices(
        &mut self,
    ) -> &[DecoderChoice<RootedConnectedNonStereoWalkerStateData>] {
        self.cached_choices
            .get_or_insert_with(|| frontier_choices(self.graph.as_ref(), &self.frontier))
            .as_slice()
    }

    fn take_frontier_choices(
        &mut self,
    ) -> Vec<DecoderChoice<RootedConnectedNonStereoWalkerStateData>> {
        self.cached_choices
            .take()
            .unwrap_or_else(|| frontier_choices(self.graph.as_ref(), &self.frontier))
    }
}

#[pymethods]
impl PyRootedConnectedNonStereoDecoder {
    #[new]
    fn new(graph: &Bound<'_, PyAny>, root_idx: isize) -> PyResult<Self> {
        let graph = Arc::new(PreparedSmilesGraphData::from_any(graph)?);
        let frontier = initial_frontier_for_root_spec(graph.as_ref(), root_idx)?;
        Ok(Self::from_frontier(graph, frontier))
    }

    fn next_token_support(&mut self) -> Vec<String> {
        token_support_from_choices(self.cached_frontier_choices())
    }

    fn advance_token(&mut self, chosen_token: &str) -> PyResult<()> {
        let choices = self.take_frontier_choices();
        self.frontier = take_token_support_successors_or_err(choices, chosen_token)?;
        Ok(())
    }

    fn next_choice_texts(&mut self) -> Vec<String> {
        branch_choice_texts(self.cached_frontier_choices())
    }

    fn advance_choice(&mut self, chosen_idx: usize) -> PyResult<()> {
        let choices = self.take_frontier_choices();
        self.frontier = take_branch_choice_successors_or_err(choices, chosen_idx)?;
        Ok(())
    }

    fn prefix(&self) -> String {
        frontier_prefix(&self.frontier)
    }

    fn cache_key(&self) -> String {
        format!("{:?}", self.frontier)
    }

    fn is_terminal(&self) -> bool {
        self.cached_choices
            .as_ref()
            .map(|choices| choices.is_empty())
            .unwrap_or_else(|| frontier_is_terminal(self.graph.as_ref(), &self.frontier))
    }

    fn copy(&self) -> Self {
        self.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "RootedConnectedNonStereoDecoder(prefix={:?}, frontier_size={}, atom_count={})",
            self.prefix(),
            self.frontier.len(),
            self.graph.atom_count(),
        )
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use pyo3::Python;

    use super::{
        advance_token_state, enumerate_rooted_connected_nonstereo_smiles_support,
        enumerate_support_from_state, for_each_cartesian_choice, initial_frontier_for_root_spec,
        initial_state_for_root, is_terminal_state, next_token_support_for_state,
        permutations_py_order, permutations_py_order_copy, validate_root_idx, Action, AtomBitSet,
    };
    use crate::prepared_graph::{
        PreparedSmilesGraphData, CONNECTED_NONSTEREO_SURFACE, PREPARED_SMILES_GRAPH_SCHEMA_VERSION,
    };

    #[derive(Clone, Debug, Default, PartialEq, Eq)]
    struct ExactFrontierStats {
        frontier_count: usize,
        terminal_frontier_count: usize,
        raw_successor_count: usize,
        deduped_successor_count: usize,
        duplicate_bucket_count: usize,
        max_frontier_width: usize,
        max_raw_bucket_size: usize,
        max_deduped_bucket_size: usize,
    }

    fn nonstereo_support_set(graph: &PreparedSmilesGraphData, root_idx: isize) -> BTreeSet<String> {
        enumerate_rooted_connected_nonstereo_smiles_support(graph, root_idx)
            .expect("nonstereo enumeration should succeed")
            .into_iter()
            .collect()
    }

    #[test]
    fn atom_bit_set_tracks_membership_without_cross_word_leakage() {
        let mut visited = AtomBitSet::new(130);
        assert_eq!(130, visited.len());
        for atom_idx in [0, 1, 63, 64, 65, 129] {
            assert!(!visited.contains(atom_idx));
            visited.insert(atom_idx);
            assert!(visited.contains(atom_idx));
        }
        assert!(!visited.contains(2));
        assert!(!visited.contains(62));
        assert!(!visited.contains(66));
        assert!(!visited.contains(128));
    }

    #[test]
    fn atom_bit_set_clone_is_independent_after_insert() {
        let mut left = AtomBitSet::new(8);
        left.insert(3);
        let mut right = left.clone();
        right.insert(4);

        assert!(left.contains(3));
        assert!(!left.contains(4));
        assert!(right.contains(3));
        assert!(right.contains(4));
        assert_ne!(left, right);
    }

    #[test]
    fn atom_bit_set_order_matches_logical_bool_vector_order() {
        fn with_visited(atom_count: usize, atom_idx: usize) -> AtomBitSet {
            let mut visited = AtomBitSet::new(atom_count);
            visited.insert(atom_idx);
            visited
        }

        assert_eq!(
            vec![true, false].cmp(&vec![false, true]),
            with_visited(2, 0).cmp(&with_visited(2, 1))
        );
        assert_eq!(
            {
                let mut left = vec![false; 130];
                left[63] = true;
                let mut right = vec![false; 130];
                right[64] = true;
                left.cmp(&right)
            },
            with_visited(130, 63).cmp(&with_visited(130, 64))
        );
    }

    fn collect_exact_frontier_stats(
        graph: &PreparedSmilesGraphData,
        frontier: Vec<super::RootedConnectedNonStereoWalkerStateData>,
        stats: &mut ExactFrontierStats,
    ) {
        stats.frontier_count += 1;
        stats.max_frontier_width = stats.max_frontier_width.max(frontier.len());

        let mut transitions = rustc_hash::FxHashMap::<
            String,
            Vec<super::RootedConnectedNonStereoWalkerStateData>,
        >::default();
        for state in frontier {
            super::for_each_exact_successor_by_token_owned(graph, state, |token, successor| {
                transitions.entry(token).or_default().push(successor);
            });
        }

        if transitions.is_empty() {
            stats.terminal_frontier_count += 1;
            return;
        }

        for mut states in transitions.into_values() {
            let raw_count = states.len();
            stats.raw_successor_count += raw_count;
            stats.max_raw_bucket_size = stats.max_raw_bucket_size.max(raw_count);

            states.sort_unstable();
            states.dedup();

            let deduped_count = states.len();
            stats.deduped_successor_count += deduped_count;
            stats.max_deduped_bucket_size = stats.max_deduped_bucket_size.max(deduped_count);
            if raw_count > deduped_count {
                stats.duplicate_bucket_count += 1;
            }

            collect_exact_frontier_stats(graph, states, stats);
        }
    }

    fn linear_ccc_graph() -> PreparedSmilesGraphData {
        PreparedSmilesGraphData {
            schema_version: PREPARED_SMILES_GRAPH_SCHEMA_VERSION,
            surface_kind: CONNECTED_NONSTEREO_SURFACE.to_owned(),
            policy_name: "test_policy".to_owned(),
            policy_digest: "deadbeef".to_owned(),
            rdkit_version: "2025.09.6".to_owned(),
            identity_smiles: "CCC".to_owned(),
            atom_count: 3,
            bond_count: 2,
            atom_atomic_numbers: vec![6, 6, 6],
            atom_is_aromatic: vec![false, false, false],
            atom_isotopes: vec![0, 0, 0],
            atom_formal_charges: vec![0, 0, 0],
            atom_total_hs: vec![3, 2, 3],
            atom_radical_electrons: vec![0, 0, 0],
            atom_map_numbers: vec![0, 0, 0],
            atom_tokens: vec!["C".to_owned(), "C".to_owned(), "C".to_owned()],
            neighbors: vec![vec![1], vec![0, 2], vec![1]],
            neighbor_bond_tokens: vec![
                vec!["".to_owned()],
                vec!["".to_owned(), "".to_owned()],
                vec!["".to_owned()],
            ],
            bond_pairs: vec![(0, 1), (1, 2)],
            bond_kinds: vec!["SINGLE".to_owned(), "SINGLE".to_owned()],
            writer_do_isomeric_smiles: true,
            writer_kekule_smiles: false,
            writer_all_bonds_explicit: false,
            writer_all_hs_explicit: false,
            writer_ignore_atom_map_numbers: false,
            identity_parse_with_rdkit: true,
            identity_canonical: true,
            identity_do_isomeric_smiles: true,
            identity_kekule_smiles: false,
            identity_rooted_at_atom: -1,
            identity_all_bonds_explicit: false,
            identity_all_hs_explicit: false,
            identity_do_random: false,
            identity_ignore_atom_map_numbers: false,
            atom_chiral_tags: Vec::new(),
            atom_stereo_neighbor_orders: Vec::new(),
            atom_explicit_h_counts: Vec::new(),
            atom_implicit_h_counts: Vec::new(),
            bond_stereo_kinds: Vec::new(),
            bond_stereo_atoms: Vec::new(),
            bond_dirs: Vec::new(),
            bond_begin_atom_indices: Vec::new(),
            bond_end_atom_indices: Vec::new(),
        }
    }

    fn cyclopropane_graph() -> PreparedSmilesGraphData {
        PreparedSmilesGraphData {
            schema_version: PREPARED_SMILES_GRAPH_SCHEMA_VERSION,
            surface_kind: CONNECTED_NONSTEREO_SURFACE.to_owned(),
            policy_name: "test_policy".to_owned(),
            policy_digest: "deadbeef".to_owned(),
            rdkit_version: "2025.09.6".to_owned(),
            identity_smiles: "C1CC1".to_owned(),
            atom_count: 3,
            bond_count: 3,
            atom_atomic_numbers: vec![6, 6, 6],
            atom_is_aromatic: vec![false, false, false],
            atom_isotopes: vec![0, 0, 0],
            atom_formal_charges: vec![0, 0, 0],
            atom_total_hs: vec![2, 2, 2],
            atom_radical_electrons: vec![0, 0, 0],
            atom_map_numbers: vec![0, 0, 0],
            atom_tokens: vec!["C".to_owned(), "C".to_owned(), "C".to_owned()],
            neighbors: vec![vec![1, 2], vec![0, 2], vec![0, 1]],
            neighbor_bond_tokens: vec![
                vec!["".to_owned(), "".to_owned()],
                vec!["".to_owned(), "".to_owned()],
                vec!["".to_owned(), "".to_owned()],
            ],
            bond_pairs: vec![(0, 1), (0, 2), (1, 2)],
            bond_kinds: vec![
                "SINGLE".to_owned(),
                "SINGLE".to_owned(),
                "SINGLE".to_owned(),
            ],
            writer_do_isomeric_smiles: true,
            writer_kekule_smiles: false,
            writer_all_bonds_explicit: false,
            writer_all_hs_explicit: false,
            writer_ignore_atom_map_numbers: false,
            identity_parse_with_rdkit: true,
            identity_canonical: true,
            identity_do_isomeric_smiles: true,
            identity_kekule_smiles: false,
            identity_rooted_at_atom: -1,
            identity_all_bonds_explicit: false,
            identity_all_hs_explicit: false,
            identity_do_random: false,
            identity_ignore_atom_map_numbers: false,
            atom_chiral_tags: Vec::new(),
            atom_stereo_neighbor_orders: Vec::new(),
            atom_explicit_h_counts: Vec::new(),
            atom_implicit_h_counts: Vec::new(),
            bond_stereo_kinds: Vec::new(),
            bond_stereo_atoms: Vec::new(),
            bond_dirs: Vec::new(),
            bond_begin_atom_indices: Vec::new(),
            bond_end_atom_indices: Vec::new(),
        }
    }

    fn titanium_dioxide_graph() -> PreparedSmilesGraphData {
        PreparedSmilesGraphData {
            schema_version: PREPARED_SMILES_GRAPH_SCHEMA_VERSION,
            surface_kind: CONNECTED_NONSTEREO_SURFACE.to_owned(),
            policy_name: "test_policy".to_owned(),
            policy_digest: "deadbeef".to_owned(),
            rdkit_version: "2025.09.6".to_owned(),
            identity_smiles: "[O]=[Ti]=[O]".to_owned(),
            atom_count: 3,
            bond_count: 2,
            atom_atomic_numbers: vec![8, 22, 8],
            atom_is_aromatic: vec![false, false, false],
            atom_isotopes: vec![0, 0, 0],
            atom_formal_charges: vec![0, 0, 0],
            atom_total_hs: vec![0, 0, 0],
            atom_radical_electrons: vec![0, 0, 0],
            atom_map_numbers: vec![0, 0, 0],
            atom_tokens: vec!["[O]".to_owned(), "[Ti]".to_owned(), "[O]".to_owned()],
            neighbors: vec![vec![1], vec![0, 2], vec![1]],
            neighbor_bond_tokens: vec![
                vec!["=".to_owned()],
                vec!["=".to_owned(), "=".to_owned()],
                vec!["=".to_owned()],
            ],
            bond_pairs: vec![(0, 1), (1, 2)],
            bond_kinds: vec!["DOUBLE".to_owned(), "DOUBLE".to_owned()],
            writer_do_isomeric_smiles: true,
            writer_kekule_smiles: false,
            writer_all_bonds_explicit: false,
            writer_all_hs_explicit: false,
            writer_ignore_atom_map_numbers: false,
            identity_parse_with_rdkit: true,
            identity_canonical: true,
            identity_do_isomeric_smiles: true,
            identity_kekule_smiles: false,
            identity_rooted_at_atom: -1,
            identity_all_bonds_explicit: false,
            identity_all_hs_explicit: false,
            identity_do_random: false,
            identity_ignore_atom_map_numbers: false,
            atom_chiral_tags: Vec::new(),
            atom_stereo_neighbor_orders: Vec::new(),
            atom_explicit_h_counts: Vec::new(),
            atom_implicit_h_counts: Vec::new(),
            bond_stereo_kinds: Vec::new(),
            bond_stereo_atoms: Vec::new(),
            bond_dirs: Vec::new(),
            bond_begin_atom_indices: Vec::new(),
            bond_end_atom_indices: Vec::new(),
        }
    }

    fn toluene_graph() -> PreparedSmilesGraphData {
        PreparedSmilesGraphData {
            schema_version: PREPARED_SMILES_GRAPH_SCHEMA_VERSION,
            surface_kind: CONNECTED_NONSTEREO_SURFACE.to_owned(),
            policy_name: "test_policy".to_owned(),
            policy_digest: "deadbeef".to_owned(),
            rdkit_version: "2026.03.1".to_owned(),
            identity_smiles: "Cc1ccccc1".to_owned(),
            atom_count: 7,
            bond_count: 7,
            atom_atomic_numbers: vec![6, 6, 6, 6, 6, 6, 6],
            atom_is_aromatic: vec![false, true, true, true, true, true, true],
            atom_isotopes: vec![0; 7],
            atom_formal_charges: vec![0; 7],
            atom_total_hs: vec![3, 0, 1, 1, 1, 1, 1],
            atom_radical_electrons: vec![0; 7],
            atom_map_numbers: vec![0; 7],
            atom_tokens: vec![
                "C".to_owned(),
                "c".to_owned(),
                "c".to_owned(),
                "c".to_owned(),
                "c".to_owned(),
                "c".to_owned(),
                "c".to_owned(),
            ],
            neighbors: vec![
                vec![1],
                vec![0, 2, 6],
                vec![1, 3],
                vec![2, 4],
                vec![3, 5],
                vec![4, 6],
                vec![1, 5],
            ],
            neighbor_bond_tokens: vec![
                vec!["".to_owned()],
                vec!["".to_owned(), "".to_owned(), "".to_owned()],
                vec!["".to_owned(), "".to_owned()],
                vec!["".to_owned(), "".to_owned()],
                vec!["".to_owned(), "".to_owned()],
                vec!["".to_owned(), "".to_owned()],
                vec!["".to_owned(), "".to_owned()],
            ],
            bond_pairs: vec![(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (1, 6)],
            bond_kinds: vec![
                "SINGLE".to_owned(),
                "AROMATIC".to_owned(),
                "AROMATIC".to_owned(),
                "AROMATIC".to_owned(),
                "AROMATIC".to_owned(),
                "AROMATIC".to_owned(),
                "AROMATIC".to_owned(),
            ],
            writer_do_isomeric_smiles: true,
            writer_kekule_smiles: false,
            writer_all_bonds_explicit: false,
            writer_all_hs_explicit: false,
            writer_ignore_atom_map_numbers: false,
            identity_parse_with_rdkit: true,
            identity_canonical: true,
            identity_do_isomeric_smiles: true,
            identity_kekule_smiles: false,
            identity_rooted_at_atom: -1,
            identity_all_bonds_explicit: false,
            identity_all_hs_explicit: false,
            identity_do_random: false,
            identity_ignore_atom_map_numbers: false,
            atom_chiral_tags: Vec::new(),
            atom_stereo_neighbor_orders: Vec::new(),
            atom_explicit_h_counts: Vec::new(),
            atom_implicit_h_counts: Vec::new(),
            bond_stereo_kinds: Vec::new(),
            bond_stereo_atoms: Vec::new(),
            bond_dirs: Vec::new(),
            bond_begin_atom_indices: Vec::new(),
            bond_end_atom_indices: Vec::new(),
        }
    }

    #[test]
    fn permutation_order_matches_python_style() {
        let perms = permutations_py_order(&[1usize, 2usize, 3usize]);
        assert_eq!(
            vec![
                vec![1, 2, 3],
                vec![1, 3, 2],
                vec![2, 1, 3],
                vec![2, 3, 1],
                vec![3, 1, 2],
                vec![3, 2, 1],
            ],
            perms
        );
    }

    #[test]
    fn copy_permutation_order_matches_generic_order() {
        assert_eq!(
            permutations_py_order_copy(&[1usize, 2usize, 3usize]),
            permutations_py_order(&[1usize, 2usize, 3usize])
        );
        assert_eq!(
            permutations_py_order_copy::<usize>(&[]),
            vec![Vec::<usize>::new()]
        );
    }

    #[test]
    fn empty_cartesian_choice_invokes_once() {
        let mut seen = Vec::new();
        for_each_cartesian_choice(&[], &mut |choice| seen.push(choice.to_vec()));
        assert_eq!(seen, vec![Vec::<usize>::new()]);
    }

    #[test]
    fn root_validation_rejects_out_of_range_indices() {
        let graph = linear_ccc_graph();
        assert!(validate_root_idx(&graph, -1).is_err());
        assert!(validate_root_idx(&graph, 3).is_err());
    }

    #[test]
    fn all_roots_initial_frontier_keeps_one_empty_state_per_atom() {
        let graph = linear_ccc_graph();
        let frontier =
            initial_frontier_for_root_spec(&graph, -1).expect("all-roots frontier should build");

        assert_eq!(graph.atom_count(), frontier.len());
        for (root_idx, state) in frontier.iter().enumerate() {
            assert_eq!(graph.atom_count(), state.visited.len());
            assert_eq!(0, state.visited_count);
            for atom_idx in 0..graph.atom_count() {
                assert!(!state.visited.contains(atom_idx));
            }
            assert_eq!(
                &[Action::EnterAtom(root_idx)],
                state.action_stack.as_slice()
            );
        }
    }

    #[test]
    fn linear_chain_support_matches_expected() {
        let graph = linear_ccc_graph();
        let support = nonstereo_support_set(&graph, 1);
        assert_eq!(BTreeSet::from(["C(C)C".to_owned()]), support);
    }

    #[test]
    fn simple_ring_support_matches_expected() {
        let graph = cyclopropane_graph();
        let support = nonstereo_support_set(&graph, 0);
        assert_eq!(BTreeSet::from(["C1CC1".to_owned()]), support);
    }

    #[test]
    fn awkward_metal_case_support_matches_expected() {
        let graph = titanium_dioxide_graph();
        let support_root_0 = nonstereo_support_set(&graph, 0);
        let support_root_1 = nonstereo_support_set(&graph, 1);

        assert_eq!(BTreeSet::from(["[O]=[Ti]=[O]".to_owned()]), support_root_0);
        assert_eq!(
            BTreeSet::from(["[Ti](=[O])=[O]".to_owned()]),
            support_root_1
        );
    }

    #[test]
    fn direct_enumerator_matches_frontier_decoder_support() {
        let graph = titanium_dioxide_graph();
        let initial_state = initial_state_for_root(&graph, 1);

        let mut direct = BTreeSet::new();
        enumerate_support_from_state(&graph, initial_state, &mut direct);

        let frontier = nonstereo_support_set(&graph, 1);

        assert_eq!(direct, frontier);
    }

    #[test]
    fn initial_state_support_is_root_atom_token() {
        let graph = linear_ccc_graph();
        let state = initial_state_for_root(&graph, 1);
        assert_eq!(
            vec!["C".to_owned()],
            next_token_support_for_state(&graph, &state)
        );
    }

    #[test]
    fn walker_can_reach_expected_linear_chain_terminal_state() {
        let graph = linear_ccc_graph();
        let mut state = initial_state_for_root(&graph, 0);
        let mut prefix = String::new();

        while !is_terminal_state(&state) {
            let options = next_token_support_for_state(&graph, &state);
            assert_eq!(
                1,
                options.len(),
                "linear chain should have a single next token"
            );
            let chosen = options[0].clone();
            prefix.push_str(&chosen);
            state = advance_token_state(&graph, &state, &chosen)
                .expect("single-path advancement should succeed");
        }

        assert_eq!("CCC", prefix);
    }

    #[test]
    fn walker_rejects_invalid_token_with_choices() {
        let graph = linear_ccc_graph();
        let state = initial_state_for_root(&graph, 0);
        Python::initialize();
        let err =
            advance_token_state(&graph, &state, "(").expect_err("invalid token should be rejected");
        assert!(
            err.to_string().contains("choices=[\"C\"]"),
            "unexpected error: {:?}",
            err
        );
    }

    #[test]
    fn exact_frontier_stats_detect_duplicate_merging_on_toluene_ring_root() {
        let graph = toluene_graph();
        let initial_frontier = vec![initial_state_for_root(&graph, 1)];
        let mut stats = ExactFrontierStats::default();
        collect_exact_frontier_stats(&graph, initial_frontier, &mut stats);

        assert!(stats.raw_successor_count > stats.deduped_successor_count);
        assert!(stats.duplicate_bucket_count > 0);
    }
}
