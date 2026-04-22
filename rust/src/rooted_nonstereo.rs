use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};

use pyo3::exceptions::{PyIndexError, PyKeyError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyAny;

use crate::frontier::{
    choice_texts, extend_transitions, finalize_transitions,
    frontier_prefix as shared_frontier_prefix, grouped_choice_texts, take_choice_or_err,
    take_grouped_choices_or_err, DecoderChoice,
};
use crate::prepared_graph::PreparedSmilesGraphData;

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct PendingRing {
    label: usize,
    bond_token: String,
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
    EmitToken(String),
    EnterAtom(usize),
    AfterAtom(AfterAtomAction),
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct RootedConnectedNonStereoWalkerStateData {
    prefix: String,
    visited: Vec<bool>,
    visited_count: usize,
    pending: Vec<(usize, Vec<PendingRing>)>,
    free_labels: Vec<usize>,
    next_label: usize,
    action_stack: Vec<Action>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum RingAction {
    Close(usize),
    Open(usize),
}

fn ring_label_text(label: usize) -> String {
    if label < 10 {
        label.to_string()
    } else if label < 100 {
        format!("%{label}")
    } else {
        format!("%({label})")
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

fn add_pending(
    pending: &mut Vec<(usize, Vec<PendingRing>)>,
    target_atom: usize,
    ring: PendingRing,
) {
    let current = match pending.binary_search_by_key(&target_atom, |(atom_idx, _)| *atom_idx) {
        Ok(offset) => &mut pending[offset].1,
        Err(offset) => {
            pending.insert(offset, (target_atom, Vec::new()));
            &mut pending[offset].1
        }
    };
    let insert_at = current
        .binary_search_by(|candidate| {
            (candidate.label, candidate.bond_token.as_str())
                .cmp(&(ring.label, ring.bond_token.as_str()))
        })
        .unwrap_or_else(|offset| offset);
    current.insert(insert_at, ring);
}

fn take_pending_for_atom(
    pending: &mut Vec<(usize, Vec<PendingRing>)>,
    atom_idx: usize,
) -> Vec<PendingRing> {
    match pending.binary_search_by_key(&atom_idx, |(candidate_idx, _)| *candidate_idx) {
        Ok(offset) => pending.remove(offset).1,
        Err(_) => Vec::new(),
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
    visited: &[bool],
) -> Vec<Vec<usize>> {
    let mut seeds = graph
        .neighbors_of(atom_idx)
        .iter()
        .copied()
        .filter(|&neighbor_idx| !visited[neighbor_idx])
        .collect::<Vec<_>>();
    if seeds.is_empty() {
        return Vec::new();
    }
    seeds.sort_unstable();
    if seeds.len() == 1 {
        return vec![seeds];
    }
    let seed_list = seeds.clone();

    let mut groups_with_mins = Vec::<(usize, Vec<usize>)>::new();
    let mut seen = vec![false; graph.atom_count()];
    let mut queue = VecDeque::new();
    for seed in seeds {
        if seen[seed] {
            continue;
        }
        seen[seed] = true;
        queue.push_back(seed);
        let mut component_min = seed;
        let mut group = vec![seed];

        while let Some(current) = queue.pop_front() {
            if current < component_min {
                component_min = current;
            }
            for &neighbor_idx in graph.neighbors_of(current) {
                if neighbor_idx == atom_idx || visited[neighbor_idx] || seen[neighbor_idx] {
                    continue;
                }
                seen[neighbor_idx] = true;
                if seed_list.binary_search(&neighbor_idx).is_ok() {
                    group.push(neighbor_idx);
                }
                queue.push_back(neighbor_idx);
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

    let mut current = Vec::new();
    recurse(groups, 0, &mut current, f);
    if groups.is_empty() {
        f(&[]);
    }
}

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
    if items.is_empty() {
        f(&[]);
    }
}

#[cfg(test)]
fn permutations_py_order<T: Clone>(items: &[T]) -> Vec<Vec<T>> {
    let mut out = Vec::new();
    for_each_permutation_py_order(items, &mut |perm| out.push(perm.to_vec()));
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
        visited: vec![false; graph.atom_count()],
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

fn next_open_label_token(state: &RootedConnectedNonStereoWalkerStateData) -> String {
    let label = state
        .free_labels
        .first()
        .copied()
        .unwrap_or(state.next_label);
    ring_label_text(label)
}

fn edge_prefix_or_atom(
    graph: &PreparedSmilesGraphData,
    parent_idx: usize,
    child_idx: usize,
) -> String {
    let token = graph
        .bond_token(parent_idx, child_idx)
        .expect("child should be adjacent");
    if token.is_empty() {
        graph.atom_token(child_idx).to_owned()
    } else {
        token.to_owned()
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

    let main_prefix = graph
        .bond_token(parent_idx, main_child)
        .expect("main child should be adjacent");
    action_stack.push(Action::EnterAtom(main_child));
    if !main_prefix.is_empty() {
        action_stack.push(Action::EmitToken(main_prefix.to_owned()));
    }

    if branch_children.is_empty() {
        return;
    }

    for &child_idx in branch_children[1..].iter().rev() {
        action_stack.push(Action::EmitToken(")".to_owned()));
        action_stack.push(Action::EnterAtom(child_idx));
        let edge_prefix = graph
            .bond_token(parent_idx, child_idx)
            .expect("branch child should be adjacent");
        if !edge_prefix.is_empty() {
            action_stack.push(Action::EmitToken(edge_prefix.to_owned()));
        }
        action_stack.push(Action::EmitToken("(".to_owned()));
    }

    let first_branch_child = branch_children[0];
    action_stack.push(Action::EmitToken(")".to_owned()));
    action_stack.push(Action::EnterAtom(first_branch_child));
    let first_edge_prefix = graph
        .bond_token(parent_idx, first_branch_child)
        .expect("first branch child should be adjacent");
    if !first_edge_prefix.is_empty() {
        action_stack.push(Action::EmitToken(first_edge_prefix.to_owned()));
    }
    if !first_branch_open_consumed {
        action_stack.push(Action::EmitToken("(".to_owned()));
    }
}

fn consume_enter_atom(
    graph: &PreparedSmilesGraphData,
    state: &mut RootedConnectedNonStereoWalkerStateData,
    atom_idx: usize,
) {
    debug_assert!(!state.visited[atom_idx]);
    state.visited[atom_idx] = true;
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
    let mut emitted_suffix_tokens = Vec::new();

    for (index, ring_action) in ring_action_order.iter().enumerate() {
        let is_first = index == 0;
        match *ring_action {
            RingAction::Close(closure_idx) => {
                let closure = &action.closures_here[closure_idx];
                if !closure.bond_token.is_empty() {
                    if !is_first {
                        emitted_suffix_tokens.push(closure.bond_token.clone());
                    }
                    emitted_suffix_tokens.push(ring_label_text(closure.label));
                } else if !is_first {
                    emitted_suffix_tokens.push(ring_label_text(closure.label));
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
                        bond_token: graph
                            .bond_token(action.atom_idx, target_idx)
                            .expect("opening target should be adjacent")
                            .to_owned(),
                    },
                );
                if !is_first {
                    emitted_suffix_tokens.push(ring_label_text(label));
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
        Action::EmitToken(token) => vec![token.clone()],
        Action::EnterAtom(atom_idx) => vec![graph.atom_token(*atom_idx).to_owned()],
        Action::AfterAtom(action) => {
            if action.ring_action_count > 0 {
                let mut tokens = BTreeSet::new();
                for closure in &action.closures_here {
                    if closure.bond_token.is_empty() {
                        tokens.insert(ring_label_text(closure.label));
                    } else {
                        tokens.insert(closure.bond_token.clone());
                    }
                }
                if action.opening_count > 0 {
                    tokens.insert(next_open_label_token(&normalized));
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
            successor.prefix.push_str(&token);
            normalize_state(&mut successor);
            BTreeMap::from([(token, vec![successor])])
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

                    for_each_permutation_py_order(&ring_actions, &mut |ring_action_order| {
                        let first_token = match ring_action_order[0] {
                            RingAction::Close(closure_idx) => {
                                let closure = &action.closures_here[closure_idx];
                                if closure.bond_token.is_empty() {
                                    ring_label_text(closure.label)
                                } else {
                                    closure.bond_token.clone()
                                }
                            }
                            RingAction::Open(_) => next_open_label_token(&normalized),
                        };

                        for_each_permutation_py_order(chosen_children, &mut |child_order| {
                            let mut successor = normalized.clone();
                            successor.action_stack.pop();
                            successor.prefix.push_str(&first_token);
                            apply_exact_ring_plan(
                                graph,
                                &mut successor,
                                &action,
                                &ring_action_order,
                                &child_order,
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
                let edge_prefix = graph
                    .bond_token(action.atom_idx, child_idx)
                    .expect("linear child should be adjacent");
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
                for_each_permutation_py_order(&child_order_seed, &mut |child_order| {
                    let mut successor = normalized.clone();
                    successor.action_stack.pop();
                    successor.prefix.push('(');
                    push_child_actions(
                        graph,
                        &mut successor.action_stack,
                        action.atom_idx,
                        &child_order,
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
            state.prefix.push_str(&token);
            normalize_state(&mut state);
            emit(token, state);
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

                    for_each_permutation_py_order(&ring_actions, &mut |ring_action_order| {
                        let first_token = match ring_action_order[0] {
                            RingAction::Close(closure_idx) => {
                                let closure = &action.closures_here[closure_idx];
                                if closure.bond_token.is_empty() {
                                    ring_label_text(closure.label)
                                } else {
                                    closure.bond_token.clone()
                                }
                            }
                            RingAction::Open(_) => next_open_label_token(&state),
                        };

                        for_each_permutation_py_order(chosen_children, &mut |child_order| {
                            let mut successor = state.clone();
                            successor.prefix.push_str(&first_token);
                            apply_exact_ring_plan(
                                graph,
                                &mut successor,
                                &action,
                                &ring_action_order,
                                &child_order,
                            );
                            normalize_state(&mut successor);
                            emit(first_token.clone(), successor);
                        });
                    });
                });
            } else if let Some(child_idx) = action.linear_child_idx {
                let token = edge_prefix_or_atom(graph, action.atom_idx, child_idx);
                state.prefix.push_str(&token);
                let edge_prefix = graph
                    .bond_token(action.atom_idx, child_idx)
                    .expect("linear child should be adjacent");
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
                for_each_permutation_py_order(&child_order_seed, &mut |child_order| {
                    let mut successor = state.clone();
                    successor.prefix.push('(');
                    push_child_actions(
                        graph,
                        &mut successor.action_stack,
                        action.atom_idx,
                        &child_order,
                        true,
                    );
                    normalize_state(&mut successor);
                    emit("(".to_owned(), successor);
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
    let mut successors = successors_by_token(graph, state);
    let candidates = successors.remove(chosen_token).ok_or_else(|| {
        let available = successors.keys().cloned().collect::<Vec<_>>();
        PyKeyError::new_err(format!(
            "Token {chosen_token:?} is not available; choices={available:?}"
        ))
    })?;
    Ok(candidates
        .into_iter()
        .next()
        .expect("chosen token should have at least one successor"))
}

fn choices_for_state(
    graph: &PreparedSmilesGraphData,
    state: &RootedConnectedNonStereoWalkerStateData,
) -> Vec<DecoderChoice<RootedConnectedNonStereoWalkerStateData>> {
    let mut choices = Vec::new();
    for (token, successors) in successors_by_token(graph, state) {
        for successor in successors {
            choices.push(DecoderChoice {
                text: token.clone(),
                next_frontier: vec![successor],
            });
        }
    }
    choices
}

fn next_choice_texts_for_state(
    graph: &PreparedSmilesGraphData,
    state: &RootedConnectedNonStereoWalkerStateData,
) -> Vec<String> {
    choice_texts(&choices_for_state(graph, state))
}

fn advance_choice_state(
    graph: &PreparedSmilesGraphData,
    state: &RootedConnectedNonStereoWalkerStateData,
    chosen_idx: usize,
) -> PyResult<RootedConnectedNonStereoWalkerStateData> {
    let mut choices = choices_for_state(graph, state);
    Ok(take_choice_or_err(&mut choices, chosen_idx)?
        .into_iter()
        .next()
        .expect("choice should advance to exactly one successor state"))
}

fn frontier_next_token_support(
    graph: &PreparedSmilesGraphData,
    frontier: &[RootedConnectedNonStereoWalkerStateData],
) -> Vec<String> {
    frontier_transitions(graph, frontier).into_keys().collect()
}

fn frontier_transitions(
    graph: &PreparedSmilesGraphData,
    frontier: &[RootedConnectedNonStereoWalkerStateData],
) -> BTreeMap<String, Vec<RootedConnectedNonStereoWalkerStateData>> {
    let mut transitions =
        BTreeMap::<String, BTreeSet<RootedConnectedNonStereoWalkerStateData>>::new();
    for state in frontier {
        extend_transitions(&mut transitions, successors_by_token(graph, state));
    }
    finalize_transitions(transitions)
}

fn frontier_choices(
    graph: &PreparedSmilesGraphData,
    frontier: &[RootedConnectedNonStereoWalkerStateData],
) -> Vec<DecoderChoice<RootedConnectedNonStereoWalkerStateData>> {
    let mut choices = Vec::new();
    for state in frontier {
        for (token, successors) in successors_by_token(graph, state) {
            for successor in successors {
                choices.push(DecoderChoice {
                    text: token.clone(),
                    next_frontier: vec![successor],
                });
            }
        }
    }
    choices
}

fn frontier_prefix(frontier: &[RootedConnectedNonStereoWalkerStateData]) -> String {
    shared_frontier_prefix(frontier, |state| state.prefix.as_str())
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
) -> Vec<Vec<RootedConnectedNonStereoWalkerStateData>> {
    let mut transitions = HashMap::<String, Vec<RootedConnectedNonStereoWalkerStateData>>::new();
    for state in frontier {
        for_each_exact_successor_by_token_owned(graph, state, |token, successor| {
            transitions.entry(token).or_default().push(successor);
        });
    }
    transitions
        .into_values()
        .map(|mut states| {
            states.sort_unstable();
            states.dedup();
            states
        })
        .collect()
}

fn enumerate_support_from_frontier(
    graph: &PreparedSmilesGraphData,
    frontier: Vec<RootedConnectedNonStereoWalkerStateData>,
    out: &mut HashSet<String>,
) {
    let prefix = frontier_prefix(&frontier);
    let transitions = exact_frontier_successors(graph, frontier);
    if transitions.is_empty() {
        out.insert(prefix);
        return;
    }

    for next_frontier in transitions {
        enumerate_support_from_frontier(graph, next_frontier, out);
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
    let mut results = HashSet::new();
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
    graph: PreparedSmilesGraphData,
    frontier: Vec<RootedConnectedNonStereoWalkerStateData>,
    cached_choices: Option<Vec<DecoderChoice<RootedConnectedNonStereoWalkerStateData>>>,
}

#[pymethods]
impl PyRootedConnectedNonStereoDecoder {
    #[new]
    fn new(graph: &Bound<'_, PyAny>, root_idx: isize) -> PyResult<Self> {
        let graph = PreparedSmilesGraphData::from_any(graph)?;
        let root_idx = validate_root_idx(&graph, root_idx)?;
        Ok(Self {
            frontier: vec![initial_state_for_root(&graph, root_idx)],
            graph,
            cached_choices: None,
        })
    }

    fn next_token_support(&mut self) -> Vec<String> {
        if self.cached_choices.is_none() {
            self.cached_choices = Some(frontier_choices(&self.graph, &self.frontier));
        }
        grouped_choice_texts(
            self.cached_choices
                .as_ref()
                .expect("cache should be populated"),
        )
    }

    fn advance_token(&mut self, chosen_token: &str) -> PyResult<()> {
        let choices = self
            .cached_choices
            .take()
            .unwrap_or_else(|| frontier_choices(&self.graph, &self.frontier));
        self.frontier = take_grouped_choices_or_err(choices, chosen_token)?;
        Ok(())
    }

    fn next_choice_texts(&mut self) -> Vec<String> {
        if self.cached_choices.is_none() {
            self.cached_choices = Some(frontier_choices(&self.graph, &self.frontier));
        }
        choice_texts(
            self.cached_choices
                .as_ref()
                .expect("cache should be populated"),
        )
    }

    fn advance_choice(&mut self, chosen_idx: usize) -> PyResult<()> {
        let mut choices = self
            .cached_choices
            .take()
            .unwrap_or_else(|| frontier_choices(&self.graph, &self.frontier));
        self.frontier = take_choice_or_err(&mut choices, chosen_idx)?;
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
            .unwrap_or_else(|| frontier_is_terminal(&self.graph, &self.frontier))
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
        enumerate_support_from_state, initial_state_for_root, is_terminal_state,
        next_token_support_for_state, permutations_py_order, validate_root_idx,
    };
    use crate::prepared_graph::{
        PreparedSmilesGraphData, CONNECTED_NONSTEREO_SURFACE, PREPARED_SMILES_GRAPH_SCHEMA_VERSION,
    };

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
    fn root_validation_rejects_out_of_range_indices() {
        let graph = linear_ccc_graph();
        assert!(validate_root_idx(&graph, -1).is_err());
        assert!(validate_root_idx(&graph, 3).is_err());
    }

    #[test]
    fn linear_chain_support_matches_expected() {
        let graph = linear_ccc_graph();
        let support = enumerate_rooted_connected_nonstereo_smiles_support(&graph, 1)
            .expect("enumeration should succeed")
            .into_iter()
            .collect::<BTreeSet<_>>();
        assert_eq!(BTreeSet::from(["C(C)C".to_owned()]), support);
    }

    #[test]
    fn simple_ring_support_matches_expected() {
        let graph = cyclopropane_graph();
        let support = enumerate_rooted_connected_nonstereo_smiles_support(&graph, 0)
            .expect("enumeration should succeed")
            .into_iter()
            .collect::<BTreeSet<_>>();
        assert_eq!(BTreeSet::from(["C1CC1".to_owned()]), support);
    }

    #[test]
    fn awkward_metal_case_support_matches_expected() {
        let graph = titanium_dioxide_graph();
        let support_root_0 = enumerate_rooted_connected_nonstereo_smiles_support(&graph, 0)
            .expect("enumeration should succeed")
            .into_iter()
            .collect::<BTreeSet<_>>();
        let support_root_1 = enumerate_rooted_connected_nonstereo_smiles_support(&graph, 1)
            .expect("enumeration should succeed")
            .into_iter()
            .collect::<BTreeSet<_>>();

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

        let frontier = enumerate_rooted_connected_nonstereo_smiles_support(&graph, 1)
            .expect("frontier enumeration should succeed")
            .into_iter()
            .collect::<BTreeSet<_>>();

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
                .expect("advancing along the only path should succeed");
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
}
