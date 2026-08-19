//! Test-only connected non-stereo support oracle.
//!
//! This is a compact transcription of the established `main` reference
//! semantics: residual-neighbour grouping, one entry per group, arbitrary
//! atom-local ring-action order, and branches before one main continuation.
//! It deliberately does not use `WriterState`, the CSP, residual frames, or the
//! live South Star 2 transition implementation.

use std::collections::{BTreeMap, BTreeSet, VecDeque};

use super::{AtomId, PreparedConnectedNonStereo};

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct PendingRing {
    label: usize,
    bond_text: String,
}

#[derive(Clone, Debug)]
struct SearchResult {
    text: String,
    visited: BTreeSet<AtomId>,
    pending: BTreeMap<AtomId, Vec<PendingRing>>,
    free_labels: Vec<usize>,
    next_label: usize,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum RingAction {
    Close(usize),
    Open(AtomId),
}

pub(super) fn support(surface: &PreparedConnectedNonStereo) -> BTreeSet<String> {
    let graph = surface.molecule().graph();
    let mut support = BTreeSet::new();

    for root in graph.atom_ids() {
        for result in enumerate_from_atom(
            surface,
            root,
            BTreeSet::new(),
            BTreeMap::new(),
            Vec::new(),
            1,
        ) {
            if result.visited.len() == graph.atom_count() && result.pending.is_empty() {
                support.insert(result.text);
            }
        }
    }

    support
}

fn enumerate_from_atom(
    surface: &PreparedConnectedNonStereo,
    atom: AtomId,
    mut visited: BTreeSet<AtomId>,
    mut pending: BTreeMap<AtomId, Vec<PendingRing>>,
    free_labels: Vec<usize>,
    next_label: usize,
) -> Vec<SearchResult> {
    assert!(visited.insert(atom), "reference walk must enter each atom once");
    let closures_here = pending.remove(&atom).unwrap_or_default();
    let groups = ordered_neighbor_groups(surface, atom, &visited);
    let mut results = Vec::new();

    for chosen_children in cartesian_choices(&groups) {
        let chosen = chosen_children.iter().copied().collect::<BTreeSet<_>>();
        let opening_targets = groups
            .iter()
            .flatten()
            .copied()
            .filter(|target| !chosen.contains(target))
            .collect::<Vec<_>>();
        let ring_actions = closures_here
            .iter()
            .enumerate()
            .map(|(index, _)| RingAction::Close(index))
            .chain(opening_targets.iter().copied().map(RingAction::Open))
            .collect::<Vec<_>>();

        for ring_order in permutations(&ring_actions) {
            let mut current_pending = pending.clone();
            let mut current_free = free_labels.clone();
            let mut current_next = next_label;
            let mut ring_text = String::new();
            let mut freed_after_atom = Vec::new();

            for action in ring_order {
                match action {
                    RingAction::Close(index) => {
                        let closure = &closures_here[index];
                        ring_text.push_str(&closure.bond_text);
                        ring_text.push_str(&label_text(closure.label));
                        freed_after_atom.push(closure.label);
                    }
                    RingAction::Open(target) => {
                        let label = allocate_label(&mut current_free, &mut current_next);
                        ring_text.push_str(&label_text(label));
                        add_pending(
                            &mut current_pending,
                            target,
                            PendingRing {
                                label,
                                bond_text: bond_text_between(surface, atom, target).to_owned(),
                            },
                        );
                    }
                }
            }

            for label in freed_after_atom {
                insert_sorted(&mut current_free, label);
            }

            for child_order in permutations(&chosen_children) {
                results.extend(expand_children(
                    surface,
                    atom,
                    &child_order,
                    format!("{}{}", surface.atom_text(atom), ring_text),
                    visited.clone(),
                    current_pending.clone(),
                    current_free.clone(),
                    current_next,
                ));
            }
        }
    }

    results
}

fn expand_children(
    surface: &PreparedConnectedNonStereo,
    parent: AtomId,
    child_order: &[AtomId],
    prefix: String,
    visited: BTreeSet<AtomId>,
    pending: BTreeMap<AtomId, Vec<PendingRing>>,
    free_labels: Vec<usize>,
    next_label: usize,
) -> Vec<SearchResult> {
    let Some((&main_child, branch_children)) = child_order.split_last() else {
        return vec![SearchResult {
            text: prefix,
            visited,
            pending,
            free_labels,
            next_label,
        }];
    };

    let mut partials = vec![SearchResult {
        text: prefix,
        visited,
        pending,
        free_labels,
        next_label,
    }];

    for &branch_child in branch_children {
        let mut next_partials = Vec::new();
        for partial in partials {
            for branch in enumerate_from_atom(
                surface,
                branch_child,
                partial.visited.clone(),
                partial.pending.clone(),
                partial.free_labels.clone(),
                partial.next_label,
            ) {
                next_partials.push(SearchResult {
                    text: format!(
                        "{}({}{})",
                        partial.text,
                        bond_text_between(surface, parent, branch_child),
                        branch.text
                    ),
                    visited: branch.visited,
                    pending: branch.pending,
                    free_labels: branch.free_labels,
                    next_label: branch.next_label,
                });
            }
        }
        partials = next_partials;
    }

    let mut results = Vec::new();
    for partial in partials {
        for main in enumerate_from_atom(
            surface,
            main_child,
            partial.visited.clone(),
            partial.pending.clone(),
            partial.free_labels.clone(),
            partial.next_label,
        ) {
            results.push(SearchResult {
                text: format!(
                    "{}{}{}",
                    partial.text,
                    bond_text_between(surface, parent, main_child),
                    main.text
                ),
                visited: main.visited,
                pending: main.pending,
                free_labels: main.free_labels,
                next_label: main.next_label,
            });
        }
    }
    results
}

fn ordered_neighbor_groups(
    surface: &PreparedConnectedNonStereo,
    atom: AtomId,
    visited: &BTreeSet<AtomId>,
) -> Vec<Vec<AtomId>> {
    let graph = surface.molecule().graph();
    let mut remaining = graph
        .neighbors(atom)
        .expect("reference atom must belong to the prepared graph")
        .iter()
        .map(|incident| incident.atom())
        .filter(|neighbour| !visited.contains(neighbour))
        .collect::<BTreeSet<_>>();
    if remaining.is_empty() {
        return Vec::new();
    }

    let mut blocked = visited.clone();
    blocked.insert(atom);
    let mut groups = Vec::new();

    while let Some(seed) = remaining.iter().next().copied() {
        remaining.remove(&seed);
        let mut seen = BTreeSet::from([seed]);
        let mut queue = VecDeque::from([seed]);
        let mut component_min = seed;
        let mut group = vec![seed];

        while let Some(current) = queue.pop_front() {
            component_min = component_min.min(current);
            for incident in graph
                .neighbors(current)
                .expect("reference atom must belong to the prepared graph")
            {
                let neighbour = incident.atom();
                if blocked.contains(&neighbour) || !seen.insert(neighbour) {
                    continue;
                }
                if remaining.remove(&neighbour) {
                    group.push(neighbour);
                }
                queue.push_back(neighbour);
            }
        }

        group.sort_unstable();
        groups.push((component_min, group));
    }

    groups.sort_by_key(|(component_min, _)| *component_min);
    groups.into_iter().map(|(_, group)| group).collect()
}

fn bond_text_between(
    surface: &PreparedConnectedNonStereo,
    from: AtomId,
    to: AtomId,
) -> &'static str {
    let incident = surface
        .molecule()
        .graph()
        .neighbors(from)
        .expect("reference atom must belong to the prepared graph")
        .iter()
        .copied()
        .find(|incident| incident.atom() == to)
        .expect("reference child must be adjacent to its parent");
    surface.bond_text(incident.bond(), from)
}

fn cartesian_choices(groups: &[Vec<AtomId>]) -> Vec<Vec<AtomId>> {
    fn recurse(
        groups: &[Vec<AtomId>],
        index: usize,
        current: &mut Vec<AtomId>,
        output: &mut Vec<Vec<AtomId>>,
    ) {
        if index == groups.len() {
            output.push(current.clone());
            return;
        }
        for &choice in &groups[index] {
            current.push(choice);
            recurse(groups, index + 1, current, output);
            current.pop();
        }
    }

    let mut output = Vec::new();
    recurse(groups, 0, &mut Vec::with_capacity(groups.len()), &mut output);
    output
}

fn permutations<T: Copy>(items: &[T]) -> Vec<Vec<T>> {
    fn recurse<T: Copy>(
        items: &[T],
        used: &mut [bool],
        current: &mut Vec<T>,
        output: &mut Vec<Vec<T>>,
    ) {
        if current.len() == items.len() {
            output.push(current.clone());
            return;
        }
        for index in 0..items.len() {
            if used[index] {
                continue;
            }
            used[index] = true;
            current.push(items[index]);
            recurse(items, used, current, output);
            current.pop();
            used[index] = false;
        }
    }

    let mut output = Vec::new();
    recurse(
        items,
        &mut vec![false; items.len()],
        &mut Vec::with_capacity(items.len()),
        &mut output,
    );
    output
}

fn add_pending(
    pending: &mut BTreeMap<AtomId, Vec<PendingRing>>,
    target: AtomId,
    ring: PendingRing,
) {
    let rings = pending.entry(target).or_default();
    rings.push(ring);
    rings.sort();
}

fn allocate_label(free_labels: &mut Vec<usize>, next_label: &mut usize) -> usize {
    if free_labels.is_empty() {
        let label = *next_label;
        *next_label = next_label
            .checked_add(1)
            .expect("reference label space must not overflow");
        label
    } else {
        free_labels.remove(0)
    }
}

fn insert_sorted(labels: &mut Vec<usize>, label: usize) {
    let offset = labels.binary_search(&label).unwrap_or_else(|offset| offset);
    labels.insert(offset, label);
}

fn label_text(label: usize) -> String {
    if label < 10 {
        label.to_string()
    } else if label < 100 {
        format!("%{label}")
    } else {
        format!("%({label})")
    }
}
