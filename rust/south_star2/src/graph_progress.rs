//! Compact graph-writing progress shared by future traversal states.

use crate::ids::{AtomId, BondId};
use crate::prepared::{AdjacentBond, PreparedGraph};

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct DenseSet {
    universe_len: usize,
    marked_count: usize,
    words: Box<[u64]>,
}

impl DenseSet {
    fn new(universe_len: usize) -> Self {
        let word_count = universe_len / u64::BITS as usize
            + usize::from(universe_len % u64::BITS as usize != 0);
        Self {
            universe_len,
            marked_count: 0,
            words: vec![0; word_count].into_boxed_slice(),
        }
    }

    fn contains(&self, index: usize) -> bool {
        let (word, mask) = self.location(index);
        self.words[word] & mask != 0
    }

    fn insert(&mut self, index: usize) -> bool {
        let (word, mask) = self.location(index);
        if self.words[word] & mask != 0 {
            return false;
        }
        self.words[word] |= mask;
        self.marked_count += 1;
        true
    }

    const fn marked_count(&self) -> usize {
        self.marked_count
    }

    const fn is_complete(&self) -> bool {
        self.marked_count == self.universe_len
    }

    fn location(&self, index: usize) -> (usize, u64) {
        assert!(
            index < self.universe_len,
            "prepared identifier must fit the graph progress universe"
        );
        let word = index / u64::BITS as usize;
        let bit = index % u64::BITS as usize;
        (word, 1_u64 << bit)
    }
}

/// Graph facts that change as a SMILES walk represents atoms and bonds.
///
/// This deliberately excludes the active atom, branch stack, ring labels, and
/// pending emissions. Those belong to the traversal state that will compose
/// this progress record; they are not guessed here.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct GraphProgress {
    visited_atoms: DenseSet,
    written_bonds: DenseSet,
}

impl GraphProgress {
    pub(crate) fn new(graph: &PreparedGraph) -> Self {
        Self {
            visited_atoms: DenseSet::new(graph.atom_count()),
            written_bonds: DenseSet::new(graph.bond_count()),
        }
    }

    pub(crate) fn atom_is_visited(&self, atom: AtomId) -> bool {
        self.visited_atoms.contains(atom.index())
    }

    pub(crate) fn bond_is_written(&self, bond: BondId) -> bool {
        self.written_bonds.contains(bond.index())
    }

    pub(crate) fn mark_atom_visited(&mut self, atom: AtomId) -> bool {
        self.visited_atoms.insert(atom.index())
    }

    pub(crate) fn mark_bond_written(&mut self, bond: BondId) -> bool {
        self.written_bonds.insert(bond.index())
    }

    pub(crate) const fn visited_atom_count(&self) -> usize {
        self.visited_atoms.marked_count()
    }

    pub(crate) const fn written_bond_count(&self) -> usize {
        self.written_bonds.marked_count()
    }

    pub(crate) const fn all_atoms_visited(&self) -> bool {
        self.visited_atoms.is_complete()
    }

    pub(crate) const fn all_bonds_written(&self) -> bool {
        self.written_bonds.is_complete()
    }

    pub(crate) fn classify_incident(&self, incident: AdjacentBond) -> IncidentBondState {
        if self.bond_is_written(incident.bond()) {
            IncidentBondState::Written
        } else if self.atom_is_visited(incident.atom()) {
            IncidentBondState::UnwrittenToVisitedAtom
        } else {
            IncidentBondState::UnwrittenToUnvisitedAtom
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) enum IncidentBondState {
    Written,
    UnwrittenToUnvisitedAtom,
    UnwrittenToVisitedAtom,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prepared::PreparedGraphBuilder;

    fn incident(graph: &PreparedGraph, atom: AtomId, bond: BondId) -> AdjacentBond {
        graph
            .neighbors(atom)
            .expect("fixture atom must exist")
            .iter()
            .copied()
            .find(|candidate| candidate.bond() == bond)
            .expect("fixture bond must be incident to the atom")
    }

    #[test]
    fn empty_graph_progress_is_complete() {
        let graph = PreparedGraphBuilder::new().build();
        let progress = GraphProgress::new(&graph);

        assert_eq!(progress.visited_atom_count(), 0);
        assert_eq!(progress.written_bond_count(), 0);
        assert!(progress.all_atoms_visited());
        assert!(progress.all_bonds_written());
    }

    #[test]
    fn progress_classifies_tree_and_closure_incidents() {
        let mut builder = PreparedGraphBuilder::new();
        let carbon = builder.intern_token("C").unwrap();
        let atoms: [AtomId; 3] =
            std::array::from_fn(|_| builder.add_atom(carbon).unwrap());
        let bonds = [
            builder.add_bond(atoms[0], atoms[1], None).unwrap(),
            builder.add_bond(atoms[1], atoms[2], None).unwrap(),
            builder.add_bond(atoms[2], atoms[0], None).unwrap(),
        ];
        let graph = builder.build();
        let mut progress = GraphProgress::new(&graph);

        assert!(progress.mark_atom_visited(atoms[0]));
        assert_eq!(
            progress.classify_incident(incident(&graph, atoms[0], bonds[0])),
            IncidentBondState::UnwrittenToUnvisitedAtom
        );

        assert!(progress.mark_bond_written(bonds[0]));
        assert!(progress.mark_atom_visited(atoms[1]));
        assert_eq!(
            progress.classify_incident(incident(&graph, atoms[0], bonds[0])),
            IncidentBondState::Written
        );
        assert_eq!(
            progress.classify_incident(incident(&graph, atoms[0], bonds[2])),
            IncidentBondState::UnwrittenToUnvisitedAtom
        );

        assert!(progress.mark_bond_written(bonds[1]));
        assert!(progress.mark_atom_visited(atoms[2]));
        assert_eq!(
            progress.classify_incident(incident(&graph, atoms[0], bonds[2])),
            IncidentBondState::UnwrittenToVisitedAtom
        );

        assert!(progress.mark_bond_written(bonds[2]));
        assert!(progress.all_atoms_visited());
        assert!(progress.all_bonds_written());
    }

    #[test]
    fn repeated_marks_are_idempotent() {
        let mut builder = PreparedGraphBuilder::new();
        let carbon = builder.intern_token("C").unwrap();
        let atoms: [AtomId; 2] =
            std::array::from_fn(|_| builder.add_atom(carbon).unwrap());
        let bond = builder.add_bond(atoms[0], atoms[1], None).unwrap();
        let graph = builder.build();
        let mut progress = GraphProgress::new(&graph);

        assert!(progress.mark_atom_visited(atoms[0]));
        assert!(!progress.mark_atom_visited(atoms[0]));
        assert!(progress.mark_bond_written(bond));
        assert!(!progress.mark_bond_written(bond));
        assert_eq!(progress.visited_atom_count(), 1);
        assert_eq!(progress.written_bond_count(), 1);
    }

    #[test]
    fn cloned_progress_has_independent_live_bits() {
        let mut builder = PreparedGraphBuilder::new();
        let carbon = builder.intern_token("C").unwrap();
        let first = builder.add_atom(carbon).unwrap();
        builder.add_atom(carbon).unwrap();
        let graph = builder.build();
        let source = GraphProgress::new(&graph);
        let mut successor = source.clone();

        assert!(successor.mark_atom_visited(first));
        assert_eq!(source.visited_atom_count(), 0);
        assert_eq!(successor.visited_atom_count(), 1);
        assert!(!source.atom_is_visited(first));
        assert!(successor.atom_is_visited(first));
    }

    #[test]
    #[should_panic(expected = "prepared identifier must fit")]
    fn stale_prepared_ids_fail_fast() {
        let graph = PreparedGraphBuilder::new().build();
        let progress = GraphProgress::new(&graph);

        let _ = progress.atom_is_visited(AtomId::new(0));
    }
}
