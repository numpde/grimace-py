use std::collections::BTreeSet;
use std::error::Error;
use std::sync::Arc;

use super::*;
use crate::native::NativeSolverState;
use crate::native_solver::NativeSolverFailure;
use crate::prepared::PreparedGraphBuilder;

type State = NonStereoWriterState<NativeSolverState>;

#[derive(Clone, Debug, PartialEq, Eq)]
enum InjectedSolverFailure {
    Native(NativeSolverFailure),
    Restriction,
}

impl fmt::Display for InjectedSolverFailure {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Native(failure) => failure.fmt(formatter),
            Self::Restriction => formatter.write_str("injected restriction failure"),
        }
    }
}

impl Error for InjectedSolverFailure {}

#[derive(Clone, Debug)]
struct FailingRestrictionSolver(NativeSolverState);

impl ConstraintSolver for FailingRestrictionSolver {
    type Failure = InjectedSolverFailure;

    fn initial(
        model: Arc<crate::model::ConstraintModel>,
    ) -> Result<Consistency<Self>, Self::Failure> {
        Ok(<NativeSolverState as ConstraintSolver>::initial(model)
            .map_err(InjectedSolverFailure::Native)?
            .map(Self))
    }

    fn restricted(
        &self,
        _restrictions: &[(crate::ids::VariableId, crate::domain::Domain)],
    ) -> Result<Consistency<Self>, Self::Failure> {
        Err(InjectedSolverFailure::Restriction)
    }

    fn domain(&self, variable: crate::ids::VariableId) -> Option<crate::domain::Domain> {
        self.0.domain(variable)
    }
}

#[derive(Clone, Debug)]
struct FailSecondVariableSolver(NativeSolverState);

impl ConstraintSolver for FailSecondVariableSolver {
    type Failure = InjectedSolverFailure;

    fn initial(
        model: Arc<crate::model::ConstraintModel>,
    ) -> Result<Consistency<Self>, Self::Failure> {
        Ok(<NativeSolverState as ConstraintSolver>::initial(model)
            .map_err(InjectedSolverFailure::Native)?
            .map(Self))
    }

    fn restricted(
        &self,
        restrictions: &[(crate::ids::VariableId, crate::domain::Domain)],
    ) -> Result<Consistency<Self>, Self::Failure> {
        if restrictions
            .iter()
            .any(|(variable, _)| *variable == crate::ids::VariableId::new(1))
        {
            return Err(InjectedSolverFailure::Restriction);
        }
        Ok(
            <NativeSolverState as ConstraintSolver>::restricted(&self.0, restrictions)
                .map_err(InjectedSolverFailure::Native)?
                .map(Self),
        )
    }

    fn domain(&self, variable: crate::ids::VariableId) -> Option<crate::domain::Domain> {
        self.0.domain(variable)
    }
}

#[derive(Clone, Debug)]
struct RejectFirstVariableSolver(NativeSolverState);

impl ConstraintSolver for RejectFirstVariableSolver {
    type Failure = InjectedSolverFailure;

    fn initial(
        model: Arc<crate::model::ConstraintModel>,
    ) -> Result<Consistency<Self>, Self::Failure> {
        Ok(<NativeSolverState as ConstraintSolver>::initial(model)
            .map_err(InjectedSolverFailure::Native)?
            .map(Self))
    }

    fn restricted(
        &self,
        restrictions: &[(crate::ids::VariableId, crate::domain::Domain)],
    ) -> Result<Consistency<Self>, Self::Failure> {
        if restrictions
            .iter()
            .any(|(variable, _)| *variable == crate::ids::VariableId::new(0))
        {
            return Ok(Consistency::Contradiction);
        }
        Ok(
            <NativeSolverState as ConstraintSolver>::restricted(&self.0, restrictions)
                .map_err(InjectedSolverFailure::Native)?
                .map(Self),
        )
    }

    fn domain(&self, variable: crate::ids::VariableId) -> Option<crate::domain::Domain> {
        self.0.domain(variable)
    }
}

#[derive(Clone, Debug)]
struct RejectEveryRestrictionSolver(NativeSolverState);

impl ConstraintSolver for RejectEveryRestrictionSolver {
    type Failure = InjectedSolverFailure;

    fn initial(
        model: Arc<crate::model::ConstraintModel>,
    ) -> Result<Consistency<Self>, Self::Failure> {
        Ok(<NativeSolverState as ConstraintSolver>::initial(model)
            .map_err(InjectedSolverFailure::Native)?
            .map(Self))
    }

    fn restricted(
        &self,
        _restrictions: &[(crate::ids::VariableId, crate::domain::Domain)],
    ) -> Result<Consistency<Self>, Self::Failure> {
        Ok(Consistency::Contradiction)
    }

    fn domain(&self, variable: crate::ids::VariableId) -> Option<crate::domain::Domain> {
        self.0.domain(variable)
    }
}

#[derive(Clone, Debug)]
struct WriterPolicyContradictionSolver(NativeSolverState);

impl ConstraintSolver for WriterPolicyContradictionSolver {
    type Failure = InjectedSolverFailure;

    fn initial(
        model: Arc<crate::model::ConstraintModel>,
    ) -> Result<Consistency<Self>, Self::Failure> {
        Ok(<NativeSolverState as ConstraintSolver>::initial(model)
            .map_err(InjectedSolverFailure::Native)?
            .map(Self))
    }

    fn restricted(
        &self,
        restrictions: &[(crate::ids::VariableId, crate::domain::Domain)],
    ) -> Result<Consistency<Self>, Self::Failure> {
        let role_partition = BondRepresentation::role_partition();

        let first_ring = (crate::ids::VariableId::new(0), role_partition.ring_values());
        let requested_first_ring = restrictions == [first_ring];
        let effective = if requested_first_ring {
            vec![
                first_ring,
                (
                    crate::ids::VariableId::new(1),
                    role_partition.traversal_values(),
                ),
                (
                    crate::ids::VariableId::new(2),
                    role_partition.traversal_values(),
                ),
                (
                    crate::ids::VariableId::new(3),
                    role_partition.traversal_values(),
                ),
                (crate::ids::VariableId::new(4), role_partition.ring_values()),
            ]
        } else {
            restrictions.to_vec()
        };
        Ok(
            <NativeSolverState as ConstraintSolver>::restricted(&self.0, &effective)
                .map_err(InjectedSolverFailure::Native)?
                .map(Self),
        )
    }

    fn domain(&self, variable: crate::ids::VariableId) -> Option<crate::domain::Domain> {
        self.0.domain(variable)
    }
}

#[derive(Clone, Debug)]
struct PendingContradictionSolver(NativeSolverState);

impl ConstraintSolver for PendingContradictionSolver {
    type Failure = InjectedSolverFailure;

    fn initial(
        model: Arc<crate::model::ConstraintModel>,
    ) -> Result<Consistency<Self>, Self::Failure> {
        use crate::model::BondRole;

        let native = <NativeSolverState as ConstraintSolver>::initial(model)
            .map_err(InjectedSolverFailure::Native)?
            .unwrap_consistent();
        let restrictions = [
            (
                crate::ids::VariableId::new(0),
                BondRole::Traversal.singleton_domain(),
            ),
            (
                crate::ids::VariableId::new(1),
                BondRole::Traversal.singleton_domain(),
            ),
            (
                crate::ids::VariableId::new(2),
                BondRole::Traversal.singleton_domain(),
            ),
            (
                crate::ids::VariableId::new(3),
                BondRole::Ring.singleton_domain(),
            ),
        ];
        Ok(
            <NativeSolverState as ConstraintSolver>::restricted(&native, &restrictions)
                .map_err(InjectedSolverFailure::Native)?
                .map(Self),
        )
    }

    fn restricted(
        &self,
        restrictions: &[(crate::ids::VariableId, crate::domain::Domain)],
    ) -> Result<Consistency<Self>, Self::Failure> {
        Ok(
            <NativeSolverState as ConstraintSolver>::restricted(&self.0, restrictions)
                .map_err(InjectedSolverFailure::Native)?
                .map(Self),
        )
    }

    fn domain(&self, variable: crate::ids::VariableId) -> Option<crate::domain::Domain> {
        self.0.domain(variable)
    }
}

fn fixture(
    atom_text: &[&str],
    edges: &[(usize, usize, NonStereoBondToken)],
) -> (PreparedNonStereo, Vec<AtomId>, Vec<BondId>) {
    let mut graph = PreparedGraphBuilder::new();
    let atoms = atom_text
        .iter()
        .map(|_| graph.add_atom().unwrap())
        .collect::<Vec<_>>();
    let mut bonds = Vec::with_capacity(edges.len());
    let mut bond_tokens = Vec::with_capacity(edges.len());
    for &(a, b, token) in edges {
        bonds.push(graph.add_bond(atoms[a], atoms[b]).unwrap());
        bond_tokens.push(token);
    }
    let surface = PreparedNonStereo::new(
        PreparedMolecule::new(graph.build()),
        atom_text.iter().map(|text| (*text).to_owned()).collect(),
        bond_tokens,
    )
    .unwrap();
    (surface, atoms, bonds)
}

fn incident(surface: &PreparedNonStereo, atom: AtomId, bond: BondId) -> AdjacentBond {
    surface
        .molecule()
        .graph()
        .neighbors(atom)
        .expect("fixture atom must exist")
        .iter()
        .copied()
        .find(|candidate| candidate.bond() == bond)
        .expect("fixture bond must be incident to the atom")
}

fn only_choice(state: &State, text: &str) -> (String, State) {
    let choices = state.choices().unwrap();
    assert_eq!(choices.len(), 1);
    assert_eq!(choices[0].text(), text);
    let choice = choices.into_iter().next().unwrap();
    (choice.text, choice.successor)
}

fn choice_at(state: &State, index: usize) -> (String, State) {
    let choice = state.choices().unwrap().into_iter().nth(index).unwrap();
    (choice.text, choice.successor)
}

fn initial(surface: &PreparedNonStereo) -> State {
    State::initial(surface).unwrap().unwrap_consistent()
}

#[test]
fn surface_accepts_general_graphs_and_rejects_invalid_bindings() {
    let empty = PreparedMolecule::new(PreparedGraphBuilder::new().build());
    let empty = PreparedNonStereo::new(empty, Vec::new(), Vec::new()).unwrap();
    assert_eq!(empty.molecule().graph().atom_count(), 0);

    let mut graph = PreparedGraphBuilder::new();
    graph.add_atom().unwrap();
    graph.add_atom().unwrap();
    let disconnected = PreparedMolecule::new(graph.build());
    let disconnected = PreparedNonStereo::new(
        disconnected,
        vec!["C".to_owned(), "O".to_owned()],
        Vec::new(),
    )
    .unwrap();
    assert_eq!(disconnected.molecule().graph().atom_count(), 2);

    let mut graph = PreparedGraphBuilder::new();
    graph.add_atom().unwrap();
    let single = PreparedMolecule::new(graph.build());
    assert!(matches!(
        PreparedNonStereo::new(single.clone(), Vec::new(), Vec::new()),
        Err(PreparedNonStereoError::AtomTextCountMismatch { .. })
    ));
    assert!(matches!(
        PreparedNonStereo::new(single, vec![String::new()], Vec::new()),
        Err(PreparedNonStereoError::EmptyAtomText(atom))
            if atom == AtomId::new(0)
    ));

    let mut graph = PreparedGraphBuilder::new();
    let atoms: [AtomId; 2] = std::array::from_fn(|_| graph.add_atom().unwrap());
    graph.add_bond(atoms[0], atoms[1]).unwrap();
    let bonded = PreparedMolecule::new(graph.build());
    assert!(matches!(
        PreparedNonStereo::new(bonded, vec!["C".to_owned(), "O".to_owned()], Vec::new(),),
        Err(PreparedNonStereoError::BondTokenCountMismatch { .. })
    ));
}

#[test]
fn surface_prepares_local_bond_representation_domains_without_binary_factors() {
    let (surface, _, bonds) = fixture(
        &["A", "B", "C"],
        &[
            (0, 1, NonStereoBondToken::Elided),
            (1, 2, NonStereoBondToken::Double),
        ],
    );
    let molecule = surface.molecule();
    let model = molecule.constraint_model();
    let initial_domain = |bond| {
        let variable = molecule.bond_decision_variable(bond).unwrap();
        model.variable(variable).unwrap().initial_domain()
    };

    assert_eq!(
        initial_domain(bonds[0]),
        BondRepresentation::elided_domain()
    );
    assert_eq!(
        initial_domain(bonds[1]),
        BondRepresentation::explicit_domain()
    );
    assert_eq!(
        molecule.bond_role_partition(bonds[0]),
        Some(BondRepresentation::role_partition())
    );
    assert_eq!(model.factor_count(), 1);
    assert!(matches!(
        model.factor(crate::ids::FactorId::new(0)),
        Some(crate::model::FactorDefinition::SpanningTree(_))
    ));
}

#[test]
fn top_level_component_completion_is_silent_normalization() {
    let (surface, atoms, _) = fixture(&["A", "B"], &[]);
    let initial = initial(&surface);
    let first = initial
        .choices()
        .unwrap()
        .into_iter()
        .find(|choice| choice.text() == "A")
        .unwrap()
        .into_successor();

    assert_eq!(initial.active_atom(), None, "the source remains unchanged");
    assert_eq!(first.active_atom(), None);
    assert!(!first.graph_is_complete());
    assert_eq!(first.pending, None);
    assert!(first.labels.is_clean());
    assert_eq!(
        first.structural.derive_candidates().candidates(),
        &[StructuralCandidate::Root { atom: atoms[1] }]
    );
}

#[test]
fn equal_dot_choices_commit_distinct_pending_component_roots() {
    let (surface, atoms, _) = fixture(&["A", "B", "C"], &[]);
    let initial = initial(&surface);
    let after_first = initial
        .choices()
        .unwrap()
        .into_iter()
        .find(|choice| choice.text() == "A")
        .unwrap()
        .into_successor();

    let choices = after_first.choices().unwrap();
    assert_eq!(choices.len(), 2);
    assert!(choices.iter().all(|choice| choice.text() == "."));
    assert_eq!(
        after_first.structural.derive_candidates().candidates(),
        &[
            StructuralCandidate::Root { atom: atoms[1] },
            StructuralCandidate::Root { atom: atoms[2] },
        ]
    );
    assert_eq!(
        after_first.active_atom(),
        None,
        "the source remains unchanged"
    );
    for (choice, root) in choices.iter().zip(&atoms[1..]) {
        assert_eq!(choice.successor().active_atom(), Some(*root));
        assert!(choice.successor().structural.atom_is_visited(atoms[0]));
        assert!(choice.successor().structural.atom_is_visited(*root));
        assert_eq!(
            choice.successor().pending,
            Some(PendingEmission::ComponentRootAtom(*root))
        );
        let atom_choice = choice.successor().choices().unwrap();
        assert_eq!(atom_choice.len(), 1);
        assert_eq!(atom_choice[0].text(), surface.atom_text(*root));
    }
}

#[test]
fn equal_text_choices_retain_distinct_successors() {
    let (surface, atoms, _) = fixture(&["C", "C"], &[(0, 1, NonStereoBondToken::Elided)]);
    let initial = initial(&surface);

    let choices = initial.choices().unwrap();
    assert_eq!(choices.len(), 2);
    assert_eq!(choices[0].text(), choices[1].text());
    assert_eq!(choices[0].text(), "C");
    assert_eq!(choices[0].successor().active_atom(), Some(atoms[0]));
    assert_eq!(choices[1].successor().active_atom(), Some(atoms[1]));
    assert_eq!(initial.active_atom(), None);
}

#[test]
fn accepted_state_returns_an_ordinary_empty_choice_result() {
    let surface = fixture(&["C"], &[]).0;
    let initial = initial(&surface);
    let (_, accepted) = only_choice(&initial, "C");

    assert!(accepted.is_accepted());
    assert!(accepted.choices().unwrap().is_empty());
}

#[test]
fn choices_derives_the_source_frontier_once() {
    let surface = fixture(
        &["C", "C", "C"],
        &[
            (0, 1, NonStereoBondToken::Elided),
            (0, 2, NonStereoBondToken::Elided),
            (1, 2, NonStereoBondToken::Elided),
        ],
    )
    .0;
    let initial = initial(&surface);
    let rooted = initial
        .choices()
        .unwrap()
        .into_iter()
        .next()
        .unwrap()
        .into_successor();
    let before = rooted.structural.candidate_batch_derivation_count();

    let choices = rooted.choices().unwrap();

    assert_eq!(choices.len(), 2);
    assert_eq!(
        rooted.structural.candidate_batch_derivation_count(),
        before + 1
    );
}

#[test]
fn backend_failure_aborts_the_candidate_batch() {
    let surface = fixture(
        &["C", "C", "C"],
        &[
            (0, 1, NonStereoBondToken::Elided),
            (0, 2, NonStereoBondToken::Elided),
            (1, 2, NonStereoBondToken::Elided),
        ],
    )
    .0;
    let initial = NonStereoWriterState::<FailingRestrictionSolver>::initial(&surface)
        .unwrap()
        .unwrap_consistent();
    let mut rooted = initial
        .choices()
        .unwrap()
        .into_iter()
        .next()
        .unwrap()
        .into_successor();
    rooted.labels.maximum_spelling_label = Some(0);

    assert!(matches!(
        rooted.choices(),
        Err(ChoiceFailure::Backend(InjectedSolverFailure::Restriction))
    ));
}

#[test]
fn late_backend_failure_discards_an_earlier_accepted_choice() {
    let surface = fixture(
        &["C", "C", "C"],
        &[
            (0, 1, NonStereoBondToken::Elided),
            (0, 2, NonStereoBondToken::Elided),
            (1, 2, NonStereoBondToken::Elided),
        ],
    )
    .0;
    let initial = NonStereoWriterState::<FailSecondVariableSolver>::initial(&surface)
        .unwrap()
        .unwrap_consistent();
    let rooted = initial
        .choices()
        .unwrap()
        .into_iter()
        .next()
        .unwrap()
        .into_successor();

    assert!(matches!(
        rooted.choices(),
        Err(ChoiceFailure::Backend(InjectedSolverFailure::Restriction))
    ));
}

#[test]
fn contradictory_candidate_is_filtered_without_suppressing_its_sibling() {
    let (surface, atoms, bonds) = fixture(
        &["C", "C", "C"],
        &[
            (0, 1, NonStereoBondToken::Elided),
            (0, 2, NonStereoBondToken::Elided),
            (1, 2, NonStereoBondToken::Elided),
        ],
    );
    assert_eq!(
        surface.molecule().bond_decision_variable(bonds[0]),
        Some(crate::ids::VariableId::new(0))
    );
    let initial = NonStereoWriterState::<RejectFirstVariableSolver>::initial(&surface)
        .unwrap()
        .unwrap_consistent();
    let rooted = initial
        .choices()
        .unwrap()
        .into_iter()
        .find(|choice| choice.successor().active_atom() == Some(atoms[0]))
        .unwrap()
        .into_successor();

    let choices = rooted.choices().unwrap();

    assert_eq!(choices.len(), 1);
    assert_eq!(choices[0].text(), "1");
    assert_eq!(
        choices[0]
            .successor()
            .labels
            .bonds_by_slot
            .values()
            .copied()
            .collect::<Vec<_>>(),
        vec![bonds[1]]
    );
    assert!(rooted.labels.is_clean());
}

#[test]
fn writer_policy_contradiction_is_candidate_local() {
    let (surface, atoms, bonds) = fixture(
        &["R", "A", "B", "C"],
        &[
            (0, 1, NonStereoBondToken::Elided),
            (0, 2, NonStereoBondToken::Elided),
            (0, 3, NonStereoBondToken::Elided),
            (1, 2, NonStereoBondToken::Elided),
            (2, 3, NonStereoBondToken::Elided),
        ],
    );
    let initial = NonStereoWriterState::<WriterPolicyContradictionSolver>::initial(&surface)
        .unwrap()
        .unwrap_consistent();
    let rooted = initial
        .choices()
        .unwrap()
        .into_iter()
        .find(|choice| choice.successor().active_atom() == Some(atoms[0]))
        .unwrap()
        .into_successor();

    let choices = rooted.choices().unwrap();

    assert!(!choices.is_empty());
    assert!(choices.iter().all(|choice| {
        !choice
            .successor()
            .labels
            .bonds_by_slot
            .values()
            .any(|bond| *bond == bonds[0])
    }));
    assert!(choices.iter().any(|choice| {
        choice
            .successor()
            .labels
            .bonds_by_slot
            .values()
            .any(|bond| *bond == bonds[1])
    }));
}

#[test]
fn all_candidate_contradiction_is_an_explicit_live_state_failure() {
    let (surface, atoms, _) = fixture(
        &["R", "A", "B", "C"],
        &[
            (0, 1, NonStereoBondToken::Double),
            (1, 2, NonStereoBondToken::Elided),
            (1, 3, NonStereoBondToken::Elided),
            (2, 3, NonStereoBondToken::Elided),
        ],
    );
    let initial = NonStereoWriterState::<PendingContradictionSolver>::initial(&surface)
        .unwrap()
        .unwrap_consistent();
    let rooted = initial
        .choices()
        .unwrap()
        .into_iter()
        .find(|choice| choice.successor().active_atom() == Some(atoms[0]))
        .unwrap()
        .into_successor();

    assert!(matches!(
        rooted.choices(),
        Err(ChoiceFailure::Invariant(
            WriterInvariantFailure::AllCandidatesSemanticallyRejected { candidate_count: 1 }
        ))
    ));
}

#[test]
fn all_candidate_spelling_exhaustion_is_typed() {
    let (surface, atoms, _) = fixture(
        &["C", "C", "C"],
        &[
            (0, 1, NonStereoBondToken::Elided),
            (0, 2, NonStereoBondToken::Elided),
            (1, 2, NonStereoBondToken::Elided),
        ],
    );
    let initial = initial(&surface);
    let mut rooted = initial
        .choices()
        .unwrap()
        .into_iter()
        .find(|choice| choice.successor().active_atom() == Some(atoms[0]))
        .unwrap()
        .into_successor();
    rooted.labels.maximum_spelling_label = Some(0);

    assert!(matches!(
        rooted.choices(),
        Err(ChoiceFailure::Spelling(
            SpellingFailure::RingLabelExhausted {
                next_label: 1,
                maximum_label: 0,
                blocked_candidate_count: 2
            }
        ))
    ));
}

#[test]
fn viable_unspellable_candidate_outweighs_a_contradictory_sibling() {
    let (surface, atoms, _) = fixture(
        &["C", "C", "C"],
        &[
            (0, 1, NonStereoBondToken::Elided),
            (0, 2, NonStereoBondToken::Elided),
            (1, 2, NonStereoBondToken::Elided),
        ],
    );
    let initial = NonStereoWriterState::<RejectFirstVariableSolver>::initial(&surface)
        .unwrap()
        .unwrap_consistent();
    let mut rooted = initial
        .choices()
        .unwrap()
        .into_iter()
        .find(|choice| choice.successor().active_atom() == Some(atoms[0]))
        .unwrap()
        .into_successor();
    rooted.labels.maximum_spelling_label = Some(0);

    assert!(matches!(
        rooted.choices(),
        Err(ChoiceFailure::Spelling(
            SpellingFailure::RingLabelExhausted {
                next_label: 1,
                maximum_label: 0,
                blocked_candidate_count: 1
            }
        ))
    ));
}

#[test]
fn contradiction_precedes_spelling_for_each_candidate() {
    let (surface, atoms, _) = fixture(
        &["C", "C", "C"],
        &[
            (0, 1, NonStereoBondToken::Elided),
            (0, 2, NonStereoBondToken::Elided),
            (1, 2, NonStereoBondToken::Elided),
        ],
    );
    let initial = NonStereoWriterState::<RejectEveryRestrictionSolver>::initial(&surface)
        .unwrap()
        .unwrap_consistent();
    let mut rooted = initial
        .choices()
        .unwrap()
        .into_iter()
        .find(|choice| choice.successor().active_atom() == Some(atoms[0]))
        .unwrap()
        .into_successor();
    rooted.labels.maximum_spelling_label = Some(0);

    assert!(matches!(
        rooted.choices(),
        Err(ChoiceFailure::Invariant(
            WriterInvariantFailure::AllCandidatesSemanticallyRejected { candidate_count: 2 }
        ))
    ));
}

#[test]
fn unspellable_openings_do_not_suppress_a_valid_closure() {
    let (surface, atoms, bonds) = fixture(
        &["A", "B", "C", "D", "E"],
        &[
            (0, 1, NonStereoBondToken::Elided),
            (0, 2, NonStereoBondToken::Elided),
            (1, 2, NonStereoBondToken::Elided),
            (1, 3, NonStereoBondToken::Elided),
            (1, 4, NonStereoBondToken::Elided),
            (3, 4, NonStereoBondToken::Elided),
        ],
    );
    let initial = initial(&surface);
    let rooted = initial
        .choices()
        .unwrap()
        .into_iter()
        .find(|choice| choice.successor().active_atom() == Some(atoms[0]))
        .unwrap()
        .into_successor();
    let opened = rooted
        .choices()
        .unwrap()
        .into_iter()
        .find(|choice| {
            choice
                .successor()
                .labels
                .bonds_by_slot
                .values()
                .any(|bond| *bond == bonds[0])
        })
        .unwrap()
        .into_successor();
    let (_, walked) = only_choice(&opened, "C");
    let (_, mut walked) = only_choice(&walked, "B");
    assert_eq!(walked.active_atom(), Some(atoms[1]));
    walked.labels.maximum_spelling_label = Some(1);
    assert_eq!(walked.labels.next_available(), RingLabelSlot(1));

    let choices = walked.choices().unwrap();

    assert_eq!(choices.len(), 1);
    assert_eq!(choices[0].text(), "1");
    assert!(!choices[0]
        .successor()
        .labels
        .bonds_by_slot
        .values()
        .any(|bond| *bond == bonds[0]));
}

#[test]
fn closed_labels_are_immediately_reusable() {
    let first = BondId::new(0);
    let second = BondId::new(1);
    let third = BondId::new(2);
    let mut labels = RingLabels::default();

    let zero = labels.allocate(first);
    let one = labels.allocate(second);
    assert_eq!(zero, RingLabelSlot(0));
    assert_eq!(one, RingLabelSlot(1));
    labels.release(zero, first);
    assert_eq!(labels.allocate(third), RingLabelSlot(0));
}

#[test]
fn ring_label_spelling_matches_the_selected_smiles_dialect() {
    assert_eq!(ring_label_number_text(1), "1");
    assert_eq!(ring_label_number_text(9), "9");
    assert_eq!(ring_label_number_text(10), "%10");
    assert_eq!(ring_label_number_text(99), "%99");
}

#[test]
#[should_panic(expected = "above 99 require an explicit dialect policy")]
fn unselected_large_ring_label_dialect_fails_at_rendering() {
    let _ = ring_label_number_text(100);
}

#[test]
fn elided_triangle_emits_a_complete_ring() {
    let (surface, atoms, bonds) = fixture(
        &["C", "C", "C"],
        &[
            (0, 1, NonStereoBondToken::Elided),
            (0, 2, NonStereoBondToken::Elided),
            (1, 2, NonStereoBondToken::Elided),
        ],
    );
    let initial = initial(&surface);
    let left = incident(&surface, atoms[0], bonds[0]);
    let right = incident(&surface, atoms[0], bonds[1]);
    let between = incident(&surface, atoms[2], bonds[2]);
    let closing = incident(&surface, atoms[1], bonds[0]);

    let (root, rooted) = choice_at(&initial, atoms[0].index());
    let rooted_choices = rooted.choices().unwrap();
    assert_eq!(
        rooted_choices.iter().map(Choice::text).collect::<Vec<_>>(),
        vec!["1", "1"]
    );
    assert!(rooted_choices[0]
        .successor()
        .labels
        .bonds_by_slot
        .values()
        .any(|bond| *bond == left.bond()));
    assert!(rooted_choices[1]
        .successor()
        .labels
        .bonds_by_slot
        .values()
        .any(|bond| *bond == right.bond()));
    let first_opening = rooted_choices.into_iter().next().unwrap();
    let open = first_opening.text;
    let opened = first_opening.successor;
    let (first_child, walked) = only_choice(&opened, "C");
    let (second_child, walked) = only_choice(&walked, "C");
    assert_eq!(walked.structural.active_atom(), Some(atoms[1]));
    assert_eq!(between.bond(), bonds[2]);
    assert_eq!(closing.bond(), bonds[0]);
    let (close, accepted) = only_choice(&walked, "1");

    assert_eq!(
        [root, open, first_child, second_child, close].concat(),
        "C1CC1"
    );
    assert!(accepted.is_accepted());
}

#[test]
fn explicit_ring_bond_is_emitted_at_closure_before_its_label() {
    let (surface, atoms, bonds) = fixture(
        &["C", "C", "C"],
        &[
            (0, 1, NonStereoBondToken::Double),
            (0, 2, NonStereoBondToken::Elided),
            (1, 2, NonStereoBondToken::Elided),
        ],
    );
    let initial = initial(&surface);
    let ring = incident(&surface, atoms[0], bonds[0]);
    let entry = incident(&surface, atoms[0], bonds[1]);
    let between = incident(&surface, atoms[2], bonds[2]);
    let closing = incident(&surface, atoms[1], bonds[0]);

    let (root, rooted) = choice_at(&initial, atoms[0].index());
    let (open, opened) = choice_at(&rooted, 0);
    assert!(opened
        .labels
        .bonds_by_slot
        .values()
        .any(|bond| *bond == ring.bond()));
    let (first_child, walked) = only_choice(&opened, "C");
    assert_eq!(entry.bond(), bonds[1]);
    let (second_child, walked) = only_choice(&walked, "C");
    assert_eq!(between.bond(), bonds[2]);
    assert_eq!(closing.bond(), bonds[0]);

    let (bond, pending_label) = only_choice(&walked, "=");
    assert_eq!(bond, "=");
    assert_eq!(pending_label.active_atom(), Some(atoms[1]));
    assert!(!pending_label.graph_is_complete());

    let (label, accepted) = only_choice(&pending_label, "1");
    assert_eq!(
        [root, open, first_child, second_child, bond, label].concat(),
        "C1CC=1"
    );
    assert!(accepted.is_accepted());
}

#[test]
fn directed_ring_closure_uses_the_first_endpoint_orientation() {
    let (surface, atoms, bonds) = fixture(
        &["N", "B", "C"],
        &[
            (0, 1, NonStereoBondToken::DativeAToB),
            (0, 2, NonStereoBondToken::Elided),
            (1, 2, NonStereoBondToken::Elided),
        ],
    );
    let initial = initial(&surface);
    let ring = incident(&surface, atoms[0], bonds[0]);
    let entry = incident(&surface, atoms[0], bonds[1]);
    let between = incident(&surface, atoms[2], bonds[2]);
    let closing = incident(&surface, atoms[1], bonds[0]);

    let (root, rooted) = choice_at(&initial, atoms[0].index());
    let opening = rooted
        .choices()
        .unwrap()
        .into_iter()
        .find(|choice| {
            choice
                .successor()
                .labels
                .bonds_by_slot
                .values()
                .any(|bond| *bond == ring.bond())
        })
        .unwrap();
    let open = opening.text;
    let opened = opening.successor;
    let (first_child, walked) = only_choice(&opened, "C");
    assert_eq!(entry.bond(), bonds[1]);
    let (second_child, walked) = only_choice(&walked, "B");
    assert_eq!(between.bond(), bonds[2]);
    assert_eq!(closing.bond(), bonds[0]);

    let (bond, pending_label) = only_choice(&walked, "->");
    let (label, accepted) = only_choice(&pending_label, "1");

    assert_eq!(
        [root, open, first_child, second_child, bond, label].concat(),
        "N1CB->1"
    );
    assert!(accepted.is_accepted());
}

#[test]
fn explicit_inline_bond_commits_before_child_entry() {
    let (surface, atoms, bonds) = fixture(&["C", "O"], &[(0, 1, NonStereoBondToken::Double)]);
    let initial = initial(&surface);
    let edge = incident(&surface, atoms[0], bonds[0]);
    let (_, rooted) = choice_at(&initial, atoms[0].index());
    let (bond, pending) = only_choice(&rooted, "=");

    assert_eq!(bond, "=");
    assert_eq!(rooted.active_atom(), Some(atoms[0]));
    assert_eq!(pending.active_atom(), Some(atoms[0]));
    assert_eq!(pending.pending, Some(PendingEmission::InlineAtom(edge)));
    let (atom, accepted) = only_choice(&pending, "O");
    assert_eq!(atom, "O");
    assert!(accepted.is_accepted());
}

#[test]
fn dative_bond_text_follows_prepared_orientation() {
    let (surface, atoms, bonds) = fixture(&["N", "B"], &[(0, 1, NonStereoBondToken::DativeAToB)]);
    let initial = initial(&surface);
    let edge_from_n = incident(&surface, atoms[0], bonds[0]);
    let edge_from_b = incident(&surface, atoms[1], bonds[0]);

    let (_, rooted_at_n) = choice_at(&initial, atoms[0].index());
    assert_eq!(rooted_at_n.pending, None);
    assert_eq!(edge_from_n.bond(), bonds[0]);
    assert_eq!(rooted_at_n.choices().unwrap()[0].text(), "->");

    let (_, rooted_at_b) = choice_at(&initial, atoms[1].index());
    assert_eq!(edge_from_b.bond(), bonds[0]);
    assert_eq!(rooted_at_b.choices().unwrap()[0].text(), "<-");
}

#[test]
fn explicit_branch_commits_at_open_parenthesis() {
    let (surface, atoms, bonds) = fixture(
        &["C", "O", "N"],
        &[
            (0, 1, NonStereoBondToken::Double),
            (0, 2, NonStereoBondToken::Elided),
        ],
    );
    let initial = initial(&surface);
    let oxygen = incident(&surface, atoms[0], bonds[0]);
    let nitrogen = incident(&surface, atoms[0], bonds[1]);
    let (root, rooted) = choice_at(&initial, atoms[0].index());
    let branch_choice = rooted
        .choices()
        .unwrap()
        .into_iter()
        .find(|choice| choice.successor.pending == Some(PendingEmission::BranchBondOrAtom(oxygen)))
        .unwrap();
    let open = branch_choice.text;
    let pending_branch = branch_choice.successor;
    let (bond, pending_atom) = only_choice(&pending_branch, "=");
    let (atom, branch) = only_choice(&pending_atom, "O");
    let (close, restored) = only_choice(&branch, ")");
    assert_eq!(nitrogen.bond(), bonds[1]);
    let (inline, accepted) = only_choice(&restored, "N");

    assert_eq!([root, open, bond, atom, close, inline].concat(), "C(=O)N");
    assert!(accepted.is_accepted());
}

fn reachable_terminal_paths(surface: &PreparedNonStereo) -> Vec<String> {
    let initial = initial(surface);
    let mut pending = vec![(initial, String::new())];
    let mut complete = Vec::new();
    let mut explored = 0_usize;

    while let Some((state, prefix)) = pending.pop() {
        explored += 1;
        assert!(
            explored <= 100_000,
            "writer test exceeded its exploration bound"
        );
        if state.is_accepted() {
            complete.push(prefix);
            continue;
        }

        let choices = state.choices().unwrap();
        assert!(
            !choices.is_empty(),
            "writer must not dead-end before acceptance"
        );
        for choice in choices {
            let token = choice.text().to_owned();
            pending.push((choice.into_successor(), format!("{prefix}{token}")));
        }
    }
    complete
}

fn reachable_strings(surface: &PreparedNonStereo) -> BTreeSet<String> {
    reachable_terminal_paths(surface).into_iter().collect()
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

    if items.is_empty() {
        return vec![Vec::new()];
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

fn reference_tree_subtrees(
    surface: &PreparedNonStereo,
    atom: AtomId,
    parent: Option<AtomId>,
) -> BTreeSet<String> {
    let children = surface
        .molecule()
        .graph()
        .neighbors(atom)
        .expect("reference atom must exist")
        .iter()
        .copied()
        .filter(|incident| Some(incident.atom()) != parent)
        .collect::<Vec<_>>();
    let mut support = BTreeSet::new();

    for order in permutations(&children) {
        let mut partial = vec![surface.atom_text(atom).to_owned()];
        for (index, incident) in order.iter().copied().enumerate() {
            let child_support = reference_tree_subtrees(surface, incident.atom(), Some(atom));
            let bond = surface.bond_text(incident.bond(), atom);
            let inline = index + 1 == order.len();
            let mut next = Vec::new();
            for prefix in &partial {
                for child in &child_support {
                    if inline {
                        next.push(format!("{prefix}{bond}{child}"));
                    } else {
                        next.push(format!("{prefix}({bond}{child})"));
                    }
                }
            }
            partial = next;
        }
        support.extend(partial);
    }
    support
}

fn reference_tree_strings(surface: &PreparedNonStereo) -> BTreeSet<String> {
    surface
        .molecule()
        .graph()
        .atom_ids()
        .flat_map(|root| reference_tree_subtrees(surface, root, None))
        .collect()
}

fn reference_component_strings(
    surface: &PreparedNonStereo,
    component_atoms: &[AtomId],
) -> BTreeSet<String> {
    component_atoms
        .iter()
        .copied()
        .flat_map(|root| reference_tree_subtrees(surface, root, None))
        .collect()
}

fn reference_component_composition(
    surface: &PreparedNonStereo,
    components: &[Vec<AtomId>],
) -> BTreeSet<String> {
    let component_support = components
        .iter()
        .map(|atoms| reference_component_strings(surface, atoms))
        .collect::<Vec<_>>();
    let component_ids = (0..components.len()).collect::<Vec<_>>();
    let mut support = BTreeSet::new();

    for order in permutations(&component_ids) {
        let mut partial = BTreeSet::from([String::new()]);
        for component in order {
            let mut next = BTreeSet::new();
            for prefix in &partial {
                for component_text in &component_support[component] {
                    let separator = if prefix.is_empty() { "" } else { "." };
                    next.insert(format!("{prefix}{separator}{component_text}"));
                }
            }
            partial = next;
        }
        support.extend(partial);
    }
    support
}

#[test]
fn empty_and_isolated_component_semantics_are_exact() {
    let empty = fixture(&[], &[]).0;
    let empty_initial = initial(&empty);
    assert!(empty_initial.is_accepted());
    assert!(empty_initial.choices().unwrap().is_empty());
    assert_eq!(reachable_strings(&empty), BTreeSet::from([String::new()]));

    let two = fixture(&["A", "B"], &[]).0;
    assert_eq!(
        reachable_strings(&two),
        BTreeSet::from(["A.B".to_owned(), "B.A".to_owned()])
    );

    let distinct = fixture(&["A", "B", "C"], &[]).0;
    let distinct_paths = reachable_terminal_paths(&distinct);
    assert_eq!(distinct_paths.len(), 6);
    assert_eq!(distinct_paths.iter().collect::<BTreeSet<_>>().len(), 6);

    let identical = fixture(&["C", "C", "C"], &[]).0;
    let identical_paths = reachable_terminal_paths(&identical);
    assert_eq!(identical_paths.len(), 6);
    assert_eq!(
        identical_paths.into_iter().collect::<BTreeSet<_>>(),
        BTreeSet::from(["C.C.C".to_owned()])
    );
}

#[test]
fn branch_closure_and_component_separator_have_distinct_tokens() {
    let (surface, atoms, bonds) = fixture(
        &["A", "B", "C", "D"],
        &[
            (0, 1, NonStereoBondToken::Elided),
            (0, 2, NonStereoBondToken::Elided),
        ],
    );
    let initial = initial(&surface);
    let branch_incident = incident(&surface, atoms[0], bonds[0]);
    let (root, rooted) = choice_at(&initial, atoms[0].index());
    let branch_choice = rooted
        .choices()
        .unwrap()
        .into_iter()
        .find(|choice| {
            choice.successor().pending == Some(PendingEmission::BranchBondOrAtom(branch_incident))
        })
        .unwrap();
    let open = branch_choice.text;
    let (branch_atom, branch) = only_choice(&branch_choice.successor, "B");
    let (close, restored) = only_choice(&branch, ")");
    let (inline_atom, boundary) = only_choice(&restored, "C");

    assert_eq!(boundary.active_atom(), None);
    assert!(boundary.labels.is_clean());
    let (separator, pending_root) = only_choice(&boundary, ".");
    assert_eq!(
        pending_root.pending,
        Some(PendingEmission::ComponentRootAtom(atoms[3]))
    );
    let (last_atom, accepted) = only_choice(&pending_root, "D");

    assert_eq!(
        [
            root,
            open,
            branch_atom,
            close,
            inline_atom,
            separator,
            last_atom,
        ]
        .concat(),
        "A(B)C.D"
    );
    assert!(accepted.is_accepted());
}

#[test]
fn disconnected_tree_support_matches_component_product() {
    let (surface, atoms, _) = fixture(
        &["A", "B", "C", "D", "E", "F"],
        &[
            (0, 1, NonStereoBondToken::Elided),
            (2, 3, NonStereoBondToken::Elided),
            (2, 4, NonStereoBondToken::Double),
        ],
    );
    let components = vec![
        vec![atoms[0], atoms[1]],
        vec![atoms[2], atoms[3], atoms[4]],
        vec![atoms[5]],
    ];

    assert_eq!(
        reachable_strings(&surface),
        reference_component_composition(&surface, &components)
    );
}

#[test]
fn cyclic_components_close_and_reuse_ring_label_one() {
    let (surface, atoms, bonds) = fixture(
        &["C", "C", "C", "C", "C", "C"],
        &[
            (0, 1, NonStereoBondToken::Elided),
            (0, 2, NonStereoBondToken::Elided),
            (1, 2, NonStereoBondToken::Elided),
            (3, 4, NonStereoBondToken::Elided),
            (3, 5, NonStereoBondToken::Elided),
            (4, 5, NonStereoBondToken::Elided),
        ],
    );
    let initial = initial(&surface);
    let (first_root, rooted) = choice_at(&initial, atoms[0].index());
    let first_opening = rooted.choices().unwrap().into_iter().next().unwrap();
    assert_eq!(first_opening.text(), "1");
    let first_open = first_opening.text;
    let (_, walked) = only_choice(&first_opening.successor, "C");
    let (_, walked) = only_choice(&walked, "C");
    let (first_close, boundary) = only_choice(&walked, "1");

    assert_eq!(boundary.active_atom(), None);
    assert!(boundary.labels.is_clean());
    assert!(!boundary.graph_is_complete());
    let dot_choice = boundary
        .choices()
        .unwrap()
        .into_iter()
        .find(|choice| choice.successor().active_atom() == Some(atoms[3]))
        .unwrap();
    assert_eq!(dot_choice.text(), ".");
    assert_eq!(
        dot_choice.successor().pending,
        Some(PendingEmission::ComponentRootAtom(atoms[3]))
    );
    let separator = dot_choice.text;
    let (second_root, rooted) = only_choice(&dot_choice.successor, "C");
    assert!(rooted.labels.is_clean());
    let second_opening = rooted.choices().unwrap().into_iter().next().unwrap();
    assert_eq!(second_opening.text(), "1");
    let second_open = second_opening.text;
    let (_, walked) = only_choice(&second_opening.successor, "C");
    let (_, walked) = only_choice(&walked, "C");
    let (second_close, accepted) = only_choice(&walked, "1");

    assert_eq!(bonds.len(), 6);
    assert_eq!(
        [
            first_root,
            first_open,
            "CC".to_owned(),
            first_close,
            separator,
            second_root,
            second_open,
            "CC".to_owned(),
            second_close,
        ]
        .concat(),
        "C1CC1.C1CC1"
    );
    assert!(accepted.labels.is_clean());
    assert!(accepted.is_accepted());

    let terminal_paths = reachable_terminal_paths(&surface);
    assert_eq!(terminal_paths.len(), 72);
    assert_eq!(
        terminal_paths.into_iter().collect::<BTreeSet<_>>(),
        BTreeSet::from(["C1CC1.C1CC1".to_owned()])
    );
}

#[test]
fn connected_tree_support_remains_exact() {
    let fixtures = [
        fixture(&["C"], &[]).0,
        fixture(
            &["C", "N", "O"],
            &[
                (0, 1, NonStereoBondToken::Elided),
                (1, 2, NonStereoBondToken::Elided),
            ],
        )
        .0,
        fixture(
            &["C", "N", "O", "F"],
            &[
                (0, 1, NonStereoBondToken::Elided),
                (0, 2, NonStereoBondToken::Elided),
                (0, 3, NonStereoBondToken::Elided),
            ],
        )
        .0,
        fixture(
            &["C", "N", "O", "F", "S"],
            &[
                (0, 1, NonStereoBondToken::Elided),
                (0, 2, NonStereoBondToken::Elided),
                (1, 3, NonStereoBondToken::Elided),
                (1, 4, NonStereoBondToken::Double),
            ],
        )
        .0,
    ];
    for surface in fixtures {
        assert_eq!(
            reachable_strings(&surface),
            reference_tree_strings(&surface)
        );
    }
}

#[test]
fn connected_triangle_support_is_writer_shaped() {
    let surface = fixture(
        &["C", "C", "C"],
        &[
            (0, 1, NonStereoBondToken::Elided),
            (0, 2, NonStereoBondToken::Elided),
            (1, 2, NonStereoBondToken::Elided),
        ],
    )
    .0;

    assert_eq!(
        reachable_strings(&surface),
        BTreeSet::from(["C1CC1".to_owned()])
    );
}

#[test]
fn fused_and_bridged_cycles_have_complete_online_walks() {
    let fixtures = [
        fixture(
            &["A", "B", "C", "D"],
            &[
                (0, 1, NonStereoBondToken::Elided),
                (1, 2, NonStereoBondToken::Elided),
                (2, 0, NonStereoBondToken::Elided),
                (1, 3, NonStereoBondToken::Elided),
                (2, 3, NonStereoBondToken::Elided),
            ],
        )
        .0,
        fixture(
            &["A", "B", "C", "D", "E"],
            &[
                (0, 1, NonStereoBondToken::Elided),
                (1, 3, NonStereoBondToken::Elided),
                (0, 2, NonStereoBondToken::Elided),
                (2, 3, NonStereoBondToken::Elided),
                (0, 4, NonStereoBondToken::Elided),
                (4, 3, NonStereoBondToken::Elided),
            ],
        )
        .0,
    ];

    for surface in fixtures {
        assert!(!reachable_strings(&surface).is_empty());
    }
}
