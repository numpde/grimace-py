//! Internal cross-crate interface for native successor-bearing writer choices.
//!
//! Preparation supplies chemical spelling and stereo facts. This interface
//! preserves kernel outcomes while keeping traversal and solver stores private.

use std::error::Error;
use std::fmt;

use crate::native::NativeSolverState;
use crate::native_solver::NativeSolverFailure;
use crate::nonstereo_writer::{
    ChoiceFailure, NonStereoWriterState, PreparedNonStereo, PreparedNonStereoError,
};
use crate::solver::Consistency;
use crate::{AtomToken, BondToken, DirectionalRelation, PreparedGraph, PreparedMolecule};

/// Controls whether disconnected components may begin in any order or in
/// their prepared order.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ComponentOrder {
    AnyOrder,
    InputOrder,
}

#[derive(Clone, Debug)]
pub struct PreparedWriter {
    surface: PreparedNonStereo,
    component_order: ComponentOrder,
}

impl PreparedWriter {
    pub fn new(
        graph: PreparedGraph,
        atoms: Vec<AtomToken>,
        bonds: Vec<BondToken>,
        directional: Vec<DirectionalRelation>,
        component_order: ComponentOrder,
    ) -> Result<Self, WalkerFailure> {
        PreparedNonStereo::with_atom_tokens_and_directional(
            PreparedMolecule::new(graph),
            atoms,
            bonds,
            directional,
        )
        .map(|surface| Self {
            surface,
            component_order,
        })
        .map_err(|error| WalkerFailure(FailureCause::Preparation(error)))
    }

    pub fn initial(&self) -> Result<Walker, WalkerFailure> {
        match NonStereoWriterState::initial_with_component_order(
            &self.surface,
            self.component_order,
        ) {
            Ok(Consistency::Consistent(state)) => Ok(Walker(state)),
            Ok(Consistency::Contradiction) => Err(WalkerFailure(FailureCause::ContradictoryInput)),
            Err(error) => Err(WalkerFailure(FailureCause::InitialBackend(error))),
        }
    }
}

#[derive(Clone, Debug)]
pub struct Walker(NonStereoWriterState<NativeSolverState>);

#[derive(Clone, Debug)]
pub struct WalkerChoice {
    pub text: String,
    pub successor: Walker,
}

impl Walker {
    pub fn is_accepted(&self) -> bool {
        self.0.is_accepted()
    }

    pub fn choices(&self) -> Result<Vec<WalkerChoice>, WalkerFailure> {
        self.0
            .choices()
            .map(|choices| {
                choices
                    .into_iter()
                    .map(|choice| WalkerChoice {
                        text: choice.text().to_owned(),
                        successor: Walker(choice.into_successor()),
                    })
                    .collect()
            })
            .map_err(|error| WalkerFailure(FailureCause::Choices(error)))
    }
}

/// Classification used when translating kernel failures across a language boundary.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum WalkerFailureKind {
    InvalidInput,
    Backend,
    Spelling,
    Incomplete,
    Invariant,
}

#[derive(Debug)]
enum FailureCause {
    Preparation(PreparedNonStereoError),
    ContradictoryInput,
    InitialBackend(NativeSolverFailure),
    Choices(ChoiceFailure<NativeSolverFailure>),
}

#[derive(Debug)]
pub struct WalkerFailure(FailureCause);

impl WalkerFailure {
    pub fn kind(&self) -> WalkerFailureKind {
        match &self.0 {
            FailureCause::Preparation(_) | FailureCause::ContradictoryInput => {
                WalkerFailureKind::InvalidInput
            }
            FailureCause::InitialBackend(_) | FailureCause::Choices(ChoiceFailure::Backend(_)) => {
                WalkerFailureKind::Backend
            }
            FailureCause::Choices(ChoiceFailure::Spelling(_)) => WalkerFailureKind::Spelling,
            FailureCause::Choices(ChoiceFailure::Incomplete(_)) => WalkerFailureKind::Incomplete,
            FailureCause::Choices(ChoiceFailure::Invariant(_)) => WalkerFailureKind::Invariant,
        }
    }
}

impl fmt::Display for WalkerFailure {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.0 {
            FailureCause::Preparation(error) => {
                write!(formatter, "Cannot prepare the writer: {error}")
            }
            FailureCause::ContradictoryInput => formatter
                .write_str("Cannot start the writer: the prepared constraints have no assignment"),
            FailureCause::InitialBackend(error) => {
                write!(formatter, "Cannot start the writer: {error}")
            }
            FailureCause::Choices(error) => {
                write!(formatter, "Cannot compute writer choices: {error}")
            }
        }
    }
}

impl Error for WalkerFailure {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match &self.0 {
            FailureCause::Preparation(error) => Some(error),
            FailureCause::ContradictoryInput => None,
            FailureCause::InitialBackend(error) => Some(error),
            FailureCause::Choices(error) => Some(error),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PreparedGraphBuilder;

    fn carbon_carbon_oxygen() -> PreparedWriter {
        let mut graph = PreparedGraphBuilder::new();
        let atoms = (0..3)
            .map(|_| graph.add_atom().unwrap())
            .collect::<Vec<_>>();
        graph.add_bond(atoms[0], atoms[1]).unwrap();
        graph.add_bond(atoms[1], atoms[2]).unwrap();
        PreparedWriter::new(
            graph.build(),
            ["C", "C", "O"]
                .map(|text| AtomToken::Fixed(text.to_owned()))
                .into(),
            vec![BondToken::Elided; 2],
            Vec::new(),
            ComponentOrder::AnyOrder,
        )
        .unwrap()
    }

    #[test]
    fn equal_text_keeps_distinct_root_successors_and_source_unchanged() {
        let source = carbon_carbon_oxygen().initial().unwrap();
        let carbon_choices = source
            .choices()
            .unwrap()
            .into_iter()
            .filter(|choice| choice.text == "C")
            .collect::<Vec<_>>();
        assert_eq!(carbon_choices.len(), 2);
        let next = carbon_choices
            .iter()
            .map(|choice| {
                choice
                    .successor
                    .choices()
                    .unwrap()
                    .into_iter()
                    .map(|choice| choice.text)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        assert_ne!(next[0], next[1]);
        assert_eq!(source.choices().unwrap().len(), 3);
        assert!(!source.is_accepted());
    }

    #[test]
    fn input_order_keeps_every_root_in_the_earliest_prepared_component() {
        let mut graph = PreparedGraphBuilder::new();
        let atoms = (0..3)
            .map(|_| graph.add_atom().unwrap())
            .collect::<Vec<_>>();
        graph.add_bond(atoms[0], atoms[2]).unwrap();
        let prepared = PreparedWriter::new(
            graph.build(),
            ["C", "O", "C"]
                .map(|text| AtomToken::Fixed(text.to_owned()))
                .into(),
            vec![BondToken::Elided],
            Vec::new(),
            ComponentOrder::InputOrder,
        )
        .unwrap();

        let choices = prepared.initial().unwrap().choices().unwrap();
        assert_eq!(choices.len(), 2);
        assert!(choices.iter().all(|choice| choice.text == "C"));
    }

    #[test]
    fn input_order_retains_all_roots_in_the_next_component_after_a_period() {
        let mut graph = PreparedGraphBuilder::new();
        let atoms = (0..5)
            .map(|_| graph.add_atom().unwrap())
            .collect::<Vec<_>>();
        graph.add_bond(atoms[0], atoms[3]).unwrap();
        graph.add_bond(atoms[1], atoms[4]).unwrap();
        let prepared = PreparedWriter::new(
            graph.build(),
            ["C", "O", "N", "C", "O"]
                .map(|text| AtomToken::Fixed(text.to_owned()))
                .into(),
            vec![BondToken::Elided; 2],
            Vec::new(),
            ComponentOrder::InputOrder,
        )
        .unwrap();

        let first_component_root = prepared
            .initial()
            .unwrap()
            .choices()
            .unwrap()
            .into_iter()
            .next()
            .unwrap();
        let first_component_child = first_component_root
            .successor
            .choices()
            .unwrap()
            .into_iter()
            .next()
            .unwrap();
        let separators = first_component_child.successor.choices().unwrap();
        assert_eq!(separators.len(), 2);
        assert!(separators.iter().all(|choice| choice.text == "."));

        for separator in separators {
            let next_component_roots = separator.successor.choices().unwrap();
            assert_eq!(next_component_roots.len(), 1);
            assert_eq!(next_component_roots[0].text, "O");
        }
    }

    #[test]
    fn any_order_retains_roots_from_later_components() {
        let mut graph = PreparedGraphBuilder::new();
        graph.add_atom().unwrap();
        graph.add_atom().unwrap();
        let prepared = PreparedWriter::new(
            graph.build(),
            ["C", "O"]
                .map(|text| AtomToken::Fixed(text.to_owned()))
                .into(),
            Vec::new(),
            Vec::new(),
            ComponentOrder::AnyOrder,
        )
        .unwrap();

        let choices = prepared.initial().unwrap().choices().unwrap();
        assert_eq!(
            choices
                .iter()
                .map(|choice| choice.text.as_str())
                .collect::<Vec<_>>(),
            ["C", "O"]
        );
    }

    #[test]
    fn empty_graph_accepts_without_choices() {
        let prepared = PreparedWriter::new(
            PreparedGraphBuilder::new().build(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            ComponentOrder::AnyOrder,
        )
        .unwrap();
        let initial = prepared.initial().unwrap();
        assert!(initial.is_accepted());
        assert!(initial.choices().unwrap().is_empty());
    }

    #[test]
    fn wrong_atom_count_retains_preparation_cause() {
        let error = PreparedWriter::new(
            PreparedGraphBuilder::new().build(),
            vec![AtomToken::Fixed("C".to_owned())],
            Vec::new(),
            Vec::new(),
            ComponentOrder::AnyOrder,
        )
        .unwrap_err();
        assert_eq!(error.kind(), WalkerFailureKind::InvalidInput);
        assert!(error.source().unwrap().to_string().contains("atom text"));
    }
}
