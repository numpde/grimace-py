//! Internal cross-crate interface for native successor-bearing writer choices.
//!
//! Preparation supplies chemical spelling and stereo facts. This interface
//! preserves kernel outcomes while keeping traversal and solver stores private.

use std::collections::{BTreeSet, VecDeque};
use std::error::Error;
use std::fmt;

use crate::native::NativeSolverState;
use crate::native_solver::NativeSolverFailure;
use crate::nonstereo_writer::{
    ChoiceFailure, NonStereoWriterState, PreparedNonStereo, PreparedNonStereoError,
    WriterIncompleteness,
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

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum WriterRegime {
    Generic { component_order: ComponentOrder },
    RdkitRandomNoncanonical,
}

impl WriterRegime {
    pub(crate) const fn component_order(self) -> ComponentOrder {
        match self {
            Self::Generic { component_order } => component_order,
            Self::RdkitRandomNoncanonical => ComponentOrder::InputOrder,
        }
    }
}

#[derive(Clone, Debug)]
pub struct PreparedWriter {
    surface: PreparedNonStereo,
    regime: WriterRegime,
}

impl PreparedWriter {
    pub fn new(
        graph: PreparedGraph,
        atoms: Vec<AtomToken>,
        bonds: Vec<BondToken>,
        directional: Vec<DirectionalRelation>,
        regime: WriterRegime,
    ) -> Result<Self, WalkerFailure> {
        let has_directional_relations = !directional.is_empty();
        let surface = PreparedNonStereo::with_atom_tokens_and_directional(
            PreparedMolecule::new(graph),
            atoms,
            bonds,
            directional.clone(),
        )
        .map_err(|error| WalkerFailure(FailureCause::Preparation(error)))?;
        if matches!(regime, WriterRegime::RdkitRandomNoncanonical) && has_directional_relations {
            let isolated_relations = directional_relations_are_isolated(&directional);
            let acyclic_directional_bonds =
                directional_bonds_are_acyclic(surface.molecule().graph(), &directional);
            if !isolated_relations || !acyclic_directional_bonds {
                return Err(WalkerFailure(FailureCause::RegimeIncomplete(
                    WriterIncompleteness::RdkitDirectionalAnchoring,
                )));
            }
        }
        Ok(Self { surface, regime })
    }

    pub fn initial(&self) -> Result<Walker, WalkerFailure> {
        match NonStereoWriterState::initial_with_regime(&self.surface, self.regime) {
            Ok(Consistency::Consistent(state)) => Ok(Walker(state)),
            Ok(Consistency::Contradiction) => Err(WalkerFailure(FailureCause::ContradictoryInput)),
            Err(error) => Err(WalkerFailure(FailureCause::InitialBackend(error))),
        }
    }
}

fn directional_relations_are_isolated(relations: &[DirectionalRelation]) -> bool {
    let mut carrier_bonds = BTreeSet::new();
    let mut endpoints = BTreeSet::new();
    relations.iter().all(|relation| {
        relation.left_carriers.len() == 1
            && relation.right_carriers.len() == 1
            && endpoints.insert(relation.left_endpoint)
            && endpoints.insert(relation.right_endpoint)
            && carrier_bonds.insert(relation.left_carriers[0].bond)
            && carrier_bonds.insert(relation.right_carriers[0].bond)
    })
}

fn directional_bonds_are_acyclic(graph: &PreparedGraph, relations: &[DirectionalRelation]) -> bool {
    relations.iter().all(|relation| {
        std::iter::once(relation.double_bond)
            .chain(
                relation
                    .left_carriers
                    .iter()
                    .chain(relation.right_carriers.iter())
                    .map(|carrier| carrier.bond),
            )
            .all(|bond| is_bridge(graph, bond))
    })
}

fn is_bridge(graph: &PreparedGraph, bond: crate::BondId) -> bool {
    let topology = graph
        .bond(bond)
        .expect("prepared directional bond must belong to the graph");
    let start = topology.a();
    let target = topology.b();
    let mut visited = BTreeSet::from([start]);
    let mut pending = VecDeque::from([start]);
    while let Some(atom) = pending.pop_front() {
        for incident in graph
            .neighbors(atom)
            .expect("prepared directional atom must have an adjacency row")
        {
            if incident.bond() == bond {
                continue;
            }
            if incident.atom() == target {
                return false;
            }
            if visited.insert(incident.atom()) {
                pending.push_back(incident.atom());
            }
        }
    }
    true
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
    RegimeIncomplete(WriterIncompleteness),
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
            FailureCause::RegimeIncomplete(_) => WalkerFailureKind::Incomplete,
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
            FailureCause::RegimeIncomplete(error) => {
                write!(
                    formatter,
                    "Cannot prepare the RDKit random noncanonical writer: {error}"
                )
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
            FailureCause::RegimeIncomplete(error) => Some(error),
            FailureCause::ContradictoryInput => None,
            FailureCause::InitialBackend(error) => Some(error),
            FailureCause::Choices(error) => Some(error),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nonstereo_writer::{PreparedDirectionalCarrier, PreparedDirectionalRelation};
    use crate::PreparedGraphBuilder;
    use std::collections::BTreeSet;

    fn alkene_support(
        regime: WriterRegime,
        side_phase_xor: bool,
        reverse_left_carrier: bool,
        reverse_right_carrier: bool,
        left_sibling: bool,
    ) -> BTreeSet<String> {
        fn collect(walker: Walker, prefix: String, support: &mut BTreeSet<String>) {
            if walker.is_accepted() {
                support.insert(prefix);
                return;
            }
            for choice in walker.choices().unwrap() {
                collect(
                    choice.successor,
                    format!("{prefix}{}", choice.text),
                    support,
                );
            }
        }

        let mut graph = PreparedGraphBuilder::new();
        let atoms = (0..if left_sibling { 5 } else { 4 })
            .map(|_| graph.add_atom().unwrap())
            .collect::<Vec<_>>();
        let left_carrier = if reverse_left_carrier {
            graph.add_bond(atoms[1], atoms[0]).unwrap()
        } else {
            graph.add_bond(atoms[0], atoms[1]).unwrap()
        };
        let double_bond = graph.add_bond(atoms[1], atoms[2]).unwrap();
        let right_carrier = if reverse_right_carrier {
            graph.add_bond(atoms[3], atoms[2]).unwrap()
        } else {
            graph.add_bond(atoms[2], atoms[3]).unwrap()
        };
        let mut bonds = vec![BondToken::Elided, BondToken::Double, BondToken::Elided];
        if left_sibling {
            graph.add_bond(atoms[1], atoms[4]).unwrap();
            bonds.push(BondToken::Elided);
        }
        let relation = PreparedDirectionalRelation {
            double_bond,
            left_endpoint: atoms[1],
            left_carriers: Box::new([PreparedDirectionalCarrier {
                bond: left_carrier,
                side_flip: false,
            }]),
            right_endpoint: atoms[2],
            right_carriers: Box::new([PreparedDirectionalCarrier {
                bond: right_carrier,
                side_flip: false,
            }]),
            side_phase_xor,
        };
        let prepared = PreparedWriter::new(
            graph.build(),
            vec![AtomToken::Fixed("C".to_owned()); atoms.len()],
            bonds,
            vec![relation],
            regime,
        )
        .unwrap();
        let mut support = BTreeSet::new();
        collect(prepared.initial().unwrap(), String::new(), &mut support);
        support
    }

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
            WriterRegime::Generic {
                component_order: ComponentOrder::AnyOrder,
            },
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
            WriterRegime::Generic {
                component_order: ComponentOrder::InputOrder,
            },
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
            WriterRegime::Generic {
                component_order: ComponentOrder::InputOrder,
            },
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
            WriterRegime::Generic {
                component_order: ComponentOrder::AnyOrder,
            },
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
            WriterRegime::Generic {
                component_order: ComponentOrder::AnyOrder,
            },
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
            WriterRegime::Generic {
                component_order: ComponentOrder::AnyOrder,
            },
        )
        .unwrap_err();
        assert_eq!(error.kind(), WalkerFailureKind::InvalidInput);
        assert!(error.source().unwrap().to_string().contains("atom text"));
    }

    #[test]
    fn rdkit_regime_validates_directional_relations_before_other_incompleteness() {
        let prepare = |malformed_relation: bool| {
            let mut graph = PreparedGraphBuilder::new();
            let atoms = (0..4)
                .map(|_| graph.add_atom().unwrap())
                .collect::<Vec<_>>();
            let double_bond = graph.add_bond(atoms[0], atoms[1]).unwrap();
            let left_carrier = graph.add_bond(atoms[0], atoms[2]).unwrap();
            let right_carrier = graph.add_bond(atoms[1], atoms[3]).unwrap();
            let relation = PreparedDirectionalRelation {
                double_bond,
                left_endpoint: if malformed_relation {
                    atoms[2]
                } else {
                    atoms[0]
                },
                left_carriers: Box::new([PreparedDirectionalCarrier {
                    bond: left_carrier,
                    side_flip: false,
                }]),
                right_endpoint: atoms[1],
                right_carriers: Box::new([PreparedDirectionalCarrier {
                    bond: right_carrier,
                    side_flip: false,
                }]),
                side_phase_xor: false,
            };
            PreparedWriter::new(
                graph.build(),
                ["C", "C", "C", "C"]
                    .map(|text| AtomToken::Fixed(text.to_owned()))
                    .into(),
                vec![BondToken::Double, BondToken::Elided, BondToken::Elided],
                vec![relation],
                WriterRegime::RdkitRandomNoncanonical,
            )
        };

        let malformed = prepare(true).unwrap_err();
        assert_eq!(malformed.kind(), WalkerFailureKind::InvalidInput);
        assert!(malformed.to_string().contains("does not join"));

        assert!(prepare(false).is_ok());
    }

    #[test]
    fn rdkit_regime_keeps_multiple_carrier_anchoring_incomplete() {
        let mut graph = PreparedGraphBuilder::new();
        let atoms = (0..5)
            .map(|_| graph.add_atom().unwrap())
            .collect::<Vec<_>>();
        let double_bond = graph.add_bond(atoms[0], atoms[1]).unwrap();
        let left_carrier_a = graph.add_bond(atoms[0], atoms[2]).unwrap();
        let left_carrier_b = graph.add_bond(atoms[0], atoms[3]).unwrap();
        let right_carrier = graph.add_bond(atoms[1], atoms[4]).unwrap();
        let relation = PreparedDirectionalRelation {
            double_bond,
            left_endpoint: atoms[0],
            left_carriers: Box::new([
                PreparedDirectionalCarrier {
                    bond: left_carrier_a,
                    side_flip: false,
                },
                PreparedDirectionalCarrier {
                    bond: left_carrier_b,
                    side_flip: false,
                },
            ]),
            right_endpoint: atoms[1],
            right_carriers: Box::new([PreparedDirectionalCarrier {
                bond: right_carrier,
                side_flip: false,
            }]),
            side_phase_xor: true,
        };
        let error = PreparedWriter::new(
            graph.build(),
            ["C", "C", "C", "C", "C"]
                .map(|text| AtomToken::Fixed(text.to_owned()))
                .into(),
            vec![
                BondToken::Double,
                BondToken::Elided,
                BondToken::Elided,
                BondToken::Elided,
            ],
            vec![relation],
            WriterRegime::RdkitRandomNoncanonical,
        )
        .unwrap_err();

        assert_eq!(error.kind(), WalkerFailureKind::Incomplete);
        assert!(error.to_string().contains("one carrier per endpoint"));
    }

    #[test]
    fn rdkit_regime_keeps_shared_endpoint_anchoring_incomplete() {
        let mut graph = PreparedGraphBuilder::new();
        let atoms = (0..7)
            .map(|_| graph.add_atom().unwrap())
            .collect::<Vec<_>>();
        let first_double = graph.add_bond(atoms[0], atoms[1]).unwrap();
        let first_left = graph.add_bond(atoms[0], atoms[3]).unwrap();
        let first_right = graph.add_bond(atoms[1], atoms[5]).unwrap();
        let second_double = graph.add_bond(atoms[1], atoms[2]).unwrap();
        let second_left = graph.add_bond(atoms[1], atoms[6]).unwrap();
        let second_right = graph.add_bond(atoms[2], atoms[4]).unwrap();
        let carrier = |bond| PreparedDirectionalCarrier {
            bond,
            side_flip: false,
        };
        let relation = |double_bond, left_endpoint, left_carrier, right_endpoint, right_carrier| {
            PreparedDirectionalRelation {
                double_bond,
                left_endpoint,
                left_carriers: Box::new([carrier(left_carrier)]),
                right_endpoint,
                right_carriers: Box::new([carrier(right_carrier)]),
                side_phase_xor: true,
            }
        };
        let relations = vec![
            relation(first_double, atoms[0], first_left, atoms[1], first_right),
            relation(second_double, atoms[1], second_left, atoms[2], second_right),
        ];
        let error = PreparedWriter::new(
            graph.build(),
            vec![AtomToken::Fixed("C".to_owned()); atoms.len()],
            vec![
                BondToken::Double,
                BondToken::Elided,
                BondToken::Elided,
                BondToken::Double,
                BondToken::Elided,
                BondToken::Elided,
            ],
            relations,
            WriterRegime::RdkitRandomNoncanonical,
        )
        .unwrap_err();

        assert_eq!(error.kind(), WalkerFailureKind::Incomplete);
        assert!(error.to_string().contains("distinct endpoints"));
    }

    #[test]
    fn rdkit_regime_keeps_shared_carrier_anchoring_incomplete() {
        let mut graph = PreparedGraphBuilder::new();
        let atoms = (0..6)
            .map(|_| graph.add_atom().unwrap())
            .collect::<Vec<_>>();
        let first_double = graph.add_bond(atoms[0], atoms[1]).unwrap();
        let shared_carrier = graph.add_bond(atoms[0], atoms[2]).unwrap();
        let first_other_carrier = graph.add_bond(atoms[1], atoms[4]).unwrap();
        let second_double = graph.add_bond(atoms[2], atoms[3]).unwrap();
        let second_other_carrier = graph.add_bond(atoms[3], atoms[5]).unwrap();
        let carrier = |bond| PreparedDirectionalCarrier {
            bond,
            side_flip: false,
        };
        let relation = |double_bond, left_endpoint, left_carrier, right_endpoint, right_carrier| {
            PreparedDirectionalRelation {
                double_bond,
                left_endpoint,
                left_carriers: Box::new([carrier(left_carrier)]),
                right_endpoint,
                right_carriers: Box::new([carrier(right_carrier)]),
                side_phase_xor: true,
            }
        };
        let relations = vec![
            relation(
                first_double,
                atoms[0],
                shared_carrier,
                atoms[1],
                first_other_carrier,
            ),
            relation(
                second_double,
                atoms[2],
                shared_carrier,
                atoms[3],
                second_other_carrier,
            ),
        ];
        let error = PreparedWriter::new(
            graph.build(),
            vec![AtomToken::Fixed("C".to_owned()); atoms.len()],
            vec![
                BondToken::Double,
                BondToken::Elided,
                BondToken::Elided,
                BondToken::Double,
                BondToken::Elided,
            ],
            relations,
            WriterRegime::RdkitRandomNoncanonical,
        )
        .unwrap_err();

        assert_eq!(error.kind(), WalkerFailureKind::Incomplete);
        assert!(error.to_string().contains("carrier bonds across relations"));
    }

    #[test]
    fn rdkit_regime_keeps_ring_anchoring_incomplete() {
        let mut graph = PreparedGraphBuilder::new();
        let atoms = (0..4)
            .map(|_| graph.add_atom().unwrap())
            .collect::<Vec<_>>();
        let double_bond = graph.add_bond(atoms[0], atoms[1]).unwrap();
        let right_carrier = graph.add_bond(atoms[1], atoms[2]).unwrap();
        graph.add_bond(atoms[2], atoms[3]).unwrap();
        let left_carrier = graph.add_bond(atoms[3], atoms[0]).unwrap();
        let relation = PreparedDirectionalRelation {
            double_bond,
            left_endpoint: atoms[0],
            left_carriers: Box::new([PreparedDirectionalCarrier {
                bond: left_carrier,
                side_flip: false,
            }]),
            right_endpoint: atoms[1],
            right_carriers: Box::new([PreparedDirectionalCarrier {
                bond: right_carrier,
                side_flip: false,
            }]),
            side_phase_xor: true,
        };
        let error = PreparedWriter::new(
            graph.build(),
            vec![AtomToken::Fixed("C".to_owned()); 4],
            vec![
                BondToken::Double,
                BondToken::Elided,
                BondToken::Elided,
                BondToken::Elided,
            ],
            vec![relation],
            WriterRegime::RdkitRandomNoncanonical,
        )
        .unwrap_err();

        assert_eq!(error.kind(), WalkerFailureKind::Incomplete);
        assert!(error
            .to_string()
            .contains("acyclic directional double/carrier bonds"));
    }

    #[test]
    fn rdkit_anchor_removes_only_the_simultaneous_sign_inversion() {
        let generic = alkene_support(
            WriterRegime::Generic {
                component_order: ComponentOrder::AnyOrder,
            },
            true,
            false,
            false,
            false,
        );
        let rdkit = alkene_support(
            WriterRegime::RdkitRandomNoncanonical,
            true,
            false,
            false,
            false,
        );

        assert_eq!(generic.len(), 6);
        assert_eq!(rdkit.len(), 3);
        assert!(generic.is_superset(&rdkit));
    }

    #[test]
    fn rdkit_directional_anchor_preserves_e_z_discrimination() {
        let e_support = alkene_support(
            WriterRegime::RdkitRandomNoncanonical,
            true,
            false,
            false,
            false,
        );
        let z_support = alkene_support(
            WriterRegime::RdkitRandomNoncanonical,
            false,
            false,
            false,
            false,
        );

        assert_eq!(e_support.len(), 3);
        assert_eq!(z_support.len(), 3);
        assert!(e_support.is_disjoint(&z_support));
    }

    #[test]
    fn rdkit_anchor_tracks_stored_carrier_orientation_and_all_root_orientations() {
        let original = alkene_support(
            WriterRegime::RdkitRandomNoncanonical,
            true,
            false,
            false,
            false,
        );
        for (reverse_left, reverse_right) in [(true, false), (false, true), (true, true)] {
            assert_eq!(
                alkene_support(
                    WriterRegime::RdkitRandomNoncanonical,
                    true,
                    reverse_left,
                    reverse_right,
                    false,
                ),
                original,
                "stored carrier orientation {reverse_left}/{reverse_right} changed support"
            );
        }
    }

    #[test]
    fn rdkit_anchor_persists_across_ordinary_sibling_choices() {
        let generic = alkene_support(
            WriterRegime::Generic {
                component_order: ComponentOrder::AnyOrder,
            },
            true,
            false,
            false,
            true,
        );
        let rdkit = alkene_support(
            WriterRegime::RdkitRandomNoncanonical,
            true,
            false,
            false,
            true,
        );

        assert!(!rdkit.is_empty());
        assert!(generic.is_superset(&rdkit));
        assert!(rdkit.len() < generic.len());
    }
}
