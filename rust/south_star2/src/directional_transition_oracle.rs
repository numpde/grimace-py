//! Independent one-step qualification for fixed-carrier directional transitions.
//!
//! Expected sign projections are enumerated from primitive fixed-endpoint XOR facts. This module
//! deliberately does not inspect prepared directional metadata, production sign values, solver
//! state, or production frontier derivation.

use std::collections::{BTreeMap, BTreeSet};

use super::*;
use crate::native::NativeSolverState;
use crate::prepared::PreparedGraphBuilder;

type State = NonStereoWriterState<NativeSolverState>;

const TRAVERSAL_PLAN: u8 = 0;
const FIRST_RING_PLAN: u8 = 1;
const LAST_RING_PLAN: u8 = 4;

#[derive(Copy, Clone)]
enum PrimitiveToken {
    Elided,
    Double,
}

impl PrimitiveToken {
    fn production(self) -> NonStereoBondToken {
        match self {
            Self::Elided => NonStereoBondToken::Elided,
            Self::Double => NonStereoBondToken::Double,
        }
    }
}

#[derive(Copy, Clone)]
struct PrimitiveBond {
    a: usize,
    b: usize,
    token: PrimitiveToken,
}

#[derive(Copy, Clone)]
struct PrimitiveRelation {
    double_bond: usize,
    left_endpoint: usize,
    left_carrier: usize,
    right_endpoint: usize,
    right_carrier: usize,
    outward_xor: bool,
}

struct PrimitiveFixture {
    atom_text: &'static [&'static str],
    bonds: &'static [PrimitiveBond],
    relations: &'static [PrimitiveRelation],
}

struct ProductionFixture {
    surface: PreparedNonStereo,
    atoms: Vec<AtomId>,
    bonds: Vec<BondId>,
}

#[derive(Copy, Clone)]
enum SourceStage {
    Inline,
    BranchSign,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum PendingSnapshot {
    InlineAtom {
        parent: usize,
        child: usize,
        bond: usize,
    },
    BranchAtom {
        parent: usize,
        child: usize,
        bond: usize,
    },
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct RoleDomain {
    traversal: bool,
    ring: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct SemanticSuccessor {
    active_atom: Option<usize>,
    visited_atoms: Vec<usize>,
    sign_domains: Vec<(usize, Vec<u8>)>,
    carrier_role: RoleDomain,
    pending: PendingSnapshot,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct VisibleChoice {
    text: String,
    successor: SemanticSuccessor,
}

const ISOLATED_BONDS: [PrimitiveBond; 3] = [
    PrimitiveBond {
        a: 0,
        b: 1,
        token: PrimitiveToken::Elided,
    },
    PrimitiveBond {
        a: 1,
        b: 2,
        token: PrimitiveToken::Double,
    },
    PrimitiveBond {
        a: 2,
        b: 3,
        token: PrimitiveToken::Elided,
    },
];

const ISOLATED_RELATIONS: [PrimitiveRelation; 1] = [PrimitiveRelation {
    double_bond: 1,
    left_endpoint: 1,
    left_carrier: 0,
    right_endpoint: 2,
    right_carrier: 2,
    outward_xor: false,
}];

const ISOLATED: PrimitiveFixture = PrimitiveFixture {
    atom_text: &["F", "L", "R", "Cl"],
    bonds: &ISOLATED_BONDS,
    relations: &ISOLATED_RELATIONS,
};

const SHARED_BONDS: [PrimitiveBond; 5] = [
    PrimitiveBond {
        a: 0,
        b: 1,
        token: PrimitiveToken::Elided,
    },
    PrimitiveBond {
        a: 1,
        b: 2,
        token: PrimitiveToken::Double,
    },
    PrimitiveBond {
        a: 2,
        b: 3,
        token: PrimitiveToken::Elided,
    },
    PrimitiveBond {
        a: 3,
        b: 4,
        token: PrimitiveToken::Double,
    },
    PrimitiveBond {
        a: 4,
        b: 5,
        token: PrimitiveToken::Elided,
    },
];

const SHARED_RELATIONS: [PrimitiveRelation; 2] = [
    PrimitiveRelation {
        double_bond: 1,
        left_endpoint: 1,
        left_carrier: 0,
        right_endpoint: 2,
        right_carrier: 2,
        outward_xor: false,
    },
    PrimitiveRelation {
        double_bond: 3,
        left_endpoint: 3,
        left_carrier: 2,
        right_endpoint: 4,
        right_carrier: 4,
        outward_xor: true,
    },
];

const SHARED: PrimitiveFixture = PrimitiveFixture {
    atom_text: &["A", "B", "C", "D", "E", "F"],
    bonds: &SHARED_BONDS,
    relations: &SHARED_RELATIONS,
};

fn production_fixture(spec: &PrimitiveFixture) -> ProductionFixture {
    let mut graph = PreparedGraphBuilder::new();
    let atoms = spec
        .atom_text
        .iter()
        .map(|_| graph.add_atom().unwrap())
        .collect::<Vec<_>>();
    let bonds = spec
        .bonds
        .iter()
        .map(|bond| graph.add_bond(atoms[bond.a], atoms[bond.b]).unwrap())
        .collect::<Vec<_>>();
    let relations = spec
        .relations
        .iter()
        .map(|relation| PreparedDirectionalRelation {
            double_bond: bonds[relation.double_bond],
            left_endpoint: atoms[relation.left_endpoint],
            left_carriers: vec![bonds[relation.left_carrier]].into_boxed_slice(),
            right_endpoint: atoms[relation.right_endpoint],
            right_carriers: vec![bonds[relation.right_carrier]].into_boxed_slice(),
            outward_sign_xor: relation.outward_xor,
        })
        .collect();
    let surface = PreparedNonStereo::with_atom_tokens_and_directional(
        PreparedMolecule::new(graph.build()),
        spec.atom_text
            .iter()
            .map(|text| PreparedAtomToken::Fixed((*text).to_owned()))
            .collect(),
        spec.bonds
            .iter()
            .map(|bond| bond.token.production())
            .collect(),
        relations,
    )
    .unwrap();
    ProductionFixture {
        surface,
        atoms,
        bonds,
    }
}

fn carrier_indices(spec: &PrimitiveFixture) -> Vec<usize> {
    spec.relations
        .iter()
        .flat_map(|relation| [relation.left_carrier, relation.right_carrier])
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

fn endpoint_is_fixed_b(spec: &PrimitiveFixture, bond: usize, endpoint: usize) -> bool {
    let bond = spec.bonds[bond];
    if endpoint == bond.a {
        false
    } else if endpoint == bond.b {
        true
    } else {
        panic!("primitive directional endpoint must be incident to its carrier")
    }
}

fn canonical_assignments(spec: &PrimitiveFixture) -> Vec<BTreeMap<usize, u8>> {
    let carriers = carrier_indices(spec);
    let positions = carriers
        .iter()
        .copied()
        .enumerate()
        .map(|(position, carrier)| (carrier, position))
        .collect::<BTreeMap<_, _>>();
    let mut assignments = Vec::new();

    for bits in 0..(1_usize << carriers.len()) {
        let assignment = carriers
            .iter()
            .copied()
            .map(|carrier| (carrier, ((bits >> positions[&carrier]) & 1) as u8))
            .collect::<BTreeMap<_, _>>();
        let consistent = spec.relations.iter().all(|relation| {
            let canonical_xor = u8::from(relation.outward_xor)
                ^ u8::from(endpoint_is_fixed_b(
                    spec,
                    relation.left_carrier,
                    relation.left_endpoint,
                ))
                ^ u8::from(endpoint_is_fixed_b(
                    spec,
                    relation.right_carrier,
                    relation.right_endpoint,
                ));
            assignment[&relation.left_carrier] ^ assignment[&relation.right_carrier]
                == canonical_xor
        });
        if consistent {
            assignments.push(assignment);
        }
    }
    assignments
}

fn projected_sign_domains(
    carriers: &[usize],
    assignments: &[BTreeMap<usize, u8>],
) -> Vec<(usize, Vec<u8>)> {
    carriers
        .iter()
        .copied()
        .map(|carrier| {
            let values = assignments
                .iter()
                .map(|assignment| assignment[&carrier])
                .collect::<BTreeSet<_>>()
                .into_iter()
                .collect();
            (carrier, values)
        })
        .collect()
}

fn fixed_endpoint_text(spec: &PrimitiveFixture, bond: usize, from: usize, value: u8) -> String {
    assert!(value <= 1, "canonical sign values are binary");
    let from_fixed_b = endpoint_is_fixed_b(spec, bond, from);
    if value ^ u8::from(from_fixed_b) == 0 {
        "/".to_owned()
    } else {
        "\\".to_owned()
    }
}

fn expected_choices(
    spec: &PrimitiveFixture,
    from: usize,
    carrier: usize,
    stage: SourceStage,
) -> BTreeSet<VisibleChoice> {
    let bond = spec.bonds[carrier];
    let child = if from == bond.a {
        bond.b
    } else if from == bond.b {
        bond.a
    } else {
        panic!("one-step source must be an endpoint of its carrier")
    };
    let carriers = carrier_indices(spec);
    let assignments = canonical_assignments(spec);

    (0_u8..=1)
        .map(|selected| {
            let survivors = assignments
                .iter()
                .filter(|assignment| assignment[&carrier] == selected)
                .cloned()
                .collect::<Vec<_>>();
            assert!(
                !survivors.is_empty(),
                "each canonical carrier sign must retain one XOR phase"
            );
            let pending = match stage {
                SourceStage::Inline => PendingSnapshot::InlineAtom {
                    parent: from,
                    child,
                    bond: carrier,
                },
                SourceStage::BranchSign => PendingSnapshot::BranchAtom {
                    parent: from,
                    child,
                    bond: carrier,
                },
            };
            VisibleChoice {
                text: fixed_endpoint_text(spec, carrier, from, selected),
                successor: SemanticSuccessor {
                    active_atom: Some(from),
                    visited_atoms: vec![from],
                    sign_domains: projected_sign_domains(&carriers, &survivors),
                    carrier_role: RoleDomain {
                        traversal: true,
                        ring: false,
                    },
                    pending,
                },
            }
        })
        .collect()
}

fn atom_index(production: &ProductionFixture, atom: AtomId) -> usize {
    production
        .atoms
        .iter()
        .position(|candidate| *candidate == atom)
        .expect("observed atom must belong to the primitive fixture")
}

fn bond_index(production: &ProductionFixture, bond: BondId) -> usize {
    production
        .bonds
        .iter()
        .position(|candidate| *candidate == bond)
        .expect("observed bond must belong to the primitive fixture")
}

fn observed_sign_domains(
    production: &ProductionFixture,
    observed: &ObservedNonStereoState,
) -> Vec<(usize, Vec<u8>)> {
    observed
        .directional_sign_domains
        .iter()
        .map(|(bond, domain)| {
            (
                bond_index(production, *bond),
                domain.iter().collect::<Vec<_>>(),
            )
        })
        .collect()
}

fn observed_role_domain(domain: Domain) -> RoleDomain {
    assert!(
        domain
            .iter()
            .all(|value| (TRAVERSAL_PLAN..=LAST_RING_PLAN).contains(&value)),
        "raw bond-plan observation contained an unknown representation value"
    );
    RoleDomain {
        traversal: domain.contains(TRAVERSAL_PLAN),
        ring: (FIRST_RING_PLAN..=LAST_RING_PLAN).any(|value| domain.contains(value)),
    }
}

fn observed_pending(production: &ProductionFixture, pending: &ObservedPending) -> PendingSnapshot {
    match pending {
        ObservedPending::InlineAtom {
            parent,
            child,
            bond,
        } => PendingSnapshot::InlineAtom {
            parent: atom_index(production, *parent),
            child: atom_index(production, *child),
            bond: bond_index(production, *bond),
        },
        ObservedPending::BranchAtom {
            parent,
            child,
            bond,
        } => PendingSnapshot::BranchAtom {
            parent: atom_index(production, *parent),
            child: atom_index(production, *child),
            bond: bond_index(production, *bond),
        },
        other => panic!("unexpected directional successor pending stage: {other:?}"),
    }
}

fn observed_choice(
    production: &ProductionFixture,
    carrier: usize,
    choice: &Choice<State>,
) -> VisibleChoice {
    let observed = choice.successor().observe_raw();
    VisibleChoice {
        text: choice.text().to_owned(),
        successor: SemanticSuccessor {
            active_atom: observed
                .structural
                .traversal
                .active_frame
                .as_ref()
                .map(|frame| atom_index(production, frame.atom)),
            visited_atoms: observed
                .structural
                .traversal
                .visited_atoms
                .iter()
                .map(|atom| atom_index(production, *atom))
                .collect(),
            sign_domains: observed_sign_domains(production, &observed),
            carrier_role: observed_role_domain(
                observed.structural.bond_plan_domains[production.bonds[carrier].index()],
            ),
            pending: observed_pending(
                production,
                observed
                    .pending
                    .as_ref()
                    .expect("directional sign token must leave its atom pending"),
            ),
        },
    }
}

fn rooted_state(production: &ProductionFixture, root: usize) -> State {
    let initial = State::initial(&production.surface)
        .unwrap()
        .unwrap_consistent();
    let before = initial.observe_raw();
    let mut roots = initial
        .choices()
        .unwrap()
        .into_iter()
        .filter(|choice| choice.text() == production.surface.atom_text(production.atoms[root]))
        .collect::<Vec<_>>();
    assert_eq!(
        initial.observe_raw(),
        before,
        "root choices mutated their source"
    );
    assert_eq!(roots.len(), 1, "fixture atom text must select one root");
    roots.pop().unwrap().into_successor()
}

fn directional_source(
    production: &ProductionFixture,
    root: usize,
    carrier: usize,
    stage: SourceStage,
) -> State {
    let rooted = rooted_state(production, root);
    match stage {
        SourceStage::Inline => {
            assert!(rooted.observe_raw().pending.is_none());
            rooted
        }
        SourceStage::BranchSign => {
            let before = rooted.observe_raw();
            let mut branches = rooted
                .choices()
                .unwrap()
                .into_iter()
                .filter(|choice| {
                    if choice.text() != "(" {
                        return false;
                    }
                    matches!(
                        choice.successor().observe_raw().pending,
                        Some(ObservedPending::BranchDirectionalSign { bond, .. })
                            if bond == production.bonds[carrier]
                    )
                })
                .collect::<Vec<_>>();
            assert_eq!(
                rooted.observe_raw(),
                before,
                "branch choices mutated their source"
            );
            assert_eq!(branches.len(), 1, "shared carrier branch must be unique");
            branches.pop().unwrap().into_successor()
        }
    }
}

fn assert_one_step(spec: &PrimitiveFixture, root: usize, carrier: usize, stage: SourceStage) {
    let production = production_fixture(spec);
    let source = directional_source(&production, root, carrier, stage);
    let source_before = source.observe_raw();
    let carriers = carrier_indices(spec);
    assert_eq!(
        observed_sign_domains(&production, &source_before),
        projected_sign_domains(&carriers, &canonical_assignments(spec)),
        "source sign domains must be the independent XOR projection"
    );
    assert_eq!(
        observed_role_domain(
            source_before.structural.bond_plan_domains[production.bonds[carrier].index()]
        ),
        RoleDomain {
            traversal: true,
            ring: false,
        }
    );

    let choices = source.choices().unwrap();
    assert_eq!(
        source.observe_raw(),
        source_before,
        "directional choices mutated their source"
    );
    let actual = choices
        .iter()
        .map(|choice| observed_choice(&production, carrier, choice))
        .collect::<BTreeSet<_>>();
    assert_eq!(actual, expected_choices(spec, root, carrier, stage));
}

#[test]
fn isolated_fixed_carrier_chain_matches_forward_and_reverse_one_step_projections() {
    assert_one_step(&ISOLATED, 0, 0, SourceStage::Inline);
    assert_one_step(&ISOLATED, 3, 2, SourceStage::Inline);
}

#[test]
fn shared_site_chain_matches_forward_and_reverse_one_step_projections() {
    assert_one_step(&SHARED, 2, 2, SourceStage::BranchSign);
    assert_one_step(&SHARED, 3, 2, SourceStage::BranchSign);
}
