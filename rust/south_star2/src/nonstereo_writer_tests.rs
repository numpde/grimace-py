use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use super::*;
use crate::native::NativeSolverState;
use crate::native_solver::NativeSolverFailure;
use crate::prepared::PreparedGraphBuilder;
use crate::traversal::ObservedBondProgress;

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
struct FailFirstRingAlternativeSolver(NativeSolverState);

impl ConstraintSolver for FailFirstRingAlternativeSolver {
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
        if matches!(
            restrictions,
            [(variable, domain)]
                if *variable == crate::ids::VariableId::new(0)
                    && *domain == BondRepresentation::Ring01.singleton_domain()
        ) {
            return Err(InjectedSolverFailure::Restriction);
        }
        assert!(
            !matches!(
                restrictions,
                [(variable, domain)]
                    if *variable == crate::ids::VariableId::new(0)
                        && *domain
                            == BondRepresentation::Ring10
                                .singleton_domain()
                                .union(BondRepresentation::Ring11.singleton_domain())
            ),
            "choice generation must stop after the first backend failure"
        );
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

// These deliberate contract violations exercise writer failure classification,
// not solver-backend conformance.
#[derive(Clone, Debug)]
struct NonconformingRejectTetrahedralParitySolver(NativeSolverState);

impl ConstraintSolver for NonconformingRejectTetrahedralParitySolver {
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
            .any(|(variable, _)| self.0.domain(*variable) == Some(full_order_domain()))
        {
            return Ok(Consistency::Contradiction);
        }
        Ok(
            <NativeSolverState as ConstraintSolver>::restricted(&self.0, restrictions)
                .map_err(InjectedSolverFailure::Native)?
                .map(Self),
        )
    }

    fn transitioned(
        &self,
        restrictions: &[(crate::ids::VariableId, crate::domain::Domain)],
        activate: &[crate::ids::FactorId],
    ) -> Result<Consistency<Self>, Self::Failure> {
        if restrictions
            .iter()
            .any(|(variable, _)| self.0.domain(*variable) == Some(full_order_domain()))
        {
            return Ok(Consistency::Contradiction);
        }
        Ok(
            <NativeSolverState as ConstraintSolver>::transitioned(&self.0, restrictions, activate)
                .map_err(InjectedSolverFailure::Native)?
                .map(Self),
        )
    }

    fn domain(&self, variable: crate::ids::VariableId) -> Option<crate::domain::Domain> {
        self.0.domain(variable)
    }

    fn factor_is_active(&self, factor: crate::ids::FactorId) -> Option<bool> {
        self.0.factor_is_active(factor)
    }
}

#[derive(Clone, Debug)]
struct NonconformingRejectEvenTetrahedralParitySolver(NativeSolverState);

impl ConstraintSolver for NonconformingRejectEvenTetrahedralParitySolver {
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
        if restrictions.iter().any(|(variable, domain)| {
            self.0.domain(*variable) == Some(full_order_domain())
                && domain.is_subset_of(parity_domain(TetrahedralParity::Even))
        }) {
            return Ok(Consistency::Contradiction);
        }
        Ok(
            <NativeSolverState as ConstraintSolver>::restricted(&self.0, restrictions)
                .map_err(InjectedSolverFailure::Native)?
                .map(Self),
        )
    }

    fn transitioned(
        &self,
        restrictions: &[(crate::ids::VariableId, crate::domain::Domain)],
        activate: &[crate::ids::FactorId],
    ) -> Result<Consistency<Self>, Self::Failure> {
        if restrictions.iter().any(|(variable, domain)| {
            self.0.domain(*variable) == Some(full_order_domain())
                && domain.is_subset_of(parity_domain(TetrahedralParity::Even))
        }) {
            return Ok(Consistency::Contradiction);
        }
        Ok(
            <NativeSolverState as ConstraintSolver>::transitioned(&self.0, restrictions, activate)
                .map_err(InjectedSolverFailure::Native)?
                .map(Self),
        )
    }

    fn domain(&self, variable: crate::ids::VariableId) -> Option<crate::domain::Domain> {
        self.0.domain(variable)
    }

    fn factor_is_active(&self, factor: crate::ids::FactorId) -> Option<bool> {
        self.0.factor_is_active(factor)
    }
}

#[derive(Clone, Debug)]
struct FailTetrahedralRestrictionSolver(NativeSolverState);

static TETRAHEDRAL_BACKEND_ATTEMPTS: AtomicUsize = AtomicUsize::new(0);

impl ConstraintSolver for FailTetrahedralRestrictionSolver {
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
            .any(|(variable, _)| self.0.domain(*variable) == Some(full_order_domain()))
        {
            assert_eq!(
                TETRAHEDRAL_BACKEND_ATTEMPTS.fetch_add(1, Ordering::Relaxed),
                0,
                "atom-token generation must stop at the first backend failure"
            );
            return Err(InjectedSolverFailure::Restriction);
        }
        Ok(
            <NativeSolverState as ConstraintSolver>::restricted(&self.0, restrictions)
                .map_err(InjectedSolverFailure::Native)?
                .map(Self),
        )
    }

    fn transitioned(
        &self,
        restrictions: &[(crate::ids::VariableId, crate::domain::Domain)],
        activate: &[crate::ids::FactorId],
    ) -> Result<Consistency<Self>, Self::Failure> {
        if restrictions
            .iter()
            .any(|(variable, _)| self.0.domain(*variable) == Some(full_order_domain()))
        {
            assert_eq!(
                TETRAHEDRAL_BACKEND_ATTEMPTS.fetch_add(1, Ordering::Relaxed),
                0,
                "atom-token generation must stop at the first backend failure"
            );
            return Err(InjectedSolverFailure::Restriction);
        }
        Ok(
            <NativeSolverState as ConstraintSolver>::transitioned(&self.0, restrictions, activate)
                .map_err(InjectedSolverFailure::Native)?
                .map(Self),
        )
    }

    fn domain(&self, variable: crate::ids::VariableId) -> Option<crate::domain::Domain> {
        self.0.domain(variable)
    }

    fn factor_is_active(&self, factor: crate::ids::FactorId) -> Option<bool> {
        self.0.factor_is_active(factor)
    }
}

static TETRAHEDRAL_RESTRICTION_CALLS: AtomicUsize = AtomicUsize::new(0);

#[derive(Clone, Debug)]
struct CountTetrahedralRestrictionsSolver(NativeSolverState);

impl ConstraintSolver for CountTetrahedralRestrictionsSolver {
    type Failure = NativeSolverFailure;

    fn initial(
        model: Arc<crate::model::ConstraintModel>,
    ) -> Result<Consistency<Self>, Self::Failure> {
        Ok(<NativeSolverState as ConstraintSolver>::initial(model)?.map(Self))
    }

    fn restricted(
        &self,
        restrictions: &[(crate::ids::VariableId, crate::domain::Domain)],
    ) -> Result<Consistency<Self>, Self::Failure> {
        TETRAHEDRAL_RESTRICTION_CALLS.fetch_add(1, Ordering::Relaxed);
        Ok(<NativeSolverState as ConstraintSolver>::restricted(&self.0, restrictions)?.map(Self))
    }

    fn transitioned(
        &self,
        restrictions: &[(crate::ids::VariableId, crate::domain::Domain)],
        activate: &[crate::ids::FactorId],
    ) -> Result<Consistency<Self>, Self::Failure> {
        TETRAHEDRAL_RESTRICTION_CALLS.fetch_add(1, Ordering::Relaxed);
        Ok(
            <NativeSolverState as ConstraintSolver>::transitioned(&self.0, restrictions, activate)?
                .map(Self),
        )
    }

    fn domain(&self, variable: crate::ids::VariableId) -> Option<crate::domain::Domain> {
        self.0.domain(variable)
    }

    fn factor_is_active(&self, factor: crate::ids::FactorId) -> Option<bool> {
        self.0.factor_is_active(factor)
    }
}

#[derive(Clone, Debug)]
struct NonconformingNonProjectingTetrahedralSolver(NativeSolverState);

impl ConstraintSolver for NonconformingNonProjectingTetrahedralSolver {
    type Failure = NativeSolverFailure;

    fn initial(
        model: Arc<crate::model::ConstraintModel>,
    ) -> Result<Consistency<Self>, Self::Failure> {
        Ok(<NativeSolverState as ConstraintSolver>::initial(model)?.map(Self))
    }

    fn restricted(
        &self,
        _restrictions: &[(crate::ids::VariableId, crate::domain::Domain)],
    ) -> Result<Consistency<Self>, Self::Failure> {
        Ok(Consistency::Consistent(self.clone()))
    }

    fn transitioned(
        &self,
        _restrictions: &[(crate::ids::VariableId, crate::domain::Domain)],
        activate: &[crate::ids::FactorId],
    ) -> Result<Consistency<Self>, Self::Failure> {
        Ok(
            <NativeSolverState as ConstraintSolver>::transitioned(&self.0, &[], activate)?
                .map(Self),
        )
    }

    fn domain(&self, variable: crate::ids::VariableId) -> Option<crate::domain::Domain> {
        self.0.domain(variable)
    }

    fn factor_is_active(&self, factor: crate::ids::FactorId) -> Option<bool> {
        self.0.factor_is_active(factor)
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

        let requested_first_ring = matches!(
            restrictions,
            [(variable, domain)]
                if *variable == crate::ids::VariableId::new(0)
                    && !domain.is_empty()
                    && domain.is_subset_of(role_partition.ring_values())
        );
        let effective = if requested_first_ring {
            let first_ring = restrictions[0];
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

fn tetrahedral_star_fixture(leaf_count: usize) -> (PreparedNonStereo, Vec<AtomId>, Vec<BondId>) {
    tetrahedral_star_fixture_with_tokens(leaf_count, vec![NonStereoBondToken::Elided; leaf_count])
}

fn tetrahedral_star_fixture_with_tokens(
    leaf_count: usize,
    bond_tokens: Vec<NonStereoBondToken>,
) -> (PreparedNonStereo, Vec<AtomId>, Vec<BondId>) {
    assert!(matches!(leaf_count, 3 | 4));
    assert_eq!(bond_tokens.len(), leaf_count);
    let mut graph = PreparedGraphBuilder::new();
    let atoms = (0..=leaf_count)
        .map(|_| graph.add_atom().unwrap())
        .collect::<Vec<_>>();
    let bonds = (0..leaf_count)
        .map(|index| graph.add_bond(atoms[0], atoms[index + 1]).unwrap())
        .collect::<Vec<_>>();
    let mut reference_order = bonds
        .iter()
        .copied()
        .map(TetrahedralLigand::Bond)
        .collect::<Vec<_>>();
    if leaf_count == 3 {
        reference_order.push(TetrahedralLigand::VirtualHydrogen);
    }
    let mut atom_tokens = vec![PreparedAtomToken::Tetrahedral {
        reference_order: reference_order.try_into().unwrap(),
        text_by_parity: if leaf_count == 3 {
            ["[C@H]".to_owned(), "[C@@H]".to_owned()]
        } else {
            ["[C@]".to_owned(), "[C@@]".to_owned()]
        },
    }];
    atom_tokens.extend(
        (0..leaf_count)
            .map(|index| PreparedAtomToken::Fixed(char::from(b'A' + index as u8).to_string())),
    );
    let surface = PreparedNonStereo::with_atom_tokens(
        PreparedMolecule::new(graph.build()),
        atom_tokens,
        bond_tokens,
    )
    .unwrap();
    (surface, atoms, bonds)
}

fn entered_tetrahedral_fixture(
    entry_token: NonStereoBondToken,
    parent_has_second_child: bool,
) -> (PreparedNonStereo, Vec<AtomId>, Vec<BondId>) {
    let mut graph = PreparedGraphBuilder::new();
    let atom_count = if parent_has_second_child { 6 } else { 5 };
    let atoms = (0..atom_count)
        .map(|_| graph.add_atom().unwrap())
        .collect::<Vec<_>>();
    let mut bonds = vec![graph.add_bond(atoms[0], atoms[1]).unwrap()];
    bonds.extend((2..5).map(|child| graph.add_bond(atoms[1], atoms[child]).unwrap()));
    if parent_has_second_child {
        bonds.push(graph.add_bond(atoms[0], atoms[5]).unwrap());
    }
    let mut atom_tokens = vec![
        PreparedAtomToken::Fixed("P".to_owned()),
        PreparedAtomToken::Tetrahedral {
            reference_order: [bonds[0], bonds[1], bonds[2], bonds[3]].map(TetrahedralLigand::Bond),
            text_by_parity: ["[C@]".to_owned(), "[C@@]".to_owned()],
        },
    ];
    atom_tokens.extend(
        (2..atom_count)
            .map(|index| PreparedAtomToken::Fixed(char::from(b'A' + index as u8).to_string())),
    );
    let mut bond_tokens = vec![entry_token];
    bond_tokens.extend(vec![NonStereoBondToken::Elided; bonds.len() - 1]);
    let surface = PreparedNonStereo::with_atom_tokens(
        PreparedMolecule::new(graph.build()),
        atom_tokens,
        bond_tokens,
    )
    .unwrap();
    (surface, atoms, bonds)
}

fn ring_coupled_tetrahedral_fixture(
    first_token: NonStereoBondToken,
) -> (PreparedNonStereo, Vec<AtomId>, Vec<BondId>) {
    let mut graph = PreparedGraphBuilder::new();
    let atoms = (0..4)
        .map(|_| graph.add_atom().unwrap())
        .collect::<Vec<_>>();
    let bonds = vec![
        graph.add_bond(atoms[0], atoms[1]).unwrap(),
        graph.add_bond(atoms[0], atoms[2]).unwrap(),
        graph.add_bond(atoms[1], atoms[2]).unwrap(),
        graph.add_bond(atoms[0], atoms[3]).unwrap(),
    ];
    let surface = PreparedNonStereo::with_atom_tokens(
        PreparedMolecule::new(graph.build()),
        vec![
            PreparedAtomToken::Tetrahedral {
                reference_order: [
                    TetrahedralLigand::Bond(bonds[0]),
                    TetrahedralLigand::Bond(bonds[1]),
                    TetrahedralLigand::Bond(bonds[3]),
                    TetrahedralLigand::VirtualHydrogen,
                ],
                text_by_parity: ["[C@H]".to_owned(), "[C@@H]".to_owned()],
            },
            PreparedAtomToken::Fixed("A".to_owned()),
            PreparedAtomToken::Fixed("B".to_owned()),
            PreparedAtomToken::Fixed("D".to_owned()),
        ],
        vec![
            first_token,
            NonStereoBondToken::Elided,
            NonStereoBondToken::Elided,
            NonStereoBondToken::Elided,
        ],
    )
    .unwrap();
    (surface, atoms, bonds)
}

fn adjacent_ring_coupled_tetrahedral_fixture() -> (PreparedNonStereo, Vec<AtomId>, Vec<BondId>) {
    let mut graph = PreparedGraphBuilder::new();
    let atoms = (0..6)
        .map(|_| graph.add_atom().unwrap())
        .collect::<Vec<_>>();
    let bonds = vec![
        graph.add_bond(atoms[0], atoms[1]).unwrap(),
        graph.add_bond(atoms[1], atoms[2]).unwrap(),
        graph.add_bond(atoms[2], atoms[3]).unwrap(),
        graph.add_bond(atoms[3], atoms[0]).unwrap(),
        graph.add_bond(atoms[0], atoms[4]).unwrap(),
        graph.add_bond(atoms[1], atoms[5]).unwrap(),
    ];
    let surface = PreparedNonStereo::with_atom_tokens(
        PreparedMolecule::new(graph.build()),
        vec![
            PreparedAtomToken::Tetrahedral {
                reference_order: [
                    TetrahedralLigand::Bond(bonds[0]),
                    TetrahedralLigand::Bond(bonds[3]),
                    TetrahedralLigand::Bond(bonds[4]),
                    TetrahedralLigand::VirtualHydrogen,
                ],
                text_by_parity: ["[L@H]".to_owned(), "[L@@H]".to_owned()],
            },
            PreparedAtomToken::Tetrahedral {
                reference_order: [
                    TetrahedralLigand::Bond(bonds[0]),
                    TetrahedralLigand::Bond(bonds[1]),
                    TetrahedralLigand::Bond(bonds[5]),
                    TetrahedralLigand::VirtualHydrogen,
                ],
                text_by_parity: ["[R@H]".to_owned(), "[R@@H]".to_owned()],
            },
            PreparedAtomToken::Fixed("A".to_owned()),
            PreparedAtomToken::Fixed("B".to_owned()),
            PreparedAtomToken::Fixed("X".to_owned()),
            PreparedAtomToken::Fixed("Y".to_owned()),
        ],
        vec![NonStereoBondToken::Elided; bonds.len()],
    )
    .unwrap();
    (surface, atoms, bonds)
}

fn disconnected_ring_coupled_tetrahedral_fixture() -> (PreparedNonStereo, Vec<AtomId>) {
    let mut graph = PreparedGraphBuilder::new();
    let atoms = (0..8)
        .map(|_| graph.add_atom().unwrap())
        .collect::<Vec<_>>();
    let mut bonds = Vec::new();
    for offset in [0, 4] {
        bonds.extend([
            graph.add_bond(atoms[offset], atoms[offset + 1]).unwrap(),
            graph.add_bond(atoms[offset], atoms[offset + 2]).unwrap(),
            graph
                .add_bond(atoms[offset + 1], atoms[offset + 2])
                .unwrap(),
            graph.add_bond(atoms[offset], atoms[offset + 3]).unwrap(),
        ]);
    }
    let mut atom_tokens = Vec::new();
    for component in 0..2 {
        let bond_offset = component * 4;
        atom_tokens.extend([
            PreparedAtomToken::Tetrahedral {
                reference_order: [
                    TetrahedralLigand::Bond(bonds[bond_offset]),
                    TetrahedralLigand::Bond(bonds[bond_offset + 1]),
                    TetrahedralLigand::Bond(bonds[bond_offset + 3]),
                    TetrahedralLigand::VirtualHydrogen,
                ],
                text_by_parity: [format!("[C{component}@H]"), format!("[C{component}@@H]")],
            },
            PreparedAtomToken::Fixed(format!("A{component}")),
            PreparedAtomToken::Fixed(format!("B{component}")),
            PreparedAtomToken::Fixed(format!("D{component}")),
        ]);
    }
    let surface = PreparedNonStereo::with_atom_tokens(
        PreparedMolecule::new(graph.build()),
        atom_tokens,
        vec![
            NonStereoBondToken::Double,
            NonStereoBondToken::Elided,
            NonStereoBondToken::Elided,
            NonStereoBondToken::Elided,
            NonStereoBondToken::Double,
            NonStereoBondToken::Elided,
            NonStereoBondToken::Elided,
            NonStereoBondToken::Elided,
        ],
    )
    .unwrap();
    (surface, atoms)
}

fn independent_local_layout_groups(
    atom_count: usize,
    endpoints: &[(usize, usize)],
    observed: &ObservedNonStereoState,
    active: usize,
) -> (Vec<BondId>, Vec<Vec<BondId>>) {
    let visited = observed
        .structural
        .traversal
        .visited_atoms
        .iter()
        .map(|atom| atom.index())
        .collect::<BTreeSet<_>>();
    let mut component = vec![None; atom_count];
    let mut next_component = 0;
    for root in 0..atom_count {
        if visited.contains(&root) || component[root].is_some() {
            continue;
        }
        let mut stack = vec![root];
        component[root] = Some(next_component);
        while let Some(atom) = stack.pop() {
            for &(a, b) in endpoints {
                let other = if a == atom {
                    Some(b)
                } else if b == atom {
                    Some(a)
                } else {
                    None
                };
                if let Some(other) = other {
                    if !visited.contains(&other) && component[other].is_none() {
                        component[other] = Some(next_component);
                        stack.push(other);
                    }
                }
            }
        }
        next_component += 1;
    }

    let mut waiting = Vec::new();
    let mut groups = BTreeMap::<usize, Vec<BondId>>::new();
    for (index, &(a, b)) in endpoints.iter().enumerate() {
        let other = if a == active {
            Some(b)
        } else if b == active {
            Some(a)
        } else {
            None
        };
        let Some(other) = other else { continue };
        let bond = BondId::new(u32::try_from(index).unwrap());
        match observed.structural.traversal.bond_progress[index] {
            ObservedBondProgress::RingOpen { first_endpoint }
                if first_endpoint.index() == other =>
            {
                waiting.push(bond);
            }
            ObservedBondProgress::Unrepresented if !visited.contains(&other) => {
                groups
                    .entry(component[other].expect("unvisited atom must own a component"))
                    .or_default()
                    .push(bond);
            }
            _ => {}
        }
    }
    waiting.sort_unstable();
    let mut groups = groups.into_values().collect::<Vec<_>>();
    for group in &mut groups {
        group.sort_unstable();
    }
    groups.sort_unstable();
    (waiting, groups)
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

fn reachable_terminal_states(state: State) -> Vec<State> {
    let mut pending = vec![state];
    let mut complete = Vec::new();
    let mut explored = 0_usize;
    while let Some(state) = pending.pop() {
        explored += 1;
        assert!(explored <= 100_000, "writer test exceeded its state bound");
        if state.is_accepted() {
            complete.push(state);
            continue;
        }
        let choices = state.choices().unwrap();
        assert!(!choices.is_empty());
        pending.extend(choices.into_iter().map(Choice::into_successor));
    }
    complete
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
fn surface_prepares_one_latent_layout_context_per_tetrahedral_entry() {
    let mut graph = PreparedGraphBuilder::new();
    let atoms: [AtomId; 5] = std::array::from_fn(|_| graph.add_atom().unwrap());
    let bonds: [BondId; 4] =
        std::array::from_fn(|index| graph.add_bond(atoms[0], atoms[index + 1]).unwrap());
    let surface = PreparedNonStereo::with_atom_tokens(
        PreparedMolecule::new(graph.build()),
        vec![
            PreparedAtomToken::Tetrahedral {
                reference_order: bonds.map(TetrahedralLigand::Bond),
                text_by_parity: ["[C@]".to_owned(), "[C@@]".to_owned()],
            },
            PreparedAtomToken::Fixed("A".to_owned()),
            PreparedAtomToken::Fixed("B".to_owned()),
            PreparedAtomToken::Fixed("C".to_owned()),
            PreparedAtomToken::Fixed("D".to_owned()),
        ],
        vec![NonStereoBondToken::Elided; bonds.len()],
    )
    .unwrap();
    let model = surface.molecule().constraint_model();
    let center = surface.tetrahedral_center(atoms[0]).unwrap();

    assert_eq!(model.variable_count(), bonds.len() + 2);
    assert_eq!(model.factor_count(), 1 + bonds.len() + 1);
    assert_eq!(
        model.potential_factors_for_variable(center.order_variable),
        Some(
            &[
                center.root_layout_factor,
                center.entry_layout_factors[0].1,
                center.entry_layout_factors[1].1,
                center.entry_layout_factors[2].1,
                center.entry_layout_factors[3].1,
            ][..]
        )
    );
    assert_eq!(
        model.initial_factors_for_variable(center.order_variable),
        Some(&[][..])
    );
    assert_eq!(
        model
            .variable(center.order_variable)
            .unwrap()
            .initial_domain(),
        full_order_domain()
    );
    assert_eq!(
        model
            .variable(center.role_pattern_variable)
            .unwrap()
            .initial_domain(),
        full_role_pattern_domain(bonds.len())
    );
    for factor in std::iter::once(center.root_layout_factor).chain(
        center
            .entry_layout_factors
            .iter()
            .map(|(_, factor)| *factor),
    ) {
        assert_eq!(
            model.factor_activation(factor),
            Some(crate::model::FactorActivation::Latent)
        );
    }
}

#[test]
fn surface_rejects_invalid_tetrahedral_bindings() {
    let mut graph = PreparedGraphBuilder::new();
    let atoms: [AtomId; 4] = std::array::from_fn(|_| graph.add_atom().unwrap());
    let bonds: [BondId; 3] =
        std::array::from_fn(|index| graph.add_bond(atoms[0], atoms[index + 1]).unwrap());
    let molecule = PreparedMolecule::new(graph.build());
    let fixed = || {
        vec![
            PreparedAtomToken::Fixed("A".to_owned()),
            PreparedAtomToken::Fixed("B".to_owned()),
            PreparedAtomToken::Fixed("C".to_owned()),
        ]
    };
    let prepare = |reference_order, text_by_parity| {
        let mut atoms = vec![PreparedAtomToken::Tetrahedral {
            reference_order,
            text_by_parity,
        }];
        atoms.extend(fixed());
        PreparedNonStereo::with_atom_tokens(
            molecule.clone(),
            atoms,
            vec![NonStereoBondToken::Elided; bonds.len()],
        )
    };

    assert!(matches!(
        prepare(
            [
                TetrahedralLigand::Bond(bonds[0]),
                TetrahedralLigand::Bond(bonds[1]),
                TetrahedralLigand::Bond(bonds[2]),
                TetrahedralLigand::VirtualHydrogen,
            ],
            [String::new(), "[C@@H]".to_owned()],
        ),
        Err(PreparedNonStereoError::EmptyTetrahedralAtomText(atom)) if atom == atoms[0]
    ));
    assert!(matches!(
        prepare(
            [
                TetrahedralLigand::Bond(bonds[0]),
                TetrahedralLigand::Bond(bonds[1]),
                TetrahedralLigand::Bond(bonds[2]),
                TetrahedralLigand::VirtualHydrogen,
            ],
            ["[C@H]".to_owned(), "[C@H]".to_owned()],
        ),
        Err(PreparedNonStereoError::RepeatedTetrahedralAtomText(atom)) if atom == atoms[0]
    ));
    assert!(matches!(
        prepare(
            [
                TetrahedralLigand::Bond(bonds[0]),
                TetrahedralLigand::Bond(bonds[1]),
                TetrahedralLigand::VirtualHydrogen,
                TetrahedralLigand::VirtualHydrogen,
            ],
            ["[C@H]".to_owned(), "[C@@H]".to_owned()],
        ),
        Err(PreparedNonStereoError::MultipleVirtualHydrogens(atom)) if atom == atoms[0]
    ));
    assert!(matches!(
        prepare(
            [
                TetrahedralLigand::Bond(bonds[0]),
                TetrahedralLigand::Bond(bonds[1]),
                TetrahedralLigand::Bond(bonds[1]),
                TetrahedralLigand::VirtualHydrogen,
            ],
            ["[C@H]".to_owned(), "[C@@H]".to_owned()],
        ),
        Err(PreparedNonStereoError::RepeatedTetrahedralLigand(atom)) if atom == atoms[0]
    ));
    assert!(matches!(
        prepare(
            [
                TetrahedralLigand::Bond(bonds[0]),
                TetrahedralLigand::Bond(bonds[1]),
                TetrahedralLigand::Bond(BondId::new(99)),
                TetrahedralLigand::VirtualHydrogen,
            ],
            ["[C@H]".to_owned(), "[C@@H]".to_owned()],
        ),
        Err(PreparedNonStereoError::TetrahedralLigandsDoNotMatchGraph(atom)) if atom == atoms[0]
    ));
}

fn resolved_tetrahedral_order(state: &State, surface: &PreparedNonStereo, atom: AtomId) -> u8 {
    let center = surface.tetrahedral_center(atom).unwrap();
    let domain = state.structural.semantic_domain(center.order_variable);
    assert!(domain.is_singleton());
    domain.iter().next().unwrap()
}

fn independent_tetrahedral_order(
    reference: &[TetrahedralLigand; 4],
    value: u8,
) -> [TetrahedralLigand; 4] {
    permutations(reference)[value as usize]
        .clone()
        .try_into()
        .unwrap()
}

fn independent_tetrahedral_order_is_even(reference: &[TetrahedralLigand; 4], value: u8) -> bool {
    let order = independent_tetrahedral_order(reference, value);
    let positions = order.map(|ligand| {
        reference
            .iter()
            .position(|candidate| *candidate == ligand)
            .unwrap()
    });
    (0..4)
        .flat_map(|left| ((left + 1)..4).map(move |right| (left, right)))
        .filter(|(left, right)| positions[*left] > positions[*right])
        .count()
        % 2
        == 0
}

#[test]
fn tetrahedral_domains_match_independent_permutation_and_parity_enumeration() {
    let reference = [
        TetrahedralLigand::Bond(BondId::new(3)),
        TetrahedralLigand::Bond(BondId::new(7)),
        TetrahedralLigand::VirtualHydrogen,
        TetrahedralLigand::Bond(BondId::new(11)),
    ];
    let orders = (0..24)
        .map(|value| independent_tetrahedral_order(&reference, value))
        .collect::<BTreeSet<_>>();
    assert_eq!(orders.len(), 24);
    assert_eq!(
        (0..24)
            .filter(|value| independent_tetrahedral_order_is_even(&reference, *value))
            .count(),
        12
    );
    for value in 0..24 {
        assert_eq!(
            parity_domain(TetrahedralParity::Even).contains(value),
            independent_tetrahedral_order_is_even(&reference, value)
        );
        let order = independent_tetrahedral_order(&reference, value);
        for prefix_len in 0..=4 {
            let domain = prefix_domain(&reference, &order[..prefix_len]);
            for candidate in 0..24 {
                assert_eq!(
                    domain.contains(candidate),
                    independent_tetrahedral_order(&reference, candidate)[..prefix_len]
                        == order[..prefix_len]
                );
            }
        }
    }
}

#[test]
fn four_arm_root_reaches_every_local_order_under_exactly_one_parity_token() {
    let (surface, atoms, _) = tetrahedral_star_fixture(4);
    let initial = initial(&surface);
    let root_choices = initial
        .choices()
        .unwrap()
        .into_iter()
        .filter(|choice| choice.successor().active_atom() == Some(atoms[0]))
        .collect::<Vec<_>>();

    assert_eq!(
        root_choices.iter().map(Choice::text).collect::<Vec<_>>(),
        vec!["[C@]", "[C@@]"]
    );
    let mut orders_by_token = BTreeMap::<String, BTreeSet<u8>>::new();
    for root in root_choices {
        let token = root.text().to_owned();
        for terminal in reachable_terminal_states(root.into_successor()) {
            orders_by_token
                .entry(token.clone())
                .or_default()
                .insert(resolved_tetrahedral_order(&terminal, &surface, atoms[0]));
        }
    }

    let reference = &surface
        .tetrahedral_center(atoms[0])
        .unwrap()
        .reference_order;
    let expected_even = (0..24)
        .filter(|value| independent_tetrahedral_order_is_even(reference, *value))
        .collect();
    let expected_odd = (0..24)
        .filter(|value| !independent_tetrahedral_order_is_even(reference, *value))
        .collect();
    assert_eq!(orders_by_token["[C@]"], expected_even);
    assert_eq!(orders_by_token["[C@@]"], expected_odd);
    assert_eq!(orders_by_token["[C@]"].len(), 12);
    assert_eq!(orders_by_token["[C@@]"].len(), 12);
}

#[test]
fn entered_four_arm_center_reaches_all_six_remaining_orders() {
    let (surface, atoms, bonds) = tetrahedral_star_fixture(4);
    let initial = initial(&surface);
    let rooted_leaf = initial
        .choices()
        .unwrap()
        .into_iter()
        .find(|choice| choice.successor().active_atom() == Some(atoms[1]))
        .unwrap()
        .into_successor();
    let atom_choices = rooted_leaf.choices().unwrap();

    assert_eq!(
        atom_choices.iter().map(Choice::text).collect::<Vec<_>>(),
        vec!["[C@]", "[C@@]"]
    );
    let mut orders = BTreeSet::new();
    for choice in atom_choices {
        for terminal in reachable_terminal_states(choice.into_successor()) {
            let value = resolved_tetrahedral_order(&terminal, &surface, atoms[0]);
            let order = independent_tetrahedral_order(
                &surface
                    .tetrahedral_center(atoms[0])
                    .unwrap()
                    .reference_order,
                value,
            );
            assert_eq!(order[0], TetrahedralLigand::Bond(bonds[0]));
            orders.insert(value);
        }
    }
    assert_eq!(orders.len(), 6);
}

#[test]
fn virtual_hydrogen_has_the_context_position_for_root_and_entry() {
    let (surface, atoms, bonds) = tetrahedral_star_fixture(3);
    let center = surface.tetrahedral_center(atoms[0]).unwrap();
    let initial = initial(&surface);

    let rooted_orders = initial
        .choices()
        .unwrap()
        .into_iter()
        .filter(|choice| choice.successor().active_atom() == Some(atoms[0]))
        .flat_map(|choice| reachable_terminal_states(choice.into_successor()))
        .map(|terminal| {
            independent_tetrahedral_order(
                &center.reference_order,
                resolved_tetrahedral_order(&terminal, &surface, atoms[0]),
            )
        })
        .collect::<BTreeSet<_>>();
    assert_eq!(rooted_orders.len(), 6);
    assert!(rooted_orders
        .iter()
        .all(|order| order[0] == TetrahedralLigand::VirtualHydrogen));

    let rooted_leaf = initial
        .choices()
        .unwrap()
        .into_iter()
        .find(|choice| choice.successor().active_atom() == Some(atoms[1]))
        .unwrap()
        .into_successor();
    let entered_orders = rooted_leaf
        .choices()
        .unwrap()
        .into_iter()
        .flat_map(|choice| reachable_terminal_states(choice.into_successor()))
        .map(|terminal| {
            independent_tetrahedral_order(
                &center.reference_order,
                resolved_tetrahedral_order(&terminal, &surface, atoms[0]),
            )
        })
        .collect::<BTreeSet<_>>();
    assert_eq!(entered_orders.len(), 2);
    assert!(entered_orders.iter().all(|order| {
        order[..2]
            == [
                TetrahedralLigand::Bond(bonds[0]),
                TetrahedralLigand::VirtualHydrogen,
            ]
    }));
}

#[test]
fn pending_atom_stages_preserve_tetrahedral_parity_branches() {
    let (elided_branch_surface, atoms, bonds) =
        entered_tetrahedral_fixture(NonStereoBondToken::Elided, true);
    let parent = initial(&elided_branch_surface)
        .choices()
        .unwrap()
        .into_iter()
        .find(|choice| choice.successor().active_atom() == Some(atoms[0]))
        .unwrap()
        .into_successor();
    let pending_branch = parent
        .choices()
        .unwrap()
        .into_iter()
        .find(|choice| {
            choice.successor().pending
                == Some(PendingEmission::BranchBondOrAtom(incident(
                    &elided_branch_surface,
                    atoms[0],
                    bonds[0],
                )))
        })
        .unwrap()
        .into_successor();
    assert_eq!(
        pending_branch
            .choices()
            .unwrap()
            .iter()
            .map(Choice::text)
            .collect::<Vec<_>>(),
        vec!["[C@]", "[C@@]"]
    );

    let (explicit_branch_surface, atoms, bonds) =
        entered_tetrahedral_fixture(NonStereoBondToken::Double, true);
    let parent = initial(&explicit_branch_surface)
        .choices()
        .unwrap()
        .into_iter()
        .find(|choice| choice.successor().active_atom() == Some(atoms[0]))
        .unwrap()
        .into_successor();
    let pending_bond = parent
        .choices()
        .unwrap()
        .into_iter()
        .find(|choice| {
            choice.successor().pending
                == Some(PendingEmission::BranchBondOrAtom(incident(
                    &explicit_branch_surface,
                    atoms[0],
                    bonds[0],
                )))
        })
        .unwrap()
        .into_successor();
    let (_, pending_atom) = only_choice(&pending_bond, "=");
    assert_eq!(
        pending_atom
            .choices()
            .unwrap()
            .iter()
            .map(Choice::text)
            .collect::<Vec<_>>(),
        vec!["[C@]", "[C@@]"]
    );

    let (inline_surface, atoms, _) = entered_tetrahedral_fixture(NonStereoBondToken::Double, false);
    let parent = initial(&inline_surface)
        .choices()
        .unwrap()
        .into_iter()
        .find(|choice| choice.successor().active_atom() == Some(atoms[0]))
        .unwrap()
        .into_successor();
    let (_, pending_atom) = only_choice(&parent, "=");
    assert_eq!(
        pending_atom
            .choices()
            .unwrap()
            .iter()
            .map(Choice::text)
            .collect::<Vec<_>>(),
        vec!["[C@]", "[C@@]"]
    );
}

#[test]
fn component_separator_defers_tetrahedral_parity_until_the_atom_token() {
    let mut graph = PreparedGraphBuilder::new();
    let atoms = (0..6)
        .map(|_| graph.add_atom().unwrap())
        .collect::<Vec<_>>();
    let bonds = (2..6)
        .map(|child| graph.add_bond(atoms[1], atoms[child]).unwrap())
        .collect::<Vec<_>>();
    let mut atom_tokens = vec![
        PreparedAtomToken::Fixed("X".to_owned()),
        PreparedAtomToken::Tetrahedral {
            reference_order: [bonds[0], bonds[1], bonds[2], bonds[3]].map(TetrahedralLigand::Bond),
            text_by_parity: ["[C@]".to_owned(), "[C@@]".to_owned()],
        },
    ];
    atom_tokens.extend(
        ["A", "B", "C", "D"]
            .into_iter()
            .map(|text| PreparedAtomToken::Fixed(text.to_owned())),
    );
    let surface = PreparedNonStereo::with_atom_tokens(
        PreparedMolecule::new(graph.build()),
        atom_tokens,
        vec![NonStereoBondToken::Elided; bonds.len()],
    )
    .unwrap();
    let after_first = initial(&surface)
        .choices()
        .unwrap()
        .into_iter()
        .find(|choice| choice.successor().structural.atom_is_visited(atoms[0]))
        .unwrap()
        .into_successor();
    let dot = after_first
        .choices()
        .unwrap()
        .into_iter()
        .find(|choice| choice.successor().active_atom() == Some(atoms[1]))
        .unwrap();

    assert_eq!(dot.text(), ".");
    assert_eq!(
        dot.successor()
            .choices()
            .unwrap()
            .iter()
            .map(Choice::text)
            .collect::<Vec<_>>(),
        vec!["[C@]", "[C@@]"]
    );
}

#[test]
fn suspended_tetrahedral_parent_retains_prefix_and_filters_child_order() {
    let (surface, atoms, bonds) = tetrahedral_star_fixture(4);
    let center = surface.tetrahedral_center(atoms[0]).unwrap();
    let mut parent = initial(&surface)
        .choices()
        .unwrap()
        .into_iter()
        .find(|choice| {
            choice.text() == "[C@]" && choice.successor().active_atom() == Some(atoms[0])
        })
        .unwrap()
        .into_successor();

    for &bond in &bonds[..2] {
        let selected = parent
            .choices()
            .unwrap()
            .into_iter()
            .find(|choice| {
                matches!(
                    choice.successor().pending,
                    Some(PendingEmission::BranchBondOrAtom(incident))
                        if incident.bond() == bond
                )
            })
            .unwrap()
            .into_successor();
        let child = selected.choices().unwrap().remove(0).into_successor();
        let restored = child.choices().unwrap().remove(0).into_successor();
        assert_eq!(restored.active_atom(), Some(atoms[0]));
        parent = restored;
    }

    let local = parent.structural.active_local_bond_order();
    assert_eq!(local.emitted_bonds, bonds[..2]);
    assert_eq!(
        parent.structural.semantic_domain(center.order_variable),
        center
            .prefix_domain_with_bond_order(None, &bonds[..2])
            .intersect(parity_domain(TetrahedralParity::Even))
    );
    let next = parent.choices().unwrap();
    assert_eq!(
        next.len(),
        1,
        "the parity-incompatible third bond must not be advertised"
    );
    assert_eq!(next[0].text(), "(");
    assert_eq!(
        next[0]
            .successor()
            .structural
            .active_local_bond_order()
            .emitted_bonds
            .len(),
        3
    );
}

#[test]
fn explicit_inline_bond_completes_tetrahedral_parent_before_child_atom() {
    let (surface, atoms, bonds) = tetrahedral_star_fixture_with_tokens(
        4,
        vec![
            NonStereoBondToken::Elided,
            NonStereoBondToken::Elided,
            NonStereoBondToken::Elided,
            NonStereoBondToken::Double,
        ],
    );
    let center = surface.tetrahedral_center(atoms[0]).unwrap();
    let mut parent = initial(&surface)
        .choices()
        .unwrap()
        .into_iter()
        .find(|choice| {
            choice.text() == "[C@]" && choice.successor().active_atom() == Some(atoms[0])
        })
        .unwrap()
        .into_successor();
    for &bond in &bonds[..3] {
        let pending = parent
            .choices()
            .unwrap()
            .into_iter()
            .find(|choice| {
                matches!(
                    choice.successor().pending,
                    Some(PendingEmission::BranchBondOrAtom(incident))
                        if incident.bond() == bond
                )
            })
            .unwrap()
            .into_successor();
        let child = pending.choices().unwrap().remove(0).into_successor();
        parent = child.choices().unwrap().remove(0).into_successor();
    }

    let (_, pending_atom) = only_choice(&parent, "=");
    assert_eq!(pending_atom.active_atom(), Some(atoms[0]));
    assert!(!pending_atom.structural.atom_is_visited(atoms[4]));
    assert_eq!(
        pending_atom
            .structural
            .active_local_bond_order()
            .emitted_bonds,
        bonds
    );
    assert_eq!(
        pending_atom
            .structural
            .semantic_domain(center.order_variable),
        center.completed_order_domain(None, &bonds)
    );
    let (_, accepted) = only_choice(&pending_atom, "D");
    assert!(accepted.is_accepted());
}

#[test]
fn unresolved_tetrahedral_completion_is_typed_and_leaves_source_unchanged() {
    let (surface, atoms, bonds) = tetrahedral_star_fixture(4);
    let initial =
        NonStereoWriterState::<NonconformingNonProjectingTetrahedralSolver>::initial(&surface)
            .unwrap()
            .unwrap_consistent();
    let mut parent = initial
        .choices()
        .unwrap()
        .into_iter()
        .find(|choice| {
            choice.text() == "[C@]" && choice.successor().active_atom() == Some(atoms[0])
        })
        .unwrap()
        .into_successor();
    for &bond in &bonds[..3] {
        let pending = parent
            .choices()
            .unwrap()
            .into_iter()
            .find(|choice| {
                matches!(
                    choice.successor().pending,
                    Some(PendingEmission::BranchBondOrAtom(incident))
                        if incident.bond() == bond
                )
            })
            .unwrap()
            .into_successor();
        let child = pending
            .choices()
            .unwrap()
            .into_iter()
            .next()
            .unwrap()
            .into_successor();
        parent = child
            .choices()
            .unwrap()
            .into_iter()
            .next()
            .unwrap()
            .into_successor();
    }
    let source = parent.observe_raw();

    assert!(matches!(
        parent.choices(),
        Err(ChoiceFailure::Invariant(
            WriterInvariantFailure::UnresolvedTetrahedralFrame { atom }
        )) if atom == atoms[0]
    ));
    assert_eq!(parent.observe_raw(), source);
}

#[test]
fn adjacent_tetrahedral_centers_resolve_independent_local_orders() {
    let mut graph = PreparedGraphBuilder::new();
    let atoms = (0..8)
        .map(|_| graph.add_atom().unwrap())
        .collect::<Vec<_>>();
    let shared = graph.add_bond(atoms[0], atoms[1]).unwrap();
    let left = (2..5)
        .map(|child| graph.add_bond(atoms[0], atoms[child]).unwrap())
        .collect::<Vec<_>>();
    let right = (5..8)
        .map(|child| graph.add_bond(atoms[1], atoms[child]).unwrap())
        .collect::<Vec<_>>();
    let mut atom_tokens = vec![
        PreparedAtomToken::Tetrahedral {
            reference_order: [shared, left[0], left[1], left[2]].map(TetrahedralLigand::Bond),
            text_by_parity: ["[L@]".to_owned(), "[L@@]".to_owned()],
        },
        PreparedAtomToken::Tetrahedral {
            reference_order: [shared, right[0], right[1], right[2]].map(TetrahedralLigand::Bond),
            text_by_parity: ["[R@]".to_owned(), "[R@@]".to_owned()],
        },
    ];
    atom_tokens.extend((2..8).map(|index| PreparedAtomToken::Fixed(format!("A{index}"))));
    let surface = PreparedNonStereo::with_atom_tokens(
        PreparedMolecule::new(graph.build()),
        atom_tokens,
        vec![NonStereoBondToken::Elided; 7],
    )
    .unwrap();
    let left_center = surface.tetrahedral_center(atoms[0]).unwrap();
    let right_center = surface.tetrahedral_center(atoms[1]).unwrap();
    let root = initial(&surface)
        .choices()
        .unwrap()
        .into_iter()
        .find(|choice| {
            choice.text() == "[L@]" && choice.successor().active_atom() == Some(atoms[0])
        })
        .unwrap()
        .into_successor();
    let terminals = reachable_terminal_states(root);

    assert!(!terminals.is_empty());
    assert!(terminals.iter().all(|terminal| {
        terminal
            .structural
            .semantic_domain(left_center.order_variable)
            .is_singleton()
            && terminal
                .structural
                .semantic_domain(right_center.order_variable)
                .is_singleton()
    }));
    let right_orders = terminals
        .iter()
        .map(|terminal| resolved_tetrahedral_order(terminal, &surface, atoms[1]))
        .collect::<BTreeSet<_>>();
    assert!(right_orders
        .iter()
        .any(|value| independent_tetrahedral_order_is_even(&right_center.reference_order, *value)));
    assert!(right_orders.iter().any(|value| {
        !independent_tetrahedral_order_is_even(&right_center.reference_order, *value)
    }));
    for terminal in terminals {
        let order = independent_tetrahedral_order(
            &right_center.reference_order,
            resolved_tetrahedral_order(&terminal, &surface, atoms[1]),
        );
        assert_eq!(order[0], TetrahedralLigand::Bond(shared));
    }
}

#[test]
fn ring_capable_tetrahedral_center_activates_layout_at_its_atom_event() {
    let (surface, atoms, _) = ring_coupled_tetrahedral_fixture(NonStereoBondToken::Elided);
    let state = State::initial(&surface).unwrap().unwrap_consistent();
    let source = state.observe_raw();

    let choices = state
        .choices()
        .unwrap()
        .into_iter()
        .filter(|choice| choice.successor().active_atom() == Some(atoms[0]))
        .collect::<Vec<_>>();
    assert_eq!(choices.len(), 2);
    let center = surface.tetrahedral_center(atoms[0]).unwrap();
    for choice in choices {
        let active_factors = &choice.successor().observe_raw().structural.active_factors;
        assert!(choice
            .successor()
            .structural
            .factor_is_active(center.root_layout_factor));
        assert!(active_factors.contains(&center.root_layout_factor));
        assert_eq!(
            active_factors.len(),
            source.structural.active_factors.len() + 1,
            "the chosen layout is the only newly active factor"
        );
        assert!(center
            .entry_layout_factors
            .iter()
            .all(|(_, factor)| !choice.successor().structural.factor_is_active(*factor)));
        assert!(!choice
            .successor()
            .structural
            .semantic_domain(center.role_pattern_variable)
            .is_empty());
    }
    assert_eq!(state.observe_raw(), source);
}

#[test]
fn prospective_frame_context_derives_exact_local_role_patterns() {
    let (surface, atoms, bonds) = tetrahedral_star_fixture(3);
    let state = initial(&surface);
    let center = surface.tetrahedral_center(atoms[0]).unwrap();
    let root_context = LocalLayoutContext {
        order: crate::traversal::LocalBondOrder {
            atom: atoms[0],
            entry_bond: None,
            emitted_bonds: Vec::new(),
            ring_occurrence_count: 0,
        },
        waiting_ring_bonds: Vec::new(),
        residual_attachment_bonds: vec![vec![bonds[0], bonds[1]], vec![bonds[2]]],
    };
    let entered_context = LocalLayoutContext {
        order: crate::traversal::LocalBondOrder {
            atom: atoms[0],
            entry_bond: Some(bonds[0]),
            emitted_bonds: Vec::new(),
            ring_occurrence_count: 0,
        },
        waiting_ring_bonds: vec![bonds[1]],
        residual_attachment_bonds: vec![vec![bonds[2]]],
    };

    assert_eq!(
        state.local_role_pattern_domain(center, &root_context),
        Domain::from_indices([1, 2]).unwrap()
    );
    assert_eq!(
        state.local_role_pattern_domain(center, &entered_context),
        Domain::singleton(2).unwrap()
    );
}

#[test]
fn live_tetrahedral_context_matches_declarative_graph_recomputation() {
    let (surface, atoms, bonds) = ring_coupled_tetrahedral_fixture(NonStereoBondToken::Elided);
    let endpoints = [(0, 1), (0, 2), (1, 2), (0, 3)];
    let initial = initial(&surface);
    let prospective = initial.structural.prospective_root_layout_context(atoms[0]);
    assert_eq!(prospective.order.entry_bond, None);
    assert!(prospective.waiting_ring_bonds.is_empty());
    assert_eq!(
        prospective.residual_attachment_bonds,
        vec![vec![bonds[0], bonds[1]], vec![bonds[3]]]
    );

    let mut pending = vec![initial];
    let mut saw_root = false;
    let mut saw_entry = false;
    let mut saw_waiting_closure = false;
    let mut explored = 0;
    while let Some(state) = pending.pop() {
        explored += 1;
        assert!(explored < 20_000);
        if state.pending.is_none() && state.active_atom() == Some(atoms[0]) {
            let observed = state.observe_raw();
            let context = state.structural.active_local_layout_context();
            let (waiting, mut groups) =
                independent_local_layout_groups(4, &endpoints, &observed, 0);
            let mut actual_groups = context.residual_attachment_bonds.clone();
            for group in &mut actual_groups {
                group.sort_unstable();
            }
            actual_groups.sort_unstable();
            groups.sort_unstable();
            assert_eq!(context.waiting_ring_bonds, waiting);
            assert_eq!(actual_groups, groups);
            saw_root |= context.order.entry_bond.is_none();
            saw_entry |= context.order.entry_bond.is_some();
            saw_waiting_closure |= !waiting.is_empty();
        }
        if saw_root && saw_entry && saw_waiting_closure {
            break;
        }
        pending.extend(
            state
                .choices()
                .unwrap()
                .into_iter()
                .map(Choice::into_successor),
        );
    }
    assert!(saw_root && saw_entry && saw_waiting_closure);
}

#[test]
fn explicit_ring_endpoint_commits_local_order_before_pending_label() {
    let (surface, atoms, bonds) = ring_coupled_tetrahedral_fixture(NonStereoBondToken::Double);
    let source = initial(&surface);
    let pending = source
        .choices()
        .unwrap()
        .into_iter()
        .filter(|choice| choice.successor().active_atom() == Some(atoms[0]))
        .flat_map(|choice| choice.into_successor().choices().unwrap())
        .find(|choice| {
            choice.text() == "="
                && matches!(
                    choice.successor().pending,
                    Some(PendingEmission::RingOpeningLabel { incident, .. })
                        if incident.bond() == bonds[0]
                )
        })
        .unwrap()
        .into_successor();
    let before_label = pending.structural.active_local_bond_order();

    assert_eq!(before_label.emitted_bonds, vec![bonds[0]]);
    assert_eq!(before_label.ring_occurrence_count, 1);
    let after_label = pending.choices().unwrap().remove(0).into_successor();
    assert_eq!(
        after_label.structural.active_local_bond_order(),
        before_label
    );
}

#[test]
fn every_ring_endpoint_path_commits_one_local_occurrence() {
    let (surface, atoms, bonds) = ring_coupled_tetrahedral_fixture(NonStereoBondToken::Double);
    let mut pending = vec![initial(&surface)];
    let mut saw = [false; 4];
    let mut explored = 0;

    while let Some(state) = pending.pop() {
        explored += 1;
        assert!(explored < 20_000);
        let source = state.observe_raw();
        let Some(source_frame) = source.structural.traversal.active_frame.as_ref() else {
            pending.extend(
                state
                    .choices()
                    .unwrap()
                    .into_iter()
                    .map(Choice::into_successor),
            );
            continue;
        };
        for choice in state.choices().unwrap() {
            let successor = choice.successor().observe_raw();
            let source_progress = &source.structural.traversal.bond_progress[bonds[0].index()];
            let successor_progress =
                &successor.structural.traversal.bond_progress[bonds[0].index()];
            let event = match (source_progress, successor_progress) {
                (
                    ObservedBondProgress::Unrepresented,
                    ObservedBondProgress::RingOpen { first_endpoint },
                ) if *first_endpoint == atoms[0] && source_frame.atom == atoms[0] => {
                    Some((0, choice.text() == "="))
                }
                (
                    ObservedBondProgress::RingOpen { first_endpoint },
                    ObservedBondProgress::RingClosed {
                        first_endpoint: closed_first,
                        second_endpoint,
                    },
                ) if *first_endpoint == *closed_first
                    && *second_endpoint == atoms[0]
                    && source_frame.atom == atoms[0] =>
                {
                    Some((2, choice.text() == "="))
                }
                _ => None,
            };
            if let Some((base, explicit)) = event {
                let successor_frame = successor
                    .structural
                    .traversal
                    .active_frame
                    .as_ref()
                    .expect("ring endpoint must retain the active tetrahedral frame");
                assert_eq!(
                    successor_frame.emitted_bonds.len(),
                    source_frame.emitted_bonds.len() + 1
                );
                assert_eq!(successor_frame.emitted_bonds.last(), Some(&bonds[0]));
                assert_eq!(
                    successor_frame.ring_occurrence_count,
                    source_frame.ring_occurrence_count + 1
                );
                saw[base + usize::from(explicit)] = true;

                if explicit {
                    let pending_state = choice.successor().clone();
                    let after_label = pending_state
                        .choices()
                        .unwrap()
                        .into_iter()
                        .next()
                        .unwrap()
                        .into_successor()
                        .observe_raw();
                    assert_eq!(
                        after_label.structural.traversal.active_frame,
                        successor.structural.traversal.active_frame,
                        "a pending label must not recommit the ring occurrence"
                    );
                }
            }
            pending.push(choice.into_successor());
        }
        if saw.iter().all(|value| *value) {
            break;
        }
    }

    assert_eq!(saw, [true; 4], "opening/closure emit/omit paths must occur");
}

#[test]
fn ring_coupled_tetrahedral_center_has_complete_online_walks() {
    let (surface, atoms, _) = ring_coupled_tetrahedral_fixture(NonStereoBondToken::Elided);

    let center = surface.tetrahedral_center(atoms[0]).unwrap();
    let mut pending = vec![(initial(&surface), String::new())];
    let mut paths = Vec::new();
    let mut saw_entered_context = false;
    let mut saw_waiting_closure = false;
    let mut explored = 0;
    while let Some((state, prefix)) = pending.pop() {
        explored += 1;
        assert!(explored < 20_000);
        if state.is_accepted() {
            paths.push(prefix);
            continue;
        }
        if state.pending.is_none() && state.active_atom() == Some(atoms[0]) {
            let context = state.structural.active_local_layout_context();
            if let Some(entry) = context.order.entry_bond {
                saw_entered_context = true;
                assert!(state
                    .structural
                    .factor_is_active(center.layout_factor(Some(entry))));
            }
            saw_waiting_closure |= !context.waiting_ring_bonds.is_empty();
        }
        for choice in state.choices().unwrap() {
            pending.push((
                choice.successor().clone(),
                format!("{prefix}{}", choice.text()),
            ));
        }
    }

    assert!(!paths.is_empty());
    assert!(paths.iter().any(|text| text.contains('1')));
    assert!(saw_entered_context);
    assert!(saw_waiting_closure);
}

#[test]
fn adjacent_ring_coupled_centers_activate_and_resolve_in_one_component() {
    let (surface, atoms, _) = adjacent_ring_coupled_tetrahedral_fixture();
    let centers = [
        surface.tetrahedral_center(atoms[0]).unwrap(),
        surface.tetrahedral_center(atoms[1]).unwrap(),
    ];
    let factor_ids = centers.map(|center| {
        std::iter::once(center.root_layout_factor)
            .chain(
                center
                    .entry_layout_factors
                    .iter()
                    .map(|(_, factor)| *factor),
            )
            .collect::<Vec<_>>()
    });
    let mut pending = vec![initial(&surface)];
    let mut saw_both_active = false;
    let mut saw_one_suspended_while_the_other_was_active = false;
    let mut accepted = 0;
    let mut explored = 0;

    while let Some(state) = pending.pop() {
        explored += 1;
        assert!(explored < 100_000);
        let observed = state.observe_raw();
        let center_is_active = factor_ids.each_ref().map(|ids| {
            ids.iter()
                .any(|factor| observed.structural.active_factors.contains(factor))
        });
        saw_both_active |= center_is_active == [true, true];
        if let Some(active) = observed
            .structural
            .traversal
            .active_frame
            .as_ref()
            .map(|frame| frame.atom)
        {
            let suspended = observed
                .structural
                .traversal
                .branch_returns
                .iter()
                .map(|frame| frame.atom)
                .collect::<BTreeSet<_>>();
            saw_one_suspended_while_the_other_was_active |= (active == atoms[0]
                && suspended.contains(&atoms[1]))
                || (active == atoms[1] && suspended.contains(&atoms[0]));
        }
        if state.is_accepted() {
            accepted += 1;
            for center in centers {
                assert!(state
                    .structural
                    .semantic_domain(center.order_variable)
                    .is_singleton());
                assert!(state
                    .structural
                    .semantic_domain(center.role_pattern_variable)
                    .is_singleton());
            }
            if saw_both_active && saw_one_suspended_while_the_other_was_active {
                break;
            }
            continue;
        }
        pending.extend(
            state
                .choices()
                .unwrap()
                .into_iter()
                .map(Choice::into_successor),
        );
    }

    assert!(accepted > 0);
    assert!(saw_both_active);
    assert!(saw_one_suspended_while_the_other_was_active);
}

#[test]
fn disconnected_ring_coupled_centers_keep_factors_local_and_reuse_labels() {
    let (surface, atoms) = disconnected_ring_coupled_tetrahedral_fixture();
    let centers = [
        surface.tetrahedral_center(atoms[0]).unwrap(),
        surface.tetrahedral_center(atoms[4]).unwrap(),
    ];
    let mut pending = vec![(initial(&surface), String::new())];
    let mut accepted = None;
    let mut explored = 0;
    while let Some((state, text)) = pending.pop() {
        explored += 1;
        assert!(explored < 100_000);
        if state.is_accepted() {
            accepted = Some((state, text));
            break;
        }
        pending.extend(state.choices().unwrap().into_iter().map(|choice| {
            let next_text = format!("{text}{}", choice.text());
            (choice.into_successor(), next_text)
        }));
    }
    let (accepted, text) = accepted.expect("disconnected cyclic fixture must have support");
    let components = text.split('.').collect::<Vec<_>>();
    assert_eq!(components.len(), 2);
    assert!(components.iter().all(|component| component.contains('1')));
    assert!(accepted.labels.is_clean());
    for center in centers {
        assert!(accepted
            .structural
            .semantic_domain(center.order_variable)
            .is_singleton());
        assert!(accepted
            .structural
            .semantic_domain(center.role_pattern_variable)
            .is_singleton());
        assert!(std::iter::once(center.root_layout_factor)
            .chain(
                center
                    .entry_layout_factors
                    .iter()
                    .map(|(_, factor)| *factor)
            )
            .any(|factor| accepted.structural.factor_is_active(factor)));
    }
}

#[test]
fn contradictory_tetrahedral_token_is_filtered_without_suppressing_its_sibling() {
    let (surface, atoms, _) = tetrahedral_star_fixture(4);
    let state =
        NonStereoWriterState::<NonconformingRejectEvenTetrahedralParitySolver>::initial(&surface)
            .unwrap()
            .unwrap_consistent();
    let source = state.observe_raw();
    let center_choices = state
        .choices()
        .unwrap()
        .into_iter()
        .filter(|choice| choice.successor().active_atom() == Some(atoms[0]))
        .collect::<Vec<_>>();

    assert_eq!(center_choices.len(), 1);
    assert_eq!(center_choices[0].text(), "[C@@]");
    assert_eq!(state.observe_raw(), source);
}

#[test]
fn backend_failure_aborts_the_tetrahedral_atom_token_batch() {
    let (surface, _, _) = tetrahedral_star_fixture(4);
    TETRAHEDRAL_BACKEND_ATTEMPTS.store(0, Ordering::Relaxed);
    let state = NonStereoWriterState::<FailTetrahedralRestrictionSolver>::initial(&surface)
        .unwrap()
        .unwrap_consistent();

    assert!(matches!(
        state.choices(),
        Err(ChoiceFailure::Backend(InjectedSolverFailure::Restriction))
    ));
}

#[test]
fn preceding_explicit_bond_is_hidden_when_pending_atom_tokens_all_contradict() {
    let (surface, atoms, _) = entered_tetrahedral_fixture(NonStereoBondToken::Double, false);
    let initial =
        NonStereoWriterState::<NonconformingRejectTetrahedralParitySolver>::initial(&surface)
            .unwrap()
            .unwrap_consistent();
    let parent = initial
        .choices()
        .unwrap()
        .into_iter()
        .find(|choice| choice.successor().active_atom() == Some(atoms[0]))
        .unwrap()
        .into_successor();

    assert!(matches!(
        parent.choices(),
        Err(ChoiceFailure::Invariant(
            WriterInvariantFailure::AllCandidatesSemanticallyRejected { candidate_count: 1 }
        ))
    ));
}

#[test]
fn preceding_parenthesis_is_hidden_when_pending_atom_tokens_all_contradict() {
    let (surface, atoms, bonds) = entered_tetrahedral_fixture(NonStereoBondToken::Elided, true);
    let initial =
        NonStereoWriterState::<NonconformingRejectTetrahedralParitySolver>::initial(&surface)
            .unwrap()
            .unwrap_consistent();
    let parent = initial
        .choices()
        .unwrap()
        .into_iter()
        .find(|choice| choice.successor().active_atom() == Some(atoms[0]))
        .unwrap()
        .into_successor();
    let choices = parent.choices().unwrap();

    assert_eq!(choices.len(), 1);
    assert_eq!(choices[0].text(), "(");
    assert!(matches!(
        choices[0].successor().pending,
        Some(PendingEmission::BranchBondOrAtom(incident))
            if incident.bond() == bonds[4]
    ));
}

#[test]
fn each_tetrahedral_semantic_choice_uses_one_solver_restriction_batch() {
    let (surface, atoms, _) = tetrahedral_star_fixture(4);
    TETRAHEDRAL_RESTRICTION_CALLS.store(0, Ordering::Relaxed);
    let initial = NonStereoWriterState::<CountTetrahedralRestrictionsSolver>::initial(&surface)
        .unwrap()
        .unwrap_consistent();
    let root_choices = initial.choices().unwrap();
    assert_eq!(TETRAHEDRAL_RESTRICTION_CALLS.load(Ordering::Relaxed), 2);
    let rooted = root_choices
        .into_iter()
        .find(|choice| {
            choice.text() == "[C@]" && choice.successor().active_atom() == Some(atoms[0])
        })
        .unwrap()
        .into_successor();

    TETRAHEDRAL_RESTRICTION_CALLS.store(0, Ordering::Relaxed);
    let branch_choices = rooted.choices().unwrap();
    assert_eq!(branch_choices.len(), 4);
    assert_eq!(TETRAHEDRAL_RESTRICTION_CALLS.load(Ordering::Relaxed), 4);
}

#[test]
fn ring_and_child_atom_frontiers_use_one_transition_per_semantic_choice() {
    type CountingState = NonStereoWriterState<CountTetrahedralRestrictionsSolver>;

    let (surface, atoms, bonds) = ring_coupled_tetrahedral_fixture(NonStereoBondToken::Double);
    let initial = CountingState::initial(&surface)
        .unwrap()
        .unwrap_consistent();
    let rooted = initial
        .choices()
        .unwrap()
        .into_iter()
        .find(|choice| choice.successor().active_atom() == Some(atoms[0]))
        .unwrap()
        .into_successor();
    let mut pending = vec![rooted];
    let mut saw_opening = false;
    let mut saw_closure = false;
    let mut explored = 0;
    while let Some(state) = pending.pop() {
        explored += 1;
        assert!(explored < 20_000);
        let source = state.observe_raw();
        TETRAHEDRAL_RESTRICTION_CALLS.store(0, Ordering::Relaxed);
        let choices = state.choices().unwrap();
        let ring_event = choices.iter().any(|choice| {
            let successor = choice.successor().observe_raw();
            match (
                &source.structural.traversal.bond_progress[bonds[0].index()],
                &successor.structural.traversal.bond_progress[bonds[0].index()],
            ) {
                (ObservedBondProgress::Unrepresented, ObservedBondProgress::RingOpen { .. }) => {
                    saw_opening = true;
                    true
                }
                (
                    ObservedBondProgress::RingOpen { .. },
                    ObservedBondProgress::RingClosed { .. },
                ) => {
                    saw_closure = true;
                    true
                }
                _ => false,
            }
        });
        if ring_event {
            assert_eq!(
                TETRAHEDRAL_RESTRICTION_CALLS.load(Ordering::Relaxed),
                choices.len(),
                "each ring spelling candidate must use one atomic solver transition"
            );
        }
        pending.extend(choices.into_iter().map(Choice::into_successor));
        if saw_opening && saw_closure {
            break;
        }
    }
    assert!(saw_opening && saw_closure);

    for token in [NonStereoBondToken::Elided, NonStereoBondToken::Double] {
        let (surface, atoms, _) = entered_tetrahedral_fixture(token, false);
        let initial = CountingState::initial(&surface)
            .unwrap()
            .unwrap_consistent();
        let parent = initial
            .choices()
            .unwrap()
            .into_iter()
            .find(|choice| choice.successor().active_atom() == Some(atoms[0]))
            .unwrap()
            .into_successor();
        let atom_frontier = if token == NonStereoBondToken::Double {
            parent
                .choices()
                .unwrap()
                .into_iter()
                .find(|choice| choice.text() == "=")
                .unwrap()
                .into_successor()
        } else {
            parent
        };
        TETRAHEDRAL_RESTRICTION_CALLS.store(0, Ordering::Relaxed);
        let atom_choices = atom_frontier.choices().unwrap();
        assert_eq!(atom_choices.len(), 2);
        assert_eq!(
            TETRAHEDRAL_RESTRICTION_CALLS.load(Ordering::Relaxed),
            atom_choices.len(),
            "each pending or combined child atom token must use one transition"
        );
    }
}

#[test]
fn isolated_tetrahedral_order_domains_do_not_create_exact_search_work() {
    let (surface, _, _) = tetrahedral_star_fixture(4);
    let state = initial(&surface);

    assert_eq!(
        state
            .structural
            .constraints_for_test()
            .tetrahedral_factor_revision_count(),
        0,
        "latent tetrahedral factors do no initial work"
    );
    assert_eq!(
        state.structural.constraints_for_test().exact_run_counts(),
        (0, 0)
    );
    state
        .structural
        .constraints_for_test()
        .reset_tetrahedral_factor_revision_count();
    let _choices = state.choices().unwrap();
    assert!(
        state
            .structural
            .constraints_for_test()
            .tetrahedral_factor_revision_count()
            > 0,
        "the tetrahedral root candidates revise their newly active factors"
    );
    assert_eq!(
        state.structural.constraints_for_test().exact_run_counts(),
        (0, 0)
    );
}

#[test]
fn pending_prevalidation_work_is_measured_at_the_lexical_frontier() {
    let (surface, atoms, _) = entered_tetrahedral_fixture(NonStereoBondToken::Double, false);
    let parent = initial(&surface)
        .choices()
        .unwrap()
        .into_iter()
        .find(|choice| choice.successor().active_atom() == Some(atoms[0]))
        .unwrap()
        .into_successor();

    parent.reset_writer_work_counts();
    let pending_atom = parent
        .choices()
        .unwrap()
        .into_iter()
        .find(|choice| choice.text() == "=")
        .unwrap()
        .into_successor();
    assert_eq!(
        parent.writer_work_counts(),
        (1, 2),
        "the explicit bond validates one two-choice atom frontier"
    );

    pending_atom.reset_writer_work_counts();
    assert_eq!(pending_atom.choices().unwrap().len(), 2);
    assert_eq!(
        pending_atom.writer_work_counts(),
        (1, 0),
        "publishing the pending atom frontier does not discard its successors"
    );
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
fn ring_alternative_generation_stops_at_the_first_backend_failure() {
    let (surface, atoms, bonds) = fixture(
        &["C", "C", "C"],
        &[
            (0, 1, NonStereoBondToken::Double),
            (0, 2, NonStereoBondToken::Elided),
            (1, 2, NonStereoBondToken::Elided),
        ],
    );
    let initial = NonStereoWriterState::<FailFirstRingAlternativeSolver>::initial(&surface)
        .unwrap()
        .unwrap_consistent();
    let rooted = initial
        .choices()
        .unwrap()
        .into_iter()
        .find(|choice| choice.successor().active_atom() == Some(atoms[0]))
        .unwrap()
        .into_successor();
    let candidate = rooted
        .structural
        .derive_candidates()
        .candidates()
        .iter()
        .copied()
        .find(|candidate| {
            matches!(
                candidate,
                StructuralCandidate::RingOpen { incident } if incident.bond() == bonds[0]
            )
        })
        .unwrap();
    let StructuralCandidate::RingOpen { incident } = candidate else {
        unreachable!()
    };

    assert!(matches!(
        rooted.attempt_ring_openings(candidate, incident),
        Err(InjectedSolverFailure::Restriction)
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
fn explicit_pending_labels_retain_spelling_exhaustion_classification() {
    let (surface, atoms, _) = fixture(
        &["C", "C", "C"],
        &[
            (0, 1, NonStereoBondToken::Double),
            (0, 2, NonStereoBondToken::Double),
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
                blocked_candidate_count: 4
            }
        ))
    ));
}

#[test]
fn explicit_closure_pending_label_retains_spelling_exhaustion_classification() {
    let (surface, atoms, bonds) = fixture(
        &["C", "C", "C"],
        &[
            (0, 1, NonStereoBondToken::Double),
            (0, 2, NonStereoBondToken::Elided),
            (1, 2, NonStereoBondToken::Elided),
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
    let opening = rooted
        .choices()
        .unwrap()
        .into_iter()
        .find(|choice| {
            choice.text() == "="
                && choice
                    .successor()
                    .labels
                    .bonds_by_slot
                    .values()
                    .any(|bond| *bond == bonds[0])
        })
        .unwrap()
        .into_successor();
    let (_, opened) = only_choice(&opening, "1");
    let (_, walked) = only_choice(&opened, "C");
    let (_, mut walked) = only_choice(&walked, "C");
    walked.labels.maximum_spelling_label = Some(0);

    assert!(matches!(
        walked.choices(),
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
fn explicit_endpoint_contradiction_precedes_pending_label_spelling() {
    let (surface, atoms, bonds) = fixture(
        &["R", "A", "B", "C"],
        &[
            (0, 1, NonStereoBondToken::Double),
            (0, 2, NonStereoBondToken::Elided),
            (0, 3, NonStereoBondToken::Elided),
            (1, 2, NonStereoBondToken::Elided),
            (2, 3, NonStereoBondToken::Elided),
        ],
    );
    let initial = NonStereoWriterState::<WriterPolicyContradictionSolver>::initial(&surface)
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
    let candidate = rooted
        .structural
        .derive_candidates()
        .candidates()
        .iter()
        .copied()
        .find(|candidate| {
            matches!(
                candidate,
                StructuralCandidate::RingOpen { incident } if incident.bond() == bonds[0]
            )
        })
        .unwrap();
    let StructuralCandidate::RingOpen { incident } = candidate else {
        unreachable!()
    };

    let attempts = rooted.attempt_ring_openings(candidate, incident).unwrap();

    assert_eq!(attempts.len(), 2);
    assert!(attempts.into_iter().all(|attempt| matches!(
        attempt,
        CandidateAttempt::Rejected {
            reason: CandidateRejection::Contradiction
        }
    )));
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
    assert!(pending_label.graph_is_complete());

    let (label, accepted) = only_choice(&pending_label, "1");
    assert_eq!(
        [root, open, first_child, second_child, bond, label].concat(),
        "C1CC=1"
    );
    assert!(accepted.is_accepted());
}

#[test]
fn explicit_ring_opening_choices_refine_the_fixed_endpoint_plan() {
    let (surface, atoms, bonds) = fixture(
        &["A", "B", "C"],
        &[
            (0, 1, NonStereoBondToken::Double),
            (0, 2, NonStereoBondToken::Elided),
            (1, 2, NonStereoBondToken::Elided),
        ],
    );
    let initial = initial(&surface);
    let initial_plan = BondRepresentation::explicit_domain();

    for (root, omitted_plan, emitted_plan) in [
        (
            atoms[0],
            BondRepresentation::Ring01.singleton_domain(),
            BondRepresentation::Ring10
                .singleton_domain()
                .union(BondRepresentation::Ring11.singleton_domain()),
        ),
        (
            atoms[1],
            BondRepresentation::Ring10.singleton_domain(),
            BondRepresentation::Ring01
                .singleton_domain()
                .union(BondRepresentation::Ring11.singleton_domain()),
        ),
    ] {
        let rooted = initial
            .choices()
            .unwrap()
            .into_iter()
            .find(|choice| choice.successor().active_atom() == Some(root))
            .unwrap()
            .into_successor();
        let choices = rooted
            .choices()
            .unwrap()
            .into_iter()
            .filter(|choice| {
                choice
                    .successor()
                    .labels
                    .bonds_by_slot
                    .values()
                    .any(|bond| *bond == bonds[0])
            })
            .collect::<Vec<_>>();

        assert_eq!(choices.len(), 2);
        let omitted = choices.iter().find(|choice| choice.text() == "1").unwrap();
        let emitted = choices.iter().find(|choice| choice.text() == "=").unwrap();
        assert_eq!(
            omitted
                .successor()
                .structural
                .bond_decision_domain(bonds[0]),
            omitted_plan
        );
        assert_eq!(
            emitted
                .successor()
                .structural
                .bond_decision_domain(bonds[0]),
            emitted_plan
        );
        assert!(matches!(
            emitted.successor().pending,
            Some(PendingEmission::RingOpeningLabel { incident, .. })
                if incident.bond() == bonds[0]
        ));
        assert_eq!(emitted.successor().choices().unwrap()[0].text(), "1");
        assert_eq!(
            rooted.structural.bond_decision_domain(bonds[0]),
            initial_plan
        );
    }
}

#[test]
fn explicit_ring_closure_resolves_opening_only_closure_only_and_both_plans() {
    let (surface, atoms, bonds) = fixture(
        &["C", "C", "C"],
        &[
            (0, 1, NonStereoBondToken::Double),
            (0, 2, NonStereoBondToken::Elided),
            (1, 2, NonStereoBondToken::Elided),
        ],
    );
    let initial = initial(&surface);
    let (root_text, rooted) = choice_at(&initial, atoms[0].index());
    let opening_choices = rooted
        .choices()
        .unwrap()
        .into_iter()
        .filter(|choice| {
            choice
                .successor()
                .labels
                .bonds_by_slot
                .values()
                .any(|bond| *bond == bonds[0])
        })
        .collect::<Vec<_>>();
    let omitted_opening = opening_choices
        .iter()
        .find(|choice| choice.text() == "1")
        .unwrap()
        .successor()
        .clone();
    let emitted_opening = opening_choices
        .into_iter()
        .find(|choice| choice.text() == "=")
        .unwrap()
        .into_successor();

    let (_, walked) = only_choice(&omitted_opening, "C");
    let (_, walked) = only_choice(&walked, "C");
    let (closure_text, pending_label) = only_choice(&walked, "=");
    assert_eq!(
        pending_label.structural.bond_decision_domain(bonds[0]),
        BondRepresentation::Ring01.singleton_domain()
    );
    let (closure_label, closure_only) = only_choice(&pending_label, "1");
    assert_eq!(
        [root_text.as_str(), "1CC", &closure_text, &closure_label].concat(),
        "C1CC=1"
    );
    assert!(closure_only.is_accepted());

    let (opening_label, emitted_opening) = only_choice(&emitted_opening, "1");
    let (_, walked) = only_choice(&emitted_opening, "C");
    let (_, walked) = only_choice(&walked, "C");
    assert_eq!(
        walked.structural.bond_decision_domain(bonds[0]),
        BondRepresentation::Ring10
            .singleton_domain()
            .union(BondRepresentation::Ring11.singleton_domain())
    );
    let closure_choices = walked.choices().unwrap();
    assert_eq!(closure_choices.len(), 2);

    let omitted_closure = closure_choices
        .iter()
        .find(|choice| choice.text() == "1")
        .unwrap();
    assert_eq!(
        omitted_closure
            .successor()
            .structural
            .bond_decision_domain(bonds[0]),
        BondRepresentation::Ring10.singleton_domain()
    );
    assert_eq!(
        [root_text.as_str(), "=", &opening_label, "CC", "1"].concat(),
        "C=1CC1"
    );
    assert!(omitted_closure.successor().is_accepted());

    let emitted_closure = closure_choices
        .into_iter()
        .find(|choice| choice.text() == "=")
        .unwrap()
        .into_successor();
    assert_eq!(
        emitted_closure.structural.bond_decision_domain(bonds[0]),
        BondRepresentation::Ring11.singleton_domain()
    );
    let (label, both) = only_choice(&emitted_closure, "1");
    assert_eq!(
        [root_text.as_str(), "=", &opening_label, "CC=", &label,].concat(),
        "C=1CC=1"
    );
    assert!(both.is_accepted());
}

fn independent_ring_endpoint_projection(
    domain: Domain,
    endpoint: FixedBondEndpoint,
    spelling: RingEndpointSpelling,
) -> Domain {
    Domain::from_indices(domain.iter().filter(|value| {
        let placement = match *value {
            value if value == BondRepresentation::Traversal.value_index() => return false,
            value if value == BondRepresentation::Ring00.value_index() => (false, false),
            value if value == BondRepresentation::Ring10.value_index() => (true, false),
            value if value == BondRepresentation::Ring01.value_index() => (false, true),
            value if value == BondRepresentation::Ring11.value_index() => (true, true),
            _ => panic!("ring-plan oracle received an unknown representation value"),
        };
        let emitted = match endpoint {
            FixedBondEndpoint::A => placement.0,
            FixedBondEndpoint::B => placement.1,
        };
        emitted == (spelling == RingEndpointSpelling::Emit)
    }))
    .unwrap()
}

#[test]
fn endpoint_projection_masks_match_an_independent_ring_plan_oracle() {
    for bits in 0_u64..(1_u64 << 5) {
        let domain = Domain::from_bits(bits);
        for endpoint in [FixedBondEndpoint::A, FixedBondEndpoint::B] {
            for spelling in [RingEndpointSpelling::Omit, RingEndpointSpelling::Emit] {
                assert_eq!(
                    domain.intersect(BondRepresentation::endpoint_domain(endpoint, spelling)),
                    independent_ring_endpoint_projection(domain, endpoint, spelling),
                    "domain {domain:?}, endpoint {endpoint:?}, spelling {spelling:?}"
                );
            }
        }
    }
}

fn fixed_ring_placement_results_from(
    token: NonStereoBondToken,
    root_index: usize,
) -> Vec<(String, Domain)> {
    let (surface, atoms, bonds) = fixture(
        &["A", "B", "C"],
        &[
            (0, 1, token),
            (0, 2, NonStereoBondToken::Elided),
            (1, 2, NonStereoBondToken::Elided),
        ],
    );
    let initial = initial(&surface);
    let root = initial
        .choices()
        .unwrap()
        .into_iter()
        .find(|choice| choice.successor().active_atom() == Some(atoms[root_index]))
        .unwrap();
    let root_text = root.text().to_owned();
    let mut pending = root
        .successor()
        .choices()
        .unwrap()
        .into_iter()
        .filter(|choice| {
            choice
                .successor()
                .labels
                .bonds_by_slot
                .values()
                .any(|bond| *bond == bonds[0])
        })
        .map(|choice| {
            let prefix = format!("{root_text}{}", choice.text());
            (choice.into_successor(), prefix)
        })
        .collect::<Vec<_>>();
    let mut complete = Vec::new();
    let mut visited_state_count = 0_usize;
    while let Some((state, prefix)) = pending.pop() {
        visited_state_count += 1;
        assert!(
            visited_state_count <= 100_000,
            "explicit ring exploration exceeded its bounded test envelope"
        );
        if state.is_accepted() {
            let plan = state.structural.bond_decision_domain(bonds[0]);
            assert!(plan.is_singleton());
            complete.push((prefix, plan));
            continue;
        }
        for choice in state.choices().unwrap() {
            let text = choice.text().to_owned();
            pending.push((choice.into_successor(), format!("{prefix}{text}")));
        }
    }
    complete.sort_by(|left, right| left.0.cmp(&right.0));
    complete
}

fn fixed_ring_placement_results(token: NonStereoBondToken) -> Vec<(String, Domain)> {
    fixed_ring_placement_results_from(token, 0)
}

fn assert_ring_placement_mapping(
    token: NonStereoBondToken,
    root_index: usize,
    expected: &[(&str, BondRepresentation)],
) {
    let actual = fixed_ring_placement_results_from(token, root_index);
    assert_eq!(actual.len(), expected.len());
    for &(text, plan) in expected {
        assert!(
            actual.iter().any(|(actual_text, actual_plan)| {
                actual_text == text && *actual_plan == plan.singleton_domain()
            }),
            "missing ring placement {text:?} -> {plan:?} in {actual:?}"
        );
    }
}

#[test]
fn every_prepared_ring_token_uses_endpoint_relative_placement() {
    let standard = [
        (NonStereoBondToken::Aromatic, ":"),
        (NonStereoBondToken::Single, "-"),
        (NonStereoBondToken::Double, "="),
        (NonStereoBondToken::Triple, "#"),
    ];
    for (token, text) in standard {
        assert_ring_placement_mapping(
            token,
            0,
            &[
                (&format!("A{text}1CB1"), BondRepresentation::Ring10),
                (&format!("A1CB{text}1"), BondRepresentation::Ring01),
                (&format!("A{text}1CB{text}1"), BondRepresentation::Ring11),
            ],
        );
    }

    assert_ring_placement_mapping(
        NonStereoBondToken::DativeAToB,
        0,
        &[
            ("A->1CB1", BondRepresentation::Ring10),
            ("A1CB<-1", BondRepresentation::Ring01),
            ("A->1CB<-1", BondRepresentation::Ring11),
        ],
    );
    assert_ring_placement_mapping(
        NonStereoBondToken::DativeBToA,
        0,
        &[
            ("A<-1CB1", BondRepresentation::Ring10),
            ("A1CB->1", BondRepresentation::Ring01),
            ("A<-1CB->1", BondRepresentation::Ring11),
        ],
    );
    assert_ring_placement_mapping(
        NonStereoBondToken::DativeAToB,
        1,
        &[
            ("B<-1CA1", BondRepresentation::Ring01),
            ("B1CA->1", BondRepresentation::Ring10),
            ("B<-1CA->1", BondRepresentation::Ring11),
        ],
    );
    assert_ring_placement_mapping(
        NonStereoBondToken::DativeBToA,
        1,
        &[
            ("B->1CA1", BondRepresentation::Ring01),
            ("B1CA<-1", BondRepresentation::Ring10),
            ("B->1CA<-1", BondRepresentation::Ring11),
        ],
    );
    assert_eq!(
        fixed_ring_placement_results(NonStereoBondToken::Elided),
        vec![(
            "A1CB1".to_owned(),
            BondRepresentation::Ring00.singleton_domain(),
        )]
    );
}

#[test]
fn directed_ring_closure_uses_the_emitting_fixed_endpoint_orientation() {
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

    let (bond, pending_label) = only_choice(&walked, "<-");
    let (label, accepted) = only_choice(&pending_label, "1");

    assert_eq!(
        [root, open, first_child, second_child, bond, label].concat(),
        "N1CB<-1"
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
    assert!(bonds[..3].iter().all(|bond| boundary
        .structural
        .bond_decision_domain(*bond)
        .is_singleton()));
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
    assert!(bonds.iter().all(|bond| accepted
        .structural
        .bond_decision_domain(*bond)
        .is_singleton()));

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
