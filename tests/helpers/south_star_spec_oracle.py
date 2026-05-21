from __future__ import annotations

"""Test-only South Star semantic spec oracle skeleton.

This module is evidence, not generation authority. Most helpers consume
candidate strings produced elsewhere and check them against the declared South
Star grammar and graph/stereo semantic identity contract. The small-support
helper below deliberately duplicates one-atom/diatomic rendering concepts to
provide an implementation-independent completeness witness for tiny domains;
it must not grow into a runtime path. Runtime/package code must not import this
helper, and tests must not treat parser/filter acceptance as the mechanism by
which EnumS support is generated.
"""

from dataclasses import dataclass

from rdkit import Chem

from grimace._south_star.molecule_facts import SouthStarMoleculeFacts
from tests.helpers.south_star_semantic_oracle import SouthStarConformanceReport
from tests.helpers.south_star_semantic_oracle import parse_smiles
from tests.helpers.south_star_semantic_oracle import south_star_conformance_report


SOUTH_STAR_SPEC_ORACLE_BASIS = "test_only_south_star_semantic_spec_oracle"
SOUTH_STAR_SPEC_ORACLE_GENERATION_AUTHORITY = "not_generation_authority"
SOUTH_STAR_SMALL_SUPPORT_ORACLE_BASIS = (
    "test_only_small_atom_bond_support_oracle"
)
SOUTH_STAR_SMALL_SUPPORT_SHARED_RECORD_BASIS = (
    "south_star_molecule_facts_atom_bond_text_records"
)


@dataclass(frozen=True, slots=True)
class SouthStarSpecOracleCandidateReport:
    candidate_smiles: str
    conformance_report: SouthStarConformanceReport

    @property
    def accepted(self) -> bool:
        return self.conformance_report.accepted

    @property
    def rejection_reasons(self) -> tuple[str, ...]:
        return self.conformance_report.rejection_reasons


@dataclass(frozen=True, slots=True)
class SouthStarSpecOracleReport:
    source_smiles: str
    candidate_reports: tuple[SouthStarSpecOracleCandidateReport, ...]
    basis: str = SOUTH_STAR_SPEC_ORACLE_BASIS
    generation_authority: str = SOUTH_STAR_SPEC_ORACLE_GENERATION_AUTHORITY

    @property
    def candidate_count(self) -> int:
        return len(self.candidate_reports)

    @property
    def accepted_count(self) -> int:
        return sum(1 for report in self.candidate_reports if report.accepted)

    @property
    def all_accepted(self) -> bool:
        return self.accepted_count == self.candidate_count

    @property
    def rejected_candidates(self) -> tuple[SouthStarSpecOracleCandidateReport, ...]:
        return tuple(report for report in self.candidate_reports if not report.accepted)


@dataclass(frozen=True, slots=True)
class SouthStarSmallSupportCompletenessReport:
    source_smiles: str
    expected_support: tuple[str, ...]
    observed_support: tuple[str, ...]
    atom_text_fact_count: int
    bond_text_fact_count: int
    connected: bool
    shared_record_basis: str = SOUTH_STAR_SMALL_SUPPORT_SHARED_RECORD_BASIS
    basis: str = SOUTH_STAR_SMALL_SUPPORT_ORACLE_BASIS
    generation_authority: str = SOUTH_STAR_SPEC_ORACLE_GENERATION_AUTHORITY

    @property
    def missing_candidates(self) -> tuple[str, ...]:
        observed = set(self.observed_support)
        return tuple(
            candidate
            for candidate in self.expected_support
            if candidate not in observed
        )

    @property
    def extra_candidates(self) -> tuple[str, ...]:
        expected = set(self.expected_support)
        return tuple(
            candidate
            for candidate in self.observed_support
            if candidate not in expected
        )

    @property
    def complete(self) -> bool:
        return not self.missing_candidates and not self.extra_candidates


def south_star_spec_oracle_report(
    *,
    source_smiles: str,
    candidate_smiles: tuple[str, ...],
) -> SouthStarSpecOracleReport:
    return SouthStarSpecOracleReport(
        source_smiles=source_smiles,
        candidate_reports=tuple(
            SouthStarSpecOracleCandidateReport(
                candidate_smiles=candidate,
                conformance_report=south_star_conformance_report(
                    source_smiles=source_smiles,
                    candidate_smiles=candidate,
                ),
            )
            for candidate in candidate_smiles
        ),
    )


def south_star_small_support_completeness_report(
    *,
    source_smiles: str,
    observed_support: tuple[str, ...],
) -> SouthStarSmallSupportCompletenessReport:
    mol = parse_smiles(source_smiles)
    molecule_facts = SouthStarMoleculeFacts.from_mol(mol)
    return SouthStarSmallSupportCompletenessReport(
        source_smiles=source_smiles,
        expected_support=_small_atom_bond_support(mol),
        observed_support=observed_support,
        atom_text_fact_count=len(molecule_facts.atom_text_facts),
        bond_text_fact_count=len(molecule_facts.bond_text_facts),
        connected=molecule_facts.graph_topology.connected,
    )


def _small_atom_bond_support(mol: Chem.Mol) -> tuple[str, ...]:
    if mol.GetNumAtoms() == 1 and mol.GetNumBonds() == 0:
        return (_oracle_atom_text(mol.GetAtomWithIdx(0)),)
    if mol.GetNumAtoms() != 2 or mol.GetNumBonds() != 1:
        raise NotImplementedError(
            "small support oracle currently covers one-atom and diatomic graphs"
        )
    bond = mol.GetBondWithIdx(0)
    begin_text = _oracle_atom_text(mol.GetAtomWithIdx(bond.GetBeginAtomIdx()))
    end_text = _oracle_atom_text(mol.GetAtomWithIdx(bond.GetEndAtomIdx()))
    bond_text = _oracle_bond_text(bond)
    forward = f"{begin_text}{bond_text}{end_text}"
    reverse = f"{end_text}{bond_text}{begin_text}"
    return tuple(dict.fromkeys((forward, reverse)))


def _oracle_atom_text(atom: Chem.Atom) -> str:
    if atom.GetIsAromatic():
        raise NotImplementedError("small support oracle excludes aromatic atoms")
    if atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED:
        raise NotImplementedError("small support oracle excludes atom stereo")
    symbol = atom.GetSymbol()
    if symbol not in {"H", "B", "C", "N", "O", "P", "S", "F", "Cl", "Br", "I"}:
        raise NotImplementedError(f"small support oracle excludes atom {symbol!r}")
    isotope = atom.GetIsotope()
    charge = atom.GetFormalCharge()
    hydrogens = atom.GetNumExplicitHs()
    atom_map = atom.GetAtomMapNum()
    radical = atom.GetNumRadicalElectrons()
    needs_bracket = (
        symbol == "H" or isotope or charge or hydrogens or atom_map or radical
    )
    if not needs_bracket:
        return symbol
    return (
        "["
        f"{'' if isotope == 0 else isotope}"
        f"{symbol}"
        f"{_oracle_hydrogen_text(hydrogens)}"
        f"{_oracle_charge_text(charge)}"
        f"{'' if atom_map == 0 else f':{atom_map}'}"
        "]"
    )


def _oracle_bond_text(bond: Chem.Bond) -> str:
    bond_type = bond.GetBondType()
    if bond_type == Chem.BondType.SINGLE:
        return ""
    if bond_type == Chem.BondType.DOUBLE:
        return "="
    if bond_type == Chem.BondType.TRIPLE:
        return "#"
    raise NotImplementedError(f"small support oracle excludes bond {bond_type}")


def _oracle_hydrogen_text(hydrogen_count: int) -> str:
    if hydrogen_count == 0:
        return ""
    if hydrogen_count == 1:
        return "H"
    return f"H{hydrogen_count}"


def _oracle_charge_text(charge: int) -> str:
    if charge == 0:
        return ""
    sign = "+" if charge > 0 else "-"
    magnitude = abs(charge)
    return sign if magnitude == 1 else f"{sign}{magnitude}"
