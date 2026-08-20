from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import z3
from rdkit import Chem, rdBase


PINNED_CASES = {
    "github3967_ring": {
        "smiles": r"C1=CC/C=C2C3=C/CC=CC=CC\3C\2C=C1",
        "why": "ring-closure carrier alternatives around coupled double bonds",
    },
    "manual_difficult_cis_cis": {
        "smiles": r"CC/C=C\C(CO)=C(/C)CC",
        "why": "acyclic adjacent double-bond carrier choice",
    },
    "manual_stereo_atoms_missing_surface": {
        "smiles": r"CC\C=C/C(/C=C/CC)=C(/CC)CO",
        "why": "endpoint with two directed candidate carriers",
    },
    "dataset_regression_02_porphyrin_like_fragment": {
        "smiles": r"C1=CC=C2/C/3=N/C4=C5C(=C([N-]4)/N=C/6\[N-]/C(=N\C7=C8C(=C([N-]7)/N=C(/C2=C1)\[N-]3)C=CC=C8)/C9=CC=CC=C69)C=CC=C5.[Cu]",
        "why": "large macrocycle with traversal-correlated carrier choices",
    },
    "latent_central_unspecified": {
        "smiles": r"CCC=CC(C=CCC)=C(CO)CC",
        "why": "potential-but-not-writer-effective central alkene",
    },
}

DIRECTED_DIRS = {Chem.BondDir.ENDUPRIGHT, Chem.BondDir.ENDDOWNRIGHT}
E_OR_Z = {
    Chem.BondStereo.STEREOE,
    Chem.BondStereo.STEREOTRANS,
    Chem.BondStereo.STEREOZ,
    Chem.BondStereo.STEREOCIS,
}
Z_LIKE = {Chem.BondStereo.STEREOZ, Chem.BondStereo.STEREOCIS}


@dataclass(frozen=True)
class StereoBondSpec:
    source_bond_idx: int
    begin: int
    end: int
    is_z: bool
    stereo_atoms: tuple[int, int]


@dataclass(frozen=True)
class EndpointSpec:
    stereo_bond_id: int
    endpoint: int
    other_endpoint: int
    selected_neighbor: int
    candidate_neighbors: tuple[int, ...]


@dataclass(frozen=True)
class CarrierIncidence:
    endpoint_id: int
    endpoint: int
    neighbor: int
    edge: tuple[int, int]
    is_selected_stereo_atom: bool


@dataclass(frozen=True)
class HazardSpec:
    source_bond_idx: int
    begin: int
    end: int
    begin_candidate_edges: tuple[tuple[int, int], ...]
    end_candidate_edges: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class StereoSystem:
    smiles: str
    stereo_bonds: tuple[StereoBondSpec, ...]
    endpoints: tuple[EndpointSpec, ...]
    incidences: tuple[CarrierIncidence, ...]
    hazards: tuple[HazardSpec, ...]


@dataclass(frozen=True)
class ObservedOutput:
    smiles: str
    shapes: frozenset[tuple[tuple[int, int], ...]]
    preserving_match_count: int


@dataclass(frozen=True)
class ObservationReport:
    output_count: int
    no_preserving_match_count: int
    shape_counts: Counter[tuple[tuple[int, int], ...]]
    outputs_without_preserving_match: tuple[str, ...]


@dataclass(frozen=True)
class ConstraintHypothesis:
    name: str
    directed_edges_observed_by_all_incident_endpoints: bool
    allow_two_candidate_endpoint_only_when_all_edges_shared: bool
    allow_traversal_flip_variables: bool
    enforce_strict_hazards: bool


HYPOTHESES = (
    ConstraintHypothesis(
        name="local-incidence",
        directed_edges_observed_by_all_incident_endpoints=False,
        allow_two_candidate_endpoint_only_when_all_edges_shared=False,
        allow_traversal_flip_variables=False,
        enforce_strict_hazards=False,
    ),
    ConstraintHypothesis(
        name="observed-edge-shared-pair",
        directed_edges_observed_by_all_incident_endpoints=True,
        allow_two_candidate_endpoint_only_when_all_edges_shared=True,
        allow_traversal_flip_variables=False,
        enforce_strict_hazards=False,
    ),
)


def canonical_edge(left: int, right: int) -> tuple[int, int]:
    return (left, right) if left < right else (right, left)


def atom_signature(atom: Chem.Atom) -> tuple[object, ...]:
    return (
        atom.GetAtomicNum(),
        atom.GetFormalCharge(),
        atom.GetTotalNumHs(),
        atom.GetIsAromatic(),
        atom.GetIsotope(),
    )


def bond_signature(bond: Chem.Bond) -> tuple[object, ...]:
    return (
        str(bond.GetBondType()),
        bond.GetIsAromatic(),
        bond.GetIsConjugated(),
    )


def strip_stereo(mol: Chem.Mol) -> Chem.Mol:
    copy = Chem.Mol(mol)
    Chem.RemoveStereochemistry(copy)
    return copy


def rdkit_automorphisms(mol: Chem.Mol) -> tuple[tuple[int, ...], ...]:
    stripped = strip_stereo(mol)
    return tuple(
        stripped.GetSubstructMatches(
            stripped,
            uniquify=False,
            useChirality=False,
            maxMatches=100000,
        )
    )


def z3_automorphisms(mol: Chem.Mol, *, limit: int = 100000) -> tuple[tuple[int, ...], ...]:
    stripped = strip_stereo(mol)
    atom_count = stripped.GetNumAtoms()
    mapping = [z3.Int(f"auto_map_{idx}") for idx in range(atom_count)]
    solver = z3.Solver()

    for source_idx, var in enumerate(mapping):
        allowed = [
            target_idx
            for target_idx in range(atom_count)
            if atom_signature(stripped.GetAtomWithIdx(source_idx))
            == atom_signature(stripped.GetAtomWithIdx(target_idx))
        ]
        solver.add(z3.Or(*(var == target_idx for target_idx in allowed)))
    solver.add(z3.Distinct(*mapping))

    signature_to_edges: dict[tuple[object, ...], list[tuple[int, int]]] = defaultdict(list)
    for bond in stripped.GetBonds():
        signature_to_edges[bond_signature(bond)].append(
            canonical_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
        )

    for bond in stripped.GetBonds():
        begin = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        allowed_edges = signature_to_edges[bond_signature(bond)]
        solver.add(
            z3.Or(
                *(
                    z3.Or(
                        z3.And(mapping[begin] == left, mapping[end] == right),
                        z3.And(mapping[begin] == right, mapping[end] == left),
                    )
                    for left, right in allowed_edges
                )
            )
        )

    out = []
    while len(out) < limit and solver.check() == z3.sat:
        model = solver.model()
        assignment = tuple(int(model.eval(var, model_completion=True).as_long()) for var in mapping)
        out.append(assignment)
        solver.add(z3.Or(*(var != value for var, value in zip(mapping, assignment))))
    return tuple(out)


def stereo_bond_specs(mol: Chem.Mol) -> tuple[StereoBondSpec, ...]:
    out = []
    for bond in mol.GetBonds():
        if bond.GetBondType() != Chem.BondType.DOUBLE or bond.GetStereo() not in E_OR_Z:
            continue
        stereo_atoms = tuple(int(idx) for idx in bond.GetStereoAtoms())
        if len(stereo_atoms) != 2:
            continue
        out.append(
            StereoBondSpec(
                source_bond_idx=bond.GetIdx(),
                begin=bond.GetBeginAtomIdx(),
                end=bond.GetEndAtomIdx(),
                is_z=bond.GetStereo() in Z_LIKE,
                stereo_atoms=stereo_atoms,
            )
        )
    return tuple(out)


def candidate_neighbors(mol: Chem.Mol, endpoint: int, other_endpoint: int) -> tuple[int, ...]:
    out = []
    for neighbor in mol.GetAtomWithIdx(endpoint).GetNeighbors():
        neighbor_idx = neighbor.GetIdx()
        if neighbor_idx == other_endpoint:
            continue
        bond = mol.GetBondBetweenAtoms(endpoint, neighbor_idx)
        if bond is None:
            continue
        if bond.GetBondType() in (Chem.BondType.SINGLE, Chem.BondType.AROMATIC):
            out.append(neighbor_idx)
    return tuple(sorted(out))


def endpoint_specs(mol: Chem.Mol, stereo_bonds: tuple[StereoBondSpec, ...]) -> tuple[EndpointSpec, ...]:
    out = []
    for stereo_bond_id, stereo_bond in enumerate(stereo_bonds):
        for endpoint, other_endpoint, selected_neighbor in (
            (stereo_bond.begin, stereo_bond.end, stereo_bond.stereo_atoms[0]),
            (stereo_bond.end, stereo_bond.begin, stereo_bond.stereo_atoms[1]),
        ):
            candidates = candidate_neighbors(mol, endpoint, other_endpoint)
            if not candidates:
                continue
            out.append(
                EndpointSpec(
                    stereo_bond_id=stereo_bond_id,
                    endpoint=endpoint,
                    other_endpoint=other_endpoint,
                    selected_neighbor=selected_neighbor,
                    candidate_neighbors=candidates,
                )
            )
    return tuple(out)


def carrier_incidences(endpoints: tuple[EndpointSpec, ...]) -> tuple[CarrierIncidence, ...]:
    out = []
    for endpoint_id, endpoint in enumerate(endpoints):
        for neighbor in endpoint.candidate_neighbors:
            out.append(
                CarrierIncidence(
                    endpoint_id=endpoint_id,
                    endpoint=endpoint.endpoint,
                    neighbor=neighbor,
                    edge=canonical_edge(endpoint.endpoint, neighbor),
                    is_selected_stereo_atom=neighbor == endpoint.selected_neighbor,
                )
            )
    return tuple(out)


def hazard_specs(mol: Chem.Mol, incidences: tuple[CarrierIncidence, ...]) -> tuple[HazardSpec, ...]:
    stereo_candidate_edges = {incidence.edge for incidence in incidences}
    out = []
    for bond in mol.GetBonds():
        if bond.GetBondType() != Chem.BondType.DOUBLE or bond.GetStereo() in E_OR_Z:
            continue
        begin = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        begin_edges = tuple(canonical_edge(begin, neighbor) for neighbor in candidate_neighbors(mol, begin, end))
        end_edges = tuple(canonical_edge(end, neighbor) for neighbor in candidate_neighbors(mol, end, begin))
        if not begin_edges or not end_edges:
            continue
        if any(edge in stereo_candidate_edges for edge in begin_edges) and any(
            edge in stereo_candidate_edges for edge in end_edges
        ):
            out.append(
                HazardSpec(
                    source_bond_idx=bond.GetIdx(),
                    begin=begin,
                    end=end,
                    begin_candidate_edges=begin_edges,
                    end_candidate_edges=end_edges,
                )
            )
    return tuple(out)


def stereo_system_from_mol(smiles: str, mol: Chem.Mol) -> StereoSystem:
    stereo_bonds = stereo_bond_specs(mol)
    endpoints = endpoint_specs(mol, stereo_bonds)
    incidences = carrier_incidences(endpoints)
    hazards = hazard_specs(mol, incidences)
    return StereoSystem(
        smiles=smiles,
        stereo_bonds=stereo_bonds,
        endpoints=endpoints,
        incidences=incidences,
        hazards=hazards,
    )


class Z3CarrierModel:
    def __init__(self, system: StereoSystem, hypothesis: ConstraintHypothesis) -> None:
        self.system = system
        self.hypothesis = hypothesis
        self.solver = z3.Solver()
        self.edge_vars = {
            incidence.edge: z3.Bool(f"edge_directed_{incidence.edge[0]}_{incidence.edge[1]}")
            for incidence in system.incidences
        }
        self.token_vars = {
            incidence.edge: z3.Bool(f"edge_backslash_{incidence.edge[0]}_{incidence.edge[1]}")
            for incidence in system.incidences
        }
        self.use_vars = {
            (incidence.endpoint_id, incidence.edge): z3.Bool(
                f"endpoint_{incidence.endpoint_id}_uses_{incidence.edge[0]}_{incidence.edge[1]}"
            )
            for incidence in system.incidences
        }
        self.traversal_flip_vars = {
            (incidence.endpoint_id, incidence.edge): z3.Bool(
                f"endpoint_{incidence.endpoint_id}_flip_{incidence.edge[0]}_{incidence.edge[1]}"
            )
            for incidence in system.incidences
        }
        self._add_constraints()

    def _incidences_for_endpoint(self, endpoint_id: int) -> tuple[CarrierIncidence, ...]:
        return tuple(
            incidence
            for incidence in self.system.incidences
            if incidence.endpoint_id == endpoint_id
        )

    def _effective_token(self, incidence: CarrierIncidence) -> z3.BoolRef:
        return z3.Xor(
            self.token_vars[incidence.edge],
            z3.BoolVal(not incidence.is_selected_stereo_atom),
            self.traversal_flip_vars[(incidence.endpoint_id, incidence.edge)],
        )

    def _add_constraints(self) -> None:
        if not self.hypothesis.allow_traversal_flip_variables:
            for var in self.traversal_flip_vars.values():
                self.solver.add(var == False)

        for incidence in self.system.incidences:
            use = self.use_vars[(incidence.endpoint_id, incidence.edge)]
            directed = self.edge_vars[incidence.edge]
            if self.hypothesis.directed_edges_observed_by_all_incident_endpoints:
                self.solver.add(use == directed)
            else:
                self.solver.add(z3.Implies(use, directed))

        incidence_count_by_edge = Counter(incidence.edge for incidence in self.system.incidences)
        for endpoint_id, endpoint in enumerate(self.system.endpoints):
            uses = [
                self.use_vars[(incidence.endpoint_id, incidence.edge)]
                for incidence in self._incidences_for_endpoint(endpoint_id)
            ]
            self.solver.add(z3.Or(*uses))
            if len(endpoint.candidate_neighbors) == 1:
                self.solver.add(uses[0])
            elif self.hypothesis.allow_two_candidate_endpoint_only_when_all_edges_shared:
                all_edges_shared = all(
                    incidence_count_by_edge[canonical_edge(endpoint.endpoint, candidate)] > 1
                    for candidate in endpoint.candidate_neighbors
                )
                if not all_edges_shared:
                    self.solver.add(z3.AtMost(*uses, 1))

        endpoint_ids_by_stereo_bond: dict[int, list[int]] = defaultdict(list)
        for endpoint_id, endpoint in enumerate(self.system.endpoints):
            endpoint_ids_by_stereo_bond[endpoint.stereo_bond_id].append(endpoint_id)
        for stereo_bond_id, stereo_bond in enumerate(self.system.stereo_bonds):
            endpoint_ids = endpoint_ids_by_stereo_bond[stereo_bond_id]
            if len(endpoint_ids) != 2:
                continue
            left_incidences = self._incidences_for_endpoint(endpoint_ids[0])
            right_incidences = self._incidences_for_endpoint(endpoint_ids[1])
            for left in left_incidences:
                for right in right_incidences:
                    left_use = self.use_vars[(left.endpoint_id, left.edge)]
                    right_use = self.use_vars[(right.endpoint_id, right.edge)]
                    relation_is_z = z3.Xor(self._effective_token(left), self._effective_token(right))
                    self.solver.add(
                        z3.Implies(
                            z3.And(left_use, right_use),
                            relation_is_z == z3.BoolVal(stereo_bond.is_z),
                        )
                    )

        by_edge: dict[tuple[int, int], list[CarrierIncidence]] = defaultdict(list)
        for incidence in self.system.incidences:
            by_edge[incidence.edge].append(incidence)
        for edge, incidences in by_edge.items():
            self.solver.add(
                self.edge_vars[edge]
                == z3.Or(*(self.use_vars[(incidence.endpoint_id, incidence.edge)] for incidence in incidences))
            )

        if self.hypothesis.enforce_strict_hazards:
            for hazard in self.system.hazards:
                begin_vars = [self.edge_vars[edge] for edge in hazard.begin_candidate_edges if edge in self.edge_vars]
                end_vars = [self.edge_vars[edge] for edge in hazard.end_candidate_edges if edge in self.edge_vars]
                if begin_vars and end_vars:
                    self.solver.add(z3.Not(z3.And(z3.Or(*begin_vars), z3.Or(*end_vars))))

    def projected_directed_edge_shapes(self, *, limit: int = 100000) -> frozenset[tuple[tuple[int, int], ...]]:
        shapes = set()
        self.solver.push()
        while len(shapes) < limit and self.solver.check() == z3.sat:
            model = self.solver.model()
            shape = tuple(
                edge
                for edge in sorted(self.edge_vars)
                if bool(model.eval(self.edge_vars[edge], model_completion=True))
            )
            shapes.add(shape)
            self.solver.add(
                z3.Or(
                    *(self.edge_vars[edge] != bool(edge in shape) for edge in sorted(self.edge_vars))
                )
            )
        self.solver.pop()
        return frozenset(shapes)


def sample_rdkit_outputs(mol: Chem.Mol, *, samples: int) -> tuple[str, ...]:
    outputs = {Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)}
    for seed in range(samples):
        Chem.rdBase.SeedRandomNumberGenerator(seed)
        outputs.add(
            Chem.MolToSmiles(
                Chem.Mol(mol),
                canonical=False,
                doRandom=True,
                isomericSmiles=True,
            )
        )
    return tuple(sorted(outputs))


def preserves_source_stereo(
    source: Chem.Mol,
    output: Chem.Mol,
    match: tuple[int, ...],
    stereo_bonds: tuple[StereoBondSpec, ...],
) -> bool:
    for stereo_bond in stereo_bonds:
        source_bond = source.GetBondBetweenAtoms(stereo_bond.begin, stereo_bond.end)
        output_bond = output.GetBondBetweenAtoms(match[stereo_bond.begin], match[stereo_bond.end])
        if output_bond is None or output_bond.GetStereo() != source_bond.GetStereo():
            return False
    return True


def preserving_matches(
    source: Chem.Mol,
    output: Chem.Mol,
    stereo_bonds: tuple[StereoBondSpec, ...],
) -> tuple[tuple[int, ...], ...]:
    matches = strip_stereo(output).GetSubstructMatches(
        strip_stereo(source),
        uniquify=False,
        useChirality=False,
        maxMatches=100000,
    )
    return tuple(match for match in matches if preserves_source_stereo(source, output, match, stereo_bonds))


def mapped_directed_edges(output: Chem.Mol, match: tuple[int, ...]) -> tuple[tuple[int, int], ...]:
    inverse = {target: source for source, target in enumerate(match)}
    edges = []
    for bond in output.GetBonds():
        if bond.GetBondDir() not in DIRECTED_DIRS:
            continue
        begin = inverse.get(bond.GetBeginAtomIdx())
        end = inverse.get(bond.GetEndAtomIdx())
        if begin is None or end is None:
            continue
        edges.append(canonical_edge(begin, end))
    return tuple(sorted(set(edges)))


def observe_rdkit_shapes(
    source: Chem.Mol,
    system: StereoSystem,
    *,
    samples: int,
) -> ObservationReport:
    shape_counts: Counter[tuple[tuple[int, int], ...]] = Counter()
    no_preserving = []
    for output_smiles in sample_rdkit_outputs(source, samples=samples):
        output = Chem.MolFromSmiles(output_smiles)
        if output is None:
            no_preserving.append(output_smiles)
            continue
        matches = preserving_matches(source, output, system.stereo_bonds)
        if not matches and system.stereo_bonds:
            no_preserving.append(output_smiles)
            continue
        shapes = {mapped_directed_edges(output, match) for match in matches} if matches else {()}
        for shape in shapes:
            shape_counts[shape] += 1
    return ObservationReport(
        output_count=sum(shape_counts.values()) + len(no_preserving),
        no_preserving_match_count=len(no_preserving),
        shape_counts=shape_counts,
        outputs_without_preserving_match=tuple(no_preserving[:8]),
    )


def component_summary(system: StereoSystem) -> tuple[tuple[str, ...], ...]:
    graph: dict[str, set[str]] = defaultdict(set)
    nodes = set()

    def connect(left: str, right: str) -> None:
        nodes.add(left)
        nodes.add(right)
        graph[left].add(right)
        graph[right].add(left)

    for stereo_bond_id, stereo_bond in enumerate(system.stereo_bonds):
        bond_node = f"bond:{stereo_bond_id}:{stereo_bond.begin}={stereo_bond.end}"
        nodes.add(bond_node)
    for endpoint_id, endpoint in enumerate(system.endpoints):
        connect(
            f"bond:{endpoint.stereo_bond_id}:{system.stereo_bonds[endpoint.stereo_bond_id].begin}={system.stereo_bonds[endpoint.stereo_bond_id].end}",
            f"endpoint:{endpoint_id}:{endpoint.endpoint}",
        )
    for incidence in system.incidences:
        connect(
            f"endpoint:{incidence.endpoint_id}:{incidence.endpoint}",
            f"edge:{incidence.edge[0]}-{incidence.edge[1]}",
        )
    for hazard_id, hazard in enumerate(system.hazards):
        hazard_node = f"hazard:{hazard_id}:{hazard.begin}={hazard.end}"
        nodes.add(hazard_node)
        for edge in hazard.begin_candidate_edges + hazard.end_candidate_edges:
            connect(hazard_node, f"edge:{edge[0]}-{edge[1]}")

    seen = set()
    out = []
    for node in sorted(nodes):
        if node in seen:
            continue
        stack = [node]
        seen.add(node)
        component = []
        while stack:
            current = stack.pop()
            component.append(current)
            for neighbor in graph[current]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
        out.append(tuple(sorted(component)))
    return tuple(out)


def load_cases(case_names: Iterable[str]) -> dict[str, str]:
    return {name: PINNED_CASES[name]["smiles"] for name in case_names}


def print_case_report(name: str, smiles: str, *, samples: int) -> None:
    source = Chem.MolFromSmiles(smiles)
    if source is None:
        print("=" * 120)
        print(name)
        print("source parse failed")
        return

    system = stereo_system_from_mol(smiles, source)
    rdkit_autos = rdkit_automorphisms(source)
    z3_autos = z3_automorphisms(source)
    observations = observe_rdkit_shapes(source, system, samples=samples)

    print("=" * 120)
    print(name)
    if name in PINNED_CASES:
        print("why:", PINNED_CASES[name]["why"])
    print("source:", smiles)
    print("canonical:", Chem.MolToSmiles(source, canonical=True, isomericSmiles=True))
    print(
        "automorphisms:",
        f"rdkit={len(rdkit_autos)}",
        f"z3={len(z3_autos)}",
        f"agree={set(rdkit_autos) == set(z3_autos)}",
    )
    print("stereo_bonds:", len(system.stereo_bonds), system.stereo_bonds)
    print("endpoints:", len(system.endpoints), system.endpoints)
    print("incidences:", len(system.incidences), system.incidences)
    print("hazards:", len(system.hazards), system.hazards)
    print("components:", len(component_summary(system)))
    for component in component_summary(system):
        print("  component:")
        for item in component:
            print("   ", item)

    observed_shapes = frozenset(observations.shape_counts)
    print(
        "rdkit_observations:",
        f"samples={samples}",
        f"outputs_or_shape_hits={observations.output_count}",
        f"shapes={len(observed_shapes)}",
        f"no_preserving_match={observations.no_preserving_match_count}",
    )
    for shape, count in observations.shape_counts.most_common(12):
        print("  rdkit shape:", count, shape)
    if observations.outputs_without_preserving_match:
        print("  no preserving match examples:")
        for output in observations.outputs_without_preserving_match:
            print("   ", output)

    for hypothesis in HYPOTHESES:
        model = Z3CarrierModel(system, hypothesis)
        model_shapes = model.projected_directed_edge_shapes()
        model_extra = model_shapes - observed_shapes
        observed_missing = observed_shapes - model_shapes
        print(
            "hypothesis:",
            hypothesis.name,
            f"model_shapes={len(model_shapes)}",
            f"rdkit_subset_of_model={not observed_missing}",
            f"model_extra={len(model_extra)}",
            f"rdkit_missing_from_model={len(observed_missing)}",
        )
        for shape in sorted(model_extra)[:8]:
            print("  model-only shape:", shape)
        for shape in sorted(observed_missing)[:8]:
            print("  rdkit-only shape:", shape)
        if observed_missing:
            print("  interpretation: model too strict, mapping bug, or RDKit behavior outside this hypothesis")
        elif model_extra:
            print("  interpretation: under-sampling, missing traversal constraints, or possible RDKit omission/bug")
        else:
            print("  interpretation: local hypothesis matches observed carrier-shape projection")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=1024)
    parser.add_argument("--case", action="append", choices=sorted(PINNED_CASES))
    parser.add_argument("--json", type=Path, help="Optional JSON file with {name: smiles} cases")
    args = parser.parse_args()

    cases = {}
    if args.json:
        cases.update(json.loads(args.json.read_text()))
    else:
        cases.update(load_cases(args.case or PINNED_CASES.keys()))

    print("rdkit:", rdBase.rdkitVersion)
    print("legacy_stereo_perception:", Chem.GetUseLegacyStereoPerception())
    print("z3:", z3.get_version_string())
    print("principle: RDKit outputs are observations; mismatches are investigation targets, not automatic truth labels.")
    for name, smiles in cases.items():
        print_case_report(name, smiles, samples=args.samples)


if __name__ == "__main__":
    main()
