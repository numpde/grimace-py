from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass

import z3
from rdkit import Chem, rdBase


CASES = {
    "github3967_ring": r"C1=CC/C=C2C3=C/CC=CC=CC\3C\2C=C1",
    "manual_difficult_cis_cis": r"CC/C=C\C(CO)=C(/C)CC",
    "manual_stereo_atoms_missing_surface": r"CC\C=C/C(/C=C/CC)=C(/CC)CO",
    "dataset_regression_02_porphyrin_like_fragment": r"C1=CC=C2/C/3=N/C4=C5C(=C([N-]4)/N=C/6\[N-]/C(=N\C7=C8C(=C([N-]7)/N=C(/C2=C1)\[N-]3)C=CC=C8)/C9=CC=CC=C69)C=CC=C5.[Cu]",
    "latent_central_unspecified": r"CCC=CC(C=CCC)=C(CO)CC",
}

DIRS = {Chem.BondDir.ENDUPRIGHT, Chem.BondDir.ENDDOWNRIGHT}
E_OR_Z = {
    Chem.BondStereo.STEREOE,
    Chem.BondStereo.STEREOTRANS,
    Chem.BondStereo.STEREOZ,
    Chem.BondStereo.STEREOCIS,
}
Z_LIKE = {Chem.BondStereo.STEREOZ, Chem.BondStereo.STEREOCIS}


@dataclass(frozen=True)
class StereoBond:
    bond_idx: int
    begin: int
    end: int
    is_z: bool
    stereo_atoms: tuple[int, int]


@dataclass(frozen=True)
class Endpoint:
    stereo_bond_idx: int
    endpoint: int
    other_endpoint: int
    selected_neighbor: int
    candidates: tuple[int, ...]


@dataclass(frozen=True)
class Incidence:
    endpoint_idx: int
    endpoint: int
    neighbor: int
    edge: tuple[int, int]
    selected: bool


@dataclass(frozen=True)
class HazardBond:
    bond_idx: int
    begin: int
    end: int
    begin_candidate_edges: tuple[tuple[int, int], ...]
    end_candidate_edges: tuple[tuple[int, int], ...]


def stereo_bonds(mol: Chem.Mol) -> tuple[StereoBond, ...]:
    out = []
    for bond in mol.GetBonds():
        if bond.GetBondType() != Chem.BondType.DOUBLE or bond.GetStereo() not in E_OR_Z:
            continue
        stereo_atoms = tuple(int(idx) for idx in bond.GetStereoAtoms())
        if len(stereo_atoms) != 2:
            continue
        out.append(
            StereoBond(
                bond_idx=bond.GetIdx(),
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


def endpoints_for(mol: Chem.Mol, bonds: tuple[StereoBond, ...]) -> tuple[Endpoint, ...]:
    out = []
    for bond_idx, bond in enumerate(bonds):
        for endpoint, other_endpoint, selected in (
            (bond.begin, bond.end, bond.stereo_atoms[0]),
            (bond.end, bond.begin, bond.stereo_atoms[1]),
        ):
            candidates = candidate_neighbors(mol, endpoint, other_endpoint)
            if candidates:
                out.append(
                    Endpoint(
                        stereo_bond_idx=bond_idx,
                        endpoint=endpoint,
                        other_endpoint=other_endpoint,
                        selected_neighbor=selected,
                        candidates=candidates,
                    )
                )
    return tuple(out)


def incidences_for(endpoints: tuple[Endpoint, ...]) -> tuple[Incidence, ...]:
    out = []
    for endpoint_idx, endpoint in enumerate(endpoints):
        for neighbor in endpoint.candidates:
            out.append(
                Incidence(
                    endpoint_idx=endpoint_idx,
                    endpoint=endpoint.endpoint,
                    neighbor=neighbor,
                    edge=tuple(sorted((endpoint.endpoint, neighbor))),
                    selected=neighbor == endpoint.selected_neighbor,
                )
            )
    return tuple(out)


def hazard_bonds_for(
    mol: Chem.Mol,
    incidences: tuple[Incidence, ...],
) -> tuple[HazardBond, ...]:
    directed_candidate_edges = {incidence.edge for incidence in incidences}
    out = []
    for bond in mol.GetBonds():
        if bond.GetBondType() != Chem.BondType.DOUBLE or bond.GetStereo() in E_OR_Z:
            continue
        begin = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        begin_edges = tuple(
            tuple(sorted((begin, neighbor_idx)))
            for neighbor_idx in candidate_neighbors(mol, begin, end)
        )
        end_edges = tuple(
            tuple(sorted((end, neighbor_idx)))
            for neighbor_idx in candidate_neighbors(mol, end, begin)
        )
        if not begin_edges or not end_edges:
            continue
        if not (
            any(edge in directed_candidate_edges for edge in begin_edges)
            and any(edge in directed_candidate_edges for edge in end_edges)
        ):
            continue
        out.append(
            HazardBond(
                bond_idx=bond.GetIdx(),
                begin=begin,
                end=end,
                begin_candidate_edges=begin_edges,
                end_candidate_edges=end_edges,
            )
        )
    return tuple(out)


def strip_stereo(mol: Chem.Mol) -> Chem.Mol:
    copy = Chem.Mol(mol)
    Chem.RemoveStereochemistry(copy)
    return copy


def automorphisms(mol: Chem.Mol) -> tuple[tuple[int, ...], ...]:
    stripped = strip_stereo(mol)
    return tuple(
        stripped.GetSubstructMatches(
            stripped,
            uniquify=False,
            useChirality=False,
            maxMatches=100000,
        )
    )


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


def z3_graph_automorphisms(mol: Chem.Mol, *, limit: int = 100000) -> tuple[tuple[int, ...], ...]:
    """Enumerate graph automorphisms with a finite-domain Z3 model.

    This deliberately ignores stereo and mirrors the graph used by the
    automorphism-aware mapping probes: same atoms, same bond signatures, no
    chirality. The point is not to replace RDKit matching, but to make the
    symmetry constraints explicit in the same language as the carrier model.
    """

    stripped = strip_stereo(mol)
    atom_count = stripped.GetNumAtoms()
    mapping = [z3.Int(f"auto_map_{idx}") for idx in range(atom_count)]
    solver = z3.Solver()
    for idx, var in enumerate(mapping):
        allowed_atoms = [
            target_idx
            for target_idx in range(atom_count)
            if atom_signature(stripped.GetAtomWithIdx(idx))
            == atom_signature(stripped.GetAtomWithIdx(target_idx))
        ]
        solver.add(z3.Or(*(var == target_idx for target_idx in allowed_atoms)))
    solver.add(z3.Distinct(*mapping))

    signature_to_edges: dict[tuple[object, ...], list[tuple[int, int]]] = defaultdict(list)
    for bond in stripped.GetBonds():
        edge = tuple(sorted((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())))
        signature_to_edges[bond_signature(bond)].append(edge)

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


def sample_outputs(mol: Chem.Mol, *, samples: int = 512) -> tuple[str, ...]:
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
    bonds: tuple[StereoBond, ...],
) -> bool:
    for bond in bonds:
        source_bond = source.GetBondBetweenAtoms(bond.begin, bond.end)
        output_bond = output.GetBondBetweenAtoms(match[bond.begin], match[bond.end])
        if output_bond is None or output_bond.GetStereo() != source_bond.GetStereo():
            return False
    return True


def preserving_matches(
    source: Chem.Mol,
    output: Chem.Mol,
    bonds: tuple[StereoBond, ...],
) -> tuple[tuple[int, ...], ...]:
    matches = strip_stereo(output).GetSubstructMatches(
        strip_stereo(source),
        uniquify=False,
        useChirality=False,
        maxMatches=100000,
    )
    return tuple(match for match in matches if preserves_source_stereo(source, output, match, bonds))


def mapped_directed_edges(output: Chem.Mol, match: tuple[int, ...]) -> dict[tuple[int, int], bool]:
    inverse = {target: source for source, target in enumerate(match)}
    out = {}
    for bond in output.GetBonds():
        if bond.GetBondDir() not in DIRS:
            continue
        begin = inverse.get(bond.GetBeginAtomIdx())
        end = inverse.get(bond.GetEndAtomIdx())
        if begin is None or end is None:
            continue
        edge = tuple(sorted((begin, end)))
        # Token is stored in the emitted output direction. Mapping this back to
        # a source edge loses traversal orientation; the Z3 model below accounts
        # for that with per-incidence traversal-flip variables.
        out[edge] = bond.GetBondDir() == Chem.BondDir.ENDDOWNRIGHT
    return out


class ConstraintModel:
    def __init__(
        self,
        *,
        endpoints: tuple[Endpoint, ...],
        incidences: tuple[Incidence, ...],
        bonds: tuple[StereoBond, ...],
        hazards: tuple[HazardBond, ...],
        allow_traversal_flips: bool,
        enforce_strict_hazards: bool = False,
        directed_edges_observed_by_all_incident_endpoints: bool = False,
        allow_two_candidate_endpoint_only_when_all_edges_shared: bool = False,
    ) -> None:
        self.endpoints = endpoints
        self.incidences = incidences
        self.bonds = bonds
        self.hazards = hazards
        self.allow_traversal_flips = allow_traversal_flips
        self.enforce_strict_hazards = enforce_strict_hazards
        self.directed_edges_observed_by_all_incident_endpoints = (
            directed_edges_observed_by_all_incident_endpoints
        )
        self.allow_two_candidate_endpoint_only_when_all_edges_shared = (
            allow_two_candidate_endpoint_only_when_all_edges_shared
        )
        self.solver = z3.Solver()

        self.edge_vars = {
            incidence.edge: z3.Bool(f"edge_directed_{incidence.edge[0]}_{incidence.edge[1]}")
            for incidence in incidences
        }
        self.token_vars = {
            incidence.edge: z3.Bool(f"edge_backslash_{incidence.edge[0]}_{incidence.edge[1]}")
            for incidence in incidences
        }
        self.incidence_vars = {
            (incidence.endpoint_idx, incidence.edge): z3.Bool(
                f"endpoint_{incidence.endpoint_idx}_uses_{incidence.edge[0]}_{incidence.edge[1]}"
            )
            for incidence in incidences
        }
        self.traversal_flip_vars = {
            (incidence.endpoint_idx, incidence.edge): z3.Bool(
                f"endpoint_{incidence.endpoint_idx}_traversal_flip_{incidence.edge[0]}_{incidence.edge[1]}"
            )
            for incidence in incidences
        }
        if not allow_traversal_flips:
            for var in self.traversal_flip_vars.values():
                self.solver.add(var == False)

        self._add_constraints()

    def _incidences_for_endpoint(self, endpoint_idx: int) -> list[Incidence]:
        return [
            incidence
            for incidence in self.incidences
            if incidence.endpoint_idx == endpoint_idx
        ]

    def _effective_token(self, incidence: Incidence) -> z3.BoolRef:
        token = self.token_vars[incidence.edge]
        selected_flip = z3.BoolVal(not incidence.selected)
        traversal_flip = self.traversal_flip_vars[(incidence.endpoint_idx, incidence.edge)]
        return z3.Xor(token, selected_flip, traversal_flip)

    def _add_constraints(self) -> None:
        for incidence in self.incidences:
            use = self.incidence_vars[(incidence.endpoint_idx, incidence.edge)]
            directed = self.edge_vars[incidence.edge]
            if self.directed_edges_observed_by_all_incident_endpoints:
                self.solver.add(use == directed)
            else:
                self.solver.add(z3.Implies(use, directed))

        for endpoint_idx, endpoint in enumerate(self.endpoints):
            uses = [
                self.incidence_vars[(incidence.endpoint_idx, incidence.edge)]
                for incidence in self._incidences_for_endpoint(endpoint_idx)
            ]
            self.solver.add(z3.Or(*uses))
            # Degree-2 endpoints have only one valid carrier. Degree-3 endpoints
            # may keep one or both candidate carriers after RDKit cleanup.
            if len(endpoint.candidates) == 1:
                self.solver.add(uses[0])
            elif self.allow_two_candidate_endpoint_only_when_all_edges_shared:
                edge_incidence_counts = Counter(incidence.edge for incidence in self.incidences)
                all_candidate_edges_shared = all(
                    edge_incidence_counts[tuple(sorted((endpoint.endpoint, candidate)))] > 1
                    for candidate in endpoint.candidates
                )
                if not all_candidate_edges_shared:
                    self.solver.add(z3.AtMost(*uses, 1))
            else:
                self.solver.add(z3.AtMost(*uses, len(endpoint.candidates)))

        endpoint_ids_by_bond = defaultdict(list)
        for endpoint_idx, endpoint in enumerate(self.endpoints):
            endpoint_ids_by_bond[endpoint.stereo_bond_idx].append(endpoint_idx)

        for bond_idx, bond in enumerate(self.bonds):
            endpoint_ids = endpoint_ids_by_bond[bond_idx]
            if len(endpoint_ids) != 2:
                continue
            left_incidences = self._incidences_for_endpoint(endpoint_ids[0])
            right_incidences = self._incidences_for_endpoint(endpoint_ids[1])
            for left in left_incidences:
                for right in right_incidences:
                    left_use = self.incidence_vars[(left.endpoint_idx, left.edge)]
                    right_use = self.incidence_vars[(right.endpoint_idx, right.edge)]
                    relation_is_z = z3.Xor(self._effective_token(left), self._effective_token(right))
                    self.solver.add(
                        z3.Implies(
                            z3.And(left_use, right_use),
                            relation_is_z == z3.BoolVal(bond.is_z),
                        )
                    )

        # If one physical edge serves multiple stereo endpoints, it has one
        # physical token. This is the natural shared-carrier consistency rule.
        by_edge = defaultdict(list)
        for incidence in self.incidences:
            by_edge[incidence.edge].append(incidence)
        for edge, edge_incidences in by_edge.items():
            directed = self.edge_vars[edge]
            uses = [
                self.incidence_vars[(incidence.endpoint_idx, incidence.edge)]
                for incidence in edge_incidences
            ]
            self.solver.add(directed == z3.Or(*uses))

        if self.enforce_strict_hazards:
            for hazard in self.hazards:
                begin_directed = [
                    self.edge_vars[edge]
                    for edge in hazard.begin_candidate_edges
                    if edge in self.edge_vars
                ]
                end_directed = [
                    self.edge_vars[edge]
                    for edge in hazard.end_candidate_edges
                    if edge in self.edge_vars
                ]
                if begin_directed and end_directed:
                    self.solver.add(z3.Not(z3.And(z3.Or(*begin_directed), z3.Or(*end_directed))))

    def satisfiable_with_directed_edges(self, directed_edges: set[tuple[int, int]]) -> bool:
        self.solver.push()
        for edge, var in self.edge_vars.items():
            self.solver.add(var == (edge in directed_edges))
        result = self.solver.check() == z3.sat
        self.solver.pop()
        return result

    def example_assignments(self, *, limit: int = 8) -> list[dict[str, object]]:
        out = []
        self.solver.push()
        while len(out) < limit and self.solver.check() == z3.sat:
            model = self.solver.model()
            edge_values = {
                edge: (
                    bool(model.eval(self.edge_vars[edge], model_completion=True)),
                    bool(model.eval(self.token_vars[edge], model_completion=True)),
                )
                for edge in sorted(self.edge_vars)
            }
            use_values = {
                key: bool(model.eval(var, model_completion=True))
                for key, var in sorted(self.incidence_vars.items())
            }
            flip_values = {
                key: bool(model.eval(var, model_completion=True))
                for key, var in sorted(self.traversal_flip_vars.items())
                if bool(model.eval(var, model_completion=True))
            }
            out.append(
                {
                    "edges": edge_values,
                    "uses": use_values,
                    "traversal_flips": flip_values,
                }
            )
            block = []
            for var in list(self.edge_vars.values()) + list(self.token_vars.values()) + list(
                self.incidence_vars.values()
            ):
                value = model.eval(var, model_completion=True)
                block.append(var != value)
            self.solver.add(z3.Or(*block))
        self.solver.pop()
        return out

    def directed_edge_shapes(self, *, limit: int = 10000) -> tuple[tuple[tuple[int, int], ...], ...]:
        out = []
        self.solver.push()
        while len(out) < limit and self.solver.check() == z3.sat:
            model = self.solver.model()
            shape = tuple(
                edge
                for edge in sorted(self.edge_vars)
                if bool(model.eval(self.edge_vars[edge], model_completion=True))
            )
            out.append(shape)
            self.solver.add(
                z3.Or(
                    *(
                        self.edge_vars[edge] != bool(edge in shape)
                        for edge in sorted(self.edge_vars)
                    )
                )
            )
        self.solver.pop()
        return tuple(out)


def print_components(
    endpoints: tuple[Endpoint, ...],
    incidences: tuple[Incidence, ...],
    bonds: tuple[StereoBond, ...],
    hazards: tuple[HazardBond, ...],
) -> None:
    nodes = []
    graph: dict[str, set[str]] = defaultdict(set)

    def connect(a: str, b: str) -> None:
        graph[a].add(b)
        graph[b].add(a)

    for bond_idx, bond in enumerate(bonds):
        bond_node = f"bond:{bond_idx}:{bond.begin}={bond.end}"
        nodes.append(bond_node)
    for endpoint_idx, endpoint in enumerate(endpoints):
        endpoint_node = f"endpoint:{endpoint_idx}:{endpoint.endpoint}"
        nodes.append(endpoint_node)
        connect(f"bond:{endpoint.stereo_bond_idx}:{bonds[endpoint.stereo_bond_idx].begin}={bonds[endpoint.stereo_bond_idx].end}", endpoint_node)
    for incidence in incidences:
        edge_node = f"edge:{incidence.edge[0]}-{incidence.edge[1]}"
        endpoint_node = f"endpoint:{incidence.endpoint_idx}:{incidence.endpoint}"
        nodes.append(edge_node)
        connect(endpoint_node, edge_node)
    for hazard_idx, hazard in enumerate(hazards):
        hazard_node = f"hazard:{hazard_idx}:{hazard.begin}={hazard.end}"
        nodes.append(hazard_node)
        for edge in hazard.begin_candidate_edges + hazard.end_candidate_edges:
            edge_node = f"edge:{edge[0]}-{edge[1]}"
            nodes.append(edge_node)
            connect(hazard_node, edge_node)

    seen = set()
    components = []
    for node in sorted(set(nodes)):
        if node in seen:
            continue
        stack = [node]
        seen.add(node)
        current = []
        while stack:
            head = stack.pop()
            current.append(head)
            for neighbor in graph[head]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
        components.append(tuple(sorted(current)))
    print("constraint components:", len(components))
    for component in components:
        print("  component:")
        for item in component:
            print("   ", item)


def analyze_case(label: str, smiles: str) -> None:
    source = Chem.MolFromSmiles(smiles)
    assert source is not None
    bonds = stereo_bonds(source)
    endpoints = endpoints_for(source, bonds)
    incidences = incidences_for(endpoints)
    hazards = hazard_bonds_for(source, incidences)
    print("=" * 120)
    print(label)
    print("source:", smiles)
    print("canonical:", Chem.MolToSmiles(source, canonical=True, isomericSmiles=True))
    rdkit_automorphisms = automorphisms(source)
    z3_automorphisms = z3_graph_automorphisms(source)
    print(
        "automorphisms:",
        "rdkit=",
        len(rdkit_automorphisms),
        "z3=",
        len(z3_automorphisms),
        "agree=",
        set(rdkit_automorphisms) == set(z3_automorphisms),
    )
    if len(z3_automorphisms) <= 8:
        for automorphism in z3_automorphisms:
            print("  z3 automorphism:", automorphism)
    print("stereo bonds:", bonds)
    print("endpoints:", endpoints)
    print("incidences:", incidences)
    print("hazards:", hazards)
    print_components(endpoints, incidences, bonds, hazards)

    models = {
        "source-orientation": ConstraintModel(
            endpoints=endpoints,
            incidences=incidences,
            bonds=bonds,
            hazards=hazards,
            allow_traversal_flips=False,
        ),
        "free-traversal-flips": ConstraintModel(
            endpoints=endpoints,
            incidences=incidences,
            bonds=bonds,
            hazards=hazards,
            allow_traversal_flips=True,
        ),
        "source-orientation-strict-hazards": ConstraintModel(
            endpoints=endpoints,
            incidences=incidences,
            bonds=bonds,
            hazards=hazards,
            allow_traversal_flips=False,
            enforce_strict_hazards=True,
        ),
        "observed-edge-shared-pair": ConstraintModel(
            endpoints=endpoints,
            incidences=incidences,
            bonds=bonds,
            hazards=hazards,
            allow_traversal_flips=False,
            directed_edges_observed_by_all_incident_endpoints=True,
            allow_two_candidate_endpoint_only_when_all_edges_shared=True,
        ),
    }
    for name, model in models.items():
        print(f"model {name}: base sat={model.solver.check()}")
        shapes = model.directed_edge_shapes()
        print(f"  projected directed-edge shapes={len(shapes)}")
        for shape in shapes[:8]:
            print(f"    model shape: {shape}")
        for idx, assignment in enumerate(model.example_assignments(limit=2), start=1):
            directed = {
                edge: token
                for edge, (is_directed, token) in assignment["edges"].items()
                if is_directed
            }
            print(f"  example {idx}: directed_edges={directed}")
            if assignment["traversal_flips"]:
                print(f"    traversal flips={assignment['traversal_flips']}")

    if not bonds:
        return

    shape_counts = Counter()
    satisfiable_counts = Counter()
    no_preserving = 0
    for output_smiles in sample_outputs(source):
        output = Chem.MolFromSmiles(output_smiles)
        if output is None:
            no_preserving += 1
            continue
        matches = preserving_matches(source, output, bonds)
        if not matches:
            no_preserving += 1
            continue
        # Count distinct directed-edge sets across preserving automorphisms.
        shapes = {
            frozenset(mapped_directed_edges(output, match).keys())
            for match in matches
        }
        for shape in shapes:
            shape_counts[tuple(sorted(shape))] += 1
        for name, model in models.items():
            if any(model.satisfiable_with_directed_edges(set(shape)) for shape in shapes):
                satisfiable_counts[name] += 1
    print("sample no preserving match:", no_preserving)
    print("sample directed-edge shapes:", len(shape_counts))
    for shape, count in shape_counts.most_common(8):
        print("  shape", count, shape)
    for name in models:
        print(f"sample outputs satisfiable under {name}:", satisfiable_counts[name])
        model_shapes = set(models[name].directed_edge_shapes())
        sample_shapes = set(shape_counts)
        print(
            f"sample shapes under {name}:",
            "sample_subset_of_model=",
            sample_shapes <= model_shapes,
            "model_extra_shapes=",
            len(model_shapes - sample_shapes),
        )
        for shape in sorted(model_shapes - sample_shapes)[:6]:
            print("  model-only shape:", shape)


def main() -> None:
    print("rdkit:", rdBase.rdkitVersion)
    print("legacy stereo perception:", Chem.GetUseLegacyStereoPerception())
    print("z3:", z3.get_version_string())
    for label, smiles in CASES.items():
        analyze_case(label, smiles)


if __name__ == "__main__":
    main()
