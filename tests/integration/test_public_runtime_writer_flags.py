from __future__ import annotations

from dataclasses import dataclass
import unittest

import smiles_next_token
from smiles_next_token._reference.prepared_graph import (
    CONNECTED_NONSTEREO_SURFACE,
    CONNECTED_STEREO_SURFACE,
    prepare_smiles_graph_from_mol_to_smiles_kwargs,
)
from smiles_next_token._reference.rooted_enumerator import (
    enumerate_rooted_connected_nonstereo_smiles_support,
    enumerate_rooted_connected_stereo_smiles_support,
)
from tests.helpers.kernel import CORE_MODULE
from tests.helpers.mols import parse_smiles


@dataclass(frozen=True, slots=True)
class WriterFlagCase:
    name: str
    smiles: str
    rooted_at_atom: int
    isomeric_smiles: bool = True
    kekule_smiles: bool = False
    all_bonds_explicit: bool = False
    all_hs_explicit: bool = False
    ignore_atom_map_numbers: bool = False

    @property
    def surface_kind(self) -> str:
        if self.isomeric_smiles:
            return CONNECTED_STEREO_SURFACE
        return CONNECTED_NONSTEREO_SURFACE


class PublicRuntimeWriterFlagsTests(unittest.TestCase):
    CASES = (
        WriterFlagCase(
            name="nonstereo_baseline",
            smiles="CCO",
            rooted_at_atom=0,
            isomeric_smiles=False,
        ),
        WriterFlagCase(
            name="stereo_baseline",
            smiles="F/C=C\\Cl",
            rooted_at_atom=0,
            isomeric_smiles=True,
        ),
        WriterFlagCase(
            name="kekule_smiles",
            smiles="c1ccncc1",
            rooted_at_atom=0,
            isomeric_smiles=False,
            kekule_smiles=True,
        ),
        WriterFlagCase(
            name="all_bonds_explicit",
            smiles="CC#N",
            rooted_at_atom=0,
            isomeric_smiles=False,
            all_bonds_explicit=True,
        ),
        WriterFlagCase(
            name="all_hs_explicit",
            smiles="C",
            rooted_at_atom=0,
            isomeric_smiles=False,
            all_hs_explicit=True,
        ),
        WriterFlagCase(
            name="ignore_atom_map_numbers",
            smiles="[CH3:7]C",
            rooted_at_atom=0,
            isomeric_smiles=False,
            ignore_atom_map_numbers=True,
        ),
    )

    @classmethod
    def setUpClass(cls) -> None:
        if CORE_MODULE is None:
            raise unittest.SkipTest("private Rust extension is not installed")

    def test_public_runtime_matches_internal_oracle_for_supported_writer_flags(self) -> None:
        for case in self.CASES:
            with self.subTest(case=case.name, smiles=case.smiles):
                from smiles_next_token import _runtime

                mol = parse_smiles(case.smiles)
                reference_prepared = prepare_smiles_graph_from_mol_to_smiles_kwargs(
                    mol,
                    surface_kind=case.surface_kind,
                    isomeric_smiles=case.isomeric_smiles,
                    kekule_smiles=case.kekule_smiles,
                    all_bonds_explicit=case.all_bonds_explicit,
                    all_hs_explicit=case.all_hs_explicit,
                    ignore_atom_map_numbers=case.ignore_atom_map_numbers,
                )
                if case.isomeric_smiles:
                    expected = enumerate_rooted_connected_stereo_smiles_support(
                        reference_prepared,
                        case.rooted_at_atom,
                    )
                else:
                    expected = enumerate_rooted_connected_nonstereo_smiles_support(
                        reference_prepared,
                        case.rooted_at_atom,
                    )

                actual_from_enum = set(
                    smiles_next_token.MolToSmilesEnum(
                        mol,
                        isomericSmiles=case.isomeric_smiles,
                        kekuleSmiles=case.kekule_smiles,
                        rootedAtAtom=case.rooted_at_atom,
                        canonical=False,
                        allBondsExplicit=case.all_bonds_explicit,
                        allHsExplicit=case.all_hs_explicit,
                        doRandom=True,
                        ignoreAtomMapNumbers=case.ignore_atom_map_numbers,
                    )
                )
                actual = _runtime.mol_to_smiles_support(
                    mol,
                    isomeric_smiles=case.isomeric_smiles,
                    kekule_smiles=case.kekule_smiles,
                    rooted_at_atom=case.rooted_at_atom,
                    canonical=False,
                    all_bonds_explicit=case.all_bonds_explicit,
                    all_hs_explicit=case.all_hs_explicit,
                    do_random=True,
                    ignore_atom_map_numbers=case.ignore_atom_map_numbers,
                )

                self.assertEqual(expected, actual)
                self.assertEqual(actual_from_enum, actual)
                for output_smiles in sorted(actual):
                    parsed = parse_smiles(output_smiles)
                    self.assertEqual(
                        reference_prepared.identity_smiles,
                        reference_prepared.identity_smiles_for(parsed),
                    )


if __name__ == "__main__":
    unittest.main()
