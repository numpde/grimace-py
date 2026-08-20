from __future__ import annotations

from dataclasses import dataclass
import unittest

from tests.helpers.kernel import CORE_MODULE
from tests.helpers.mols import parse_smiles
from tests.helpers.public_runtime import (
    make_decoder,
    make_determinized_decoder,
    public_enum_support,
    public_enum_support_union_over_explicit_roots,
    public_token_inventory,
    public_token_inventory_union_over_explicit_roots,
    reachable_outputs_from_decoder,
    supported_public_kwargs,
)


@dataclass(frozen=True, slots=True)
class AllRootsIdentityCase:
    name: str
    smiles: str
    isomeric_smiles: bool = True
    kekule_smiles: bool = False
    all_bonds_explicit: bool = False
    all_hs_explicit: bool = False
    ignore_atom_map_numbers: bool = False

    def runtime_kwargs(self) -> dict[str, object]:
        return supported_public_kwargs(
            isomericSmiles=self.isomeric_smiles,
            kekuleSmiles=self.kekule_smiles,
            allBondsExplicit=self.all_bonds_explicit,
            allHsExplicit=self.all_hs_explicit,
            ignoreAtomMapNumbers=self.ignore_atom_map_numbers,
        )


class PublicAllRootsIdentityTests(unittest.TestCase):
    CASES = (
        AllRootsIdentityCase(
            name="connected_nonstereo",
            smiles="CCO",
            isomeric_smiles=False,
        ),
        AllRootsIdentityCase(
            name="connected_stereo",
            smiles="F/C=C\\Cl",
            isomeric_smiles=True,
        ),
        AllRootsIdentityCase(
            name="kekule_surface",
            smiles="c1ccncc1",
            isomeric_smiles=False,
            kekule_smiles=True,
        ),
        AllRootsIdentityCase(
            name="all_bonds_explicit",
            smiles="CC#N",
            isomeric_smiles=False,
            all_bonds_explicit=True,
        ),
        AllRootsIdentityCase(
            name="all_hs_explicit",
            smiles="C",
            isomeric_smiles=False,
            all_hs_explicit=True,
        ),
        AllRootsIdentityCase(
            name="ignore_atom_map_numbers",
            smiles="[CH3:7]C",
            isomeric_smiles=False,
            ignore_atom_map_numbers=True,
        ),
        AllRootsIdentityCase(
            name="disconnected_nonstereo",
            smiles="[Na+].C#N",
            isomeric_smiles=False,
        ),
        AllRootsIdentityCase(
            name="disconnected_branching",
            smiles="CO.CCO",
            isomeric_smiles=False,
        ),
    )

    @classmethod
    def setUpClass(cls) -> None:
        if CORE_MODULE is None:
            raise unittest.SkipTest("private Rust extension is not installed")

    def test_enum_all_roots_equals_union_over_explicit_roots(self) -> None:
        for case in self.CASES:
            mol = parse_smiles(case.smiles)
            runtime_kwargs = case.runtime_kwargs()
            with self.subTest(case=case.name, smiles=case.smiles):
                self.assertEqual(
                    public_enum_support_union_over_explicit_roots(mol, **runtime_kwargs),
                    public_enum_support(
                        mol,
                        **runtime_kwargs,
                    ),
                )

    def test_inventory_all_roots_equals_union_over_explicit_roots(self) -> None:
        for case in self.CASES:
            mol = parse_smiles(case.smiles)
            runtime_kwargs = case.runtime_kwargs()
            with self.subTest(case=case.name, smiles=case.smiles):
                self.assertEqual(
                    public_token_inventory_union_over_explicit_roots(
                        mol,
                        **runtime_kwargs,
                    ),
                    public_token_inventory(
                        mol,
                        **runtime_kwargs,
                    ),
                )

    def test_decoder_all_roots_reachable_outputs_equal_enum_support(self) -> None:
        for case in self.CASES:
            mol = parse_smiles(case.smiles)
            runtime_kwargs = case.runtime_kwargs()
            with self.subTest(case=case.name, smiles=case.smiles):
                decoder = make_decoder(mol, **runtime_kwargs)
                self.assertEqual(
                    public_enum_support(mol, **runtime_kwargs),
                    reachable_outputs_from_decoder(decoder),
                )

    def test_determinized_decoder_all_roots_reachable_outputs_equal_enum_support(self) -> None:
        for case in self.CASES:
            mol = parse_smiles(case.smiles)
            runtime_kwargs = case.runtime_kwargs()
            with self.subTest(case=case.name, smiles=case.smiles):
                decoder = make_determinized_decoder(mol, **runtime_kwargs)
                self.assertEqual(
                    public_enum_support(mol, **runtime_kwargs),
                    reachable_outputs_from_decoder(decoder),
                )


if __name__ == "__main__":
    unittest.main()
