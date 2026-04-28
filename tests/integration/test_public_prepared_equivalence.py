from __future__ import annotations

from dataclasses import dataclass
import unittest

from tests.helpers.kernel import CORE_MODULE
from tests.helpers.mols import parse_smiles
from tests.helpers.public_runtime import (
    choice_texts,
    make_decoder,
    make_determinized_decoder,
    prepared_input_variants,
    public_enum_support,
    public_token_inventory,
    reachable_outputs_from_decoder,
    supported_public_kwargs,
)


@dataclass(frozen=True, slots=True)
class PreparedEquivalenceCase:
    name: str
    smiles: str
    rooted_at_atom: int = 0
    isomeric_smiles: bool = True
    kekule_smiles: bool = False
    all_bonds_explicit: bool = False
    all_hs_explicit: bool = False
    ignore_atom_map_numbers: bool = False

    def runtime_kwargs(self) -> dict[str, object]:
        return supported_public_kwargs(
            rootedAtAtom=self.rooted_at_atom,
            isomericSmiles=self.isomeric_smiles,
            kekuleSmiles=self.kekule_smiles,
            allBondsExplicit=self.all_bonds_explicit,
            allHsExplicit=self.all_hs_explicit,
            ignoreAtomMapNumbers=self.ignore_atom_map_numbers,
        )


class PublicPreparedEquivalenceTests(unittest.TestCase):
    CASES = (
        PreparedEquivalenceCase(
            name="connected_nonstereo",
            smiles="CCO",
            rooted_at_atom=0,
            isomeric_smiles=False,
        ),
        PreparedEquivalenceCase(
            name="connected_all_roots",
            smiles="CCO",
            rooted_at_atom=-1,
            isomeric_smiles=False,
        ),
        PreparedEquivalenceCase(
            name="connected_stereo",
            smiles="F/C=C\\Cl",
            rooted_at_atom=0,
            isomeric_smiles=True,
        ),
        PreparedEquivalenceCase(
            name="kekule_surface",
            smiles="c1ccncc1",
            rooted_at_atom=0,
            isomeric_smiles=False,
            kekule_smiles=True,
        ),
        PreparedEquivalenceCase(
            name="all_bonds_explicit",
            smiles="CC#N",
            rooted_at_atom=0,
            isomeric_smiles=False,
            all_bonds_explicit=True,
        ),
        PreparedEquivalenceCase(
            name="all_hs_explicit",
            smiles="C",
            rooted_at_atom=0,
            isomeric_smiles=False,
            all_hs_explicit=True,
        ),
        PreparedEquivalenceCase(
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

    def test_enum_support_matches_across_mol_and_prepared_inputs(self) -> None:
        for case in self.CASES:
            mol = parse_smiles(case.smiles)
            runtime_kwargs = case.runtime_kwargs()
            variants = prepared_input_variants(mol, **runtime_kwargs)
            expected = public_enum_support(variants[0][1], **runtime_kwargs)

            with self.subTest(case=case.name, smiles=case.smiles):
                for variant_name, variant in variants[1:]:
                    self.assertEqual(
                        expected,
                        public_enum_support(variant, **runtime_kwargs),
                        msg=f"variant={variant_name}",
                    )

    def test_token_inventory_matches_across_mol_and_prepared_inputs(self) -> None:
        for case in self.CASES:
            mol = parse_smiles(case.smiles)
            runtime_kwargs = case.runtime_kwargs()
            variants = prepared_input_variants(mol, **runtime_kwargs)
            expected = public_token_inventory(variants[0][1], **runtime_kwargs)

            with self.subTest(case=case.name, smiles=case.smiles):
                for variant_name, variant in variants[1:]:
                    self.assertEqual(
                        expected,
                        public_token_inventory(variant, **runtime_kwargs),
                        msg=f"variant={variant_name}",
                    )

    def test_decoder_reachable_outputs_match_across_mol_and_prepared_inputs(self) -> None:
        for case in self.CASES:
            mol = parse_smiles(case.smiles)
            runtime_kwargs = case.runtime_kwargs()
            variants = prepared_input_variants(mol, **runtime_kwargs)
            expected_decoder = make_decoder(variants[0][1], **runtime_kwargs)
            expected_prefix = expected_decoder.prefix
            expected_choices = choice_texts(expected_decoder)
            expected_outputs = reachable_outputs_from_decoder(expected_decoder)

            with self.subTest(case=case.name, smiles=case.smiles):
                for variant_name, variant in variants[1:]:
                    decoder = make_decoder(variant, **runtime_kwargs)
                    self.assertEqual(expected_prefix, decoder.prefix, msg=f"variant={variant_name}")
                    self.assertEqual(expected_choices, choice_texts(decoder), msg=f"variant={variant_name}")
                    self.assertEqual(
                        expected_outputs,
                        reachable_outputs_from_decoder(decoder),
                        msg=f"variant={variant_name}",
                    )

    def test_determinized_decoder_reachable_outputs_match_across_mol_and_prepared_inputs(self) -> None:
        for case in self.CASES:
            mol = parse_smiles(case.smiles)
            runtime_kwargs = case.runtime_kwargs()
            variants = prepared_input_variants(mol, **runtime_kwargs)
            expected_decoder = make_determinized_decoder(variants[0][1], **runtime_kwargs)
            expected_prefix = expected_decoder.prefix
            expected_choices = choice_texts(expected_decoder)
            expected_outputs = reachable_outputs_from_decoder(expected_decoder)

            with self.subTest(case=case.name, smiles=case.smiles):
                for variant_name, variant in variants[1:]:
                    decoder = make_determinized_decoder(variant, **runtime_kwargs)
                    self.assertEqual(expected_prefix, decoder.prefix, msg=f"variant={variant_name}")
                    self.assertEqual(expected_choices, choice_texts(decoder), msg=f"variant={variant_name}")
                    self.assertEqual(
                        expected_outputs,
                        reachable_outputs_from_decoder(decoder),
                        msg=f"variant={variant_name}",
                    )


if __name__ == "__main__":
    unittest.main()
