from __future__ import annotations

from dataclasses import fields
import inspect
import unittest

import grimace
from grimace import _runtime
from grimace._mol_to_smiles_options import (
    MOL_TO_SMILES_OPTIONS,
    MOL_TO_SMILES_PREPARED_OPTIONS,
    coerce_public_options,
)
from grimace._prepared_mol import _prepared_mol_writer_flag_values
from tests.helpers.mols import parse_smiles


def _keyword_only_signature_defaults(callable_object: object) -> tuple[tuple[str, object], ...]:
    return tuple(
        (name, parameter.default)
        for name, parameter in inspect.signature(callable_object).parameters.items()
        if parameter.kind is inspect.Parameter.KEYWORD_ONLY
    )


def _public_option_defaults(specs: tuple[object, ...]) -> tuple[tuple[str, object], ...]:
    return tuple((spec.public_name, spec.default) for spec in specs)


def _internal_option_defaults(specs: tuple[object, ...]) -> tuple[tuple[str, object], ...]:
    return tuple((spec.internal_name, spec.default) for spec in specs)


class MolToSmilesOptionInventoryTests(unittest.TestCase):
    def test_option_names_are_unique(self) -> None:
        public_names = tuple(spec.public_name for spec in MOL_TO_SMILES_OPTIONS)
        internal_names = tuple(spec.internal_name for spec in MOL_TO_SMILES_OPTIONS)

        self.assertEqual(len(public_names), len(set(public_names)))
        self.assertEqual(len(internal_names), len(set(internal_names)))

    def test_public_runtime_signatures_match_option_inventory(self) -> None:
        expected = _public_option_defaults(MOL_TO_SMILES_OPTIONS)

        for callable_object in (
            grimace.MolToSmilesEnum,
            grimace.MolToSmilesTokenInventory,
            grimace.MolToSmilesTokenInventorySuperset,
            grimace.MolToSmilesDeviation,
            grimace.MolToSmilesDecoder.__init__,
            grimace.MolToSmilesDeterminizedDecoder.__init__,
        ):
            with self.subTest(callable_object=callable_object):
                self.assertEqual(expected, _keyword_only_signature_defaults(callable_object))

    def test_prepare_mol_signature_matches_prepared_option_inventory(self) -> None:
        self.assertEqual(
            _public_option_defaults(MOL_TO_SMILES_PREPARED_OPTIONS),
            _keyword_only_signature_defaults(grimace.PrepareMol),
        )

    def test_runtime_flags_match_internal_option_inventory(self) -> None:
        self.assertEqual(
            _internal_option_defaults(MOL_TO_SMILES_OPTIONS),
            tuple((field.name, field.default) for field in fields(_runtime.MolToSmilesFlags)),
        )

    def test_prepared_and_call_options_partition_full_inventory(self) -> None:
        self.assertEqual(
            tuple(spec for spec in MOL_TO_SMILES_OPTIONS if spec.scope == "prepared"),
            MOL_TO_SMILES_PREPARED_OPTIONS,
        )
        self.assertEqual(
            {"prepared", "call"},
            {spec.scope for spec in MOL_TO_SMILES_OPTIONS},
        )

    def test_public_option_parser_uses_spec_names_defaults_and_value_rules(self) -> None:
        self.assertEqual(
            {
                "isomeric_smiles": False,
                "kekule_smiles": True,
                "rooted_at_atom": 1,
                "canonical": True,
                "all_bonds_explicit": False,
                "all_hs_explicit": False,
                "do_random": True,
                "ignore_atom_map_numbers": False,
            },
            coerce_public_options(
                MOL_TO_SMILES_OPTIONS,
                {
                    "isomericSmiles": None,
                    "kekuleSmiles": 1,
                    "rootedAtAtom": True,
                    "doRandom": 1,
                },
                context="TestContext",
            ),
        )

    def test_public_option_parser_rejects_bad_values_by_rule(self) -> None:
        bad_cases = (
            ("isomericSmiles", "false", "bool, int, or None"),
            ("rootedAtAtom", None, "integer"),
        )

        for public_name, value, expected_regex in bad_cases:
            with self.subTest(public_name=public_name):
                with self.assertRaisesRegex(TypeError, expected_regex):
                    coerce_public_options(
                        MOL_TO_SMILES_OPTIONS,
                        {public_name: value},
                        context="TestContext",
                    )

    def test_public_option_parser_maps_to_internal_names(self) -> None:
        self.assertEqual(
            {
                "isomeric_smiles": False,
                "kekule_smiles": False,
                "rooted_at_atom": -1,
                "canonical": True,
                "all_bonds_explicit": False,
                "all_hs_explicit": False,
                "do_random": False,
                "ignore_atom_map_numbers": False,
            },
            coerce_public_options(
                MOL_TO_SMILES_OPTIONS,
                {"isomericSmiles": False},
                context="TestContext",
            ),
        )

    def test_prepared_mol_writer_tuple_order_matches_prepared_options(self) -> None:
        prepared_kwargs = {
            spec.public_name: index % 2 == 0
            for index, spec in enumerate(MOL_TO_SMILES_PREPARED_OPTIONS)
        }
        prepared = grimace.PrepareMol(parse_smiles("c1ccccc1"), **prepared_kwargs)

        self.assertEqual(
            tuple(
                bool(prepared_kwargs[spec.public_name])
                for spec in MOL_TO_SMILES_PREPARED_OPTIONS
            ),
            _prepared_mol_writer_flag_values(prepared),
        )


if __name__ == "__main__":
    unittest.main()
