from __future__ import annotations

from dataclasses import fields
import inspect
import unittest

import grimace
from grimace._mol_to_smiles_options import (
    MOL_TO_SMILES_OPTIONS,
    MOL_TO_SMILES_PREPARED_OPTIONS,
    MOL_TO_SMILES_PUBLIC_OPTION_NAMES,
    coerce_internal_options,
    coerce_public_options,
    coerce_required_public_options,
    internal_option_values,
    public_options_from_internal_options,
    public_option_values,
)
from grimace._runtime_inputs import MolToSmilesFlags
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
        self.assertEqual(set(public_names), MOL_TO_SMILES_PUBLIC_OPTION_NAMES)

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

    def test_option_value_projection_keeps_only_inventory_names(self) -> None:
        values = {
            "isomericSmiles": None,
            "isomeric_smiles": None,
            "seed": 0,
        }

        self.assertEqual(
            {"isomericSmiles": None},
            public_option_values(MOL_TO_SMILES_OPTIONS, values),
        )
        self.assertEqual(
            {"isomeric_smiles": None},
            internal_option_values(MOL_TO_SMILES_OPTIONS, values),
        )

    def test_public_sample_signature_uses_option_inventory_suffix(self) -> None:
        sample_defaults = _keyword_only_signature_defaults(grimace.MolToSmilesSample)

        self.assertEqual(
            (
                ("seed", inspect.Parameter.empty),
                ("decoder_view", "determinized"),
                ("sampling_mode", "uniform_token"),
            ),
            sample_defaults[:3],
        )
        self.assertEqual(
            _public_option_defaults(MOL_TO_SMILES_OPTIONS),
            sample_defaults[3:],
        )

    def test_prepare_mol_signature_matches_prepared_option_inventory(self) -> None:
        self.assertEqual(
            _public_option_defaults(MOL_TO_SMILES_PREPARED_OPTIONS),
            _keyword_only_signature_defaults(grimace.PrepareMol),
        )

    def test_runtime_flags_match_internal_option_inventory(self) -> None:
        self.assertEqual(
            _internal_option_defaults(MOL_TO_SMILES_OPTIONS),
            tuple((field.name, field.default) for field in fields(MolToSmilesFlags)),
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

    def test_optional_option_parsers_reject_unknown_names(self) -> None:
        cases = (
            (coerce_public_options, {"seed": 0}),
            (coerce_internal_options, {"seed": 0}),
            (public_options_from_internal_options, {"seed": 0}),
        )

        for parser, values in cases:
            with self.subTest(parser=parser.__name__):
                with self.assertRaisesRegex(TypeError, "unknown option"):
                    parser(
                        MOL_TO_SMILES_OPTIONS,
                        values,
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

    def test_required_public_option_parser_preserves_public_names_without_defaults(self) -> None:
        values = {spec.public_name: spec.default for spec in MOL_TO_SMILES_OPTIONS}
        values["isomericSmiles"] = 0

        self.assertEqual(
            {
                "isomericSmiles": False,
                "kekuleSmiles": False,
                "rootedAtAtom": -1,
                "canonical": True,
                "allBondsExplicit": False,
                "allHsExplicit": False,
                "doRandom": False,
                "ignoreAtomMapNumbers": False,
            },
            coerce_required_public_options(
                MOL_TO_SMILES_OPTIONS,
                values,
                context="TestContext",
            ),
        )

    def test_required_public_option_parser_rejects_missing_options(self) -> None:
        values = {spec.public_name: spec.default for spec in MOL_TO_SMILES_OPTIONS}
        del values["doRandom"]

        with self.assertRaisesRegex(ValueError, "TestContext requires doRandom"):
            coerce_required_public_options(
                MOL_TO_SMILES_OPTIONS,
                values,
                context="TestContext",
            )

    def test_internal_option_formatter_maps_to_public_names(self) -> None:
        self.assertEqual(
            {
                "isomericSmiles": False,
                "kekuleSmiles": False,
                "rootedAtAtom": -1,
                "canonical": True,
                "allBondsExplicit": False,
                "allHsExplicit": True,
                "doRandom": False,
                "ignoreAtomMapNumbers": False,
            },
            public_options_from_internal_options(
                MOL_TO_SMILES_OPTIONS,
                {
                    "isomeric_smiles": None,
                    "all_hs_explicit": 1,
                },
                context="TestContext",
            ),
        )

    def test_prepared_mol_matches_named_prepared_options(self) -> None:
        prepared_kwargs = {
            spec.public_name: index % 2 == 0
            for index, spec in enumerate(MOL_TO_SMILES_PREPARED_OPTIONS)
        }
        prepared = grimace.PrepareMol(parse_smiles("CCO"), **prepared_kwargs)
        runtime_kwargs = {
            "rootedAtAtom": 0,
            "canonical": False,
            "doRandom": True,
            **prepared_kwargs,
        }

        self.assertGreater(len(tuple(grimace.MolToSmilesEnum(prepared, **runtime_kwargs))), 0)

        for spec in MOL_TO_SMILES_PREPARED_OPTIONS:
            mismatched_kwargs = {
                **runtime_kwargs,
                spec.public_name: not prepared_kwargs[spec.public_name],
            }
            with self.subTest(public_name=spec.public_name):
                with self.assertRaisesRegex(ValueError, "writer flags"):
                    tuple(grimace.MolToSmilesEnum(prepared, **mismatched_kwargs))


if __name__ == "__main__":
    unittest.main()
