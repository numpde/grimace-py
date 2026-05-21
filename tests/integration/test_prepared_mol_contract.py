from __future__ import annotations

from dataclasses import fields, is_dataclass
import unittest

import grimace
from tests.helpers.mols import parse_smiles


WRITER_FLAG_CASES = (
    (
        "defaults",
        {},
        {
            "isomeric_smiles": True,
            "kekule_smiles": False,
            "all_bonds_explicit": False,
            "all_hs_explicit": False,
            "ignore_atom_map_numbers": False,
        },
    ),
    (
        "nonstereo",
        {
            "isomericSmiles": False,
            "kekuleSmiles": False,
            "allBondsExplicit": False,
            "allHsExplicit": False,
            "ignoreAtomMapNumbers": False,
        },
        {
            "isomeric_smiles": False,
            "kekule_smiles": False,
            "all_bonds_explicit": False,
            "all_hs_explicit": False,
            "ignore_atom_map_numbers": False,
        },
    ),
    (
        "stereo_explicit_surface",
        {
            "isomericSmiles": True,
            "kekuleSmiles": True,
            "allBondsExplicit": True,
            "allHsExplicit": True,
            "ignoreAtomMapNumbers": True,
        },
        {
            "isomeric_smiles": True,
            "kekule_smiles": True,
            "all_bonds_explicit": True,
            "all_hs_explicit": True,
            "ignore_atom_map_numbers": True,
        },
    ),
)

GRAPH_WRITER_FLAG_KEYS = {
    "isomeric_smiles": "writer_do_isomeric_smiles",
    "kekule_smiles": "writer_kekule_smiles",
    "all_bonds_explicit": "writer_all_bonds_explicit",
    "all_hs_explicit": "writer_all_hs_explicit",
    "ignore_atom_map_numbers": "writer_ignore_atom_map_numbers",
}


class PreparedMolContractTests(unittest.TestCase):
    _PREPARED_MOL_ONLY_TESTS = {
        "test_malformed_payloads_fail_clearly",
    }

    def setUp(self) -> None:
        if self._testMethodName == "test_public_api_exposes_prepared_mol_surface":
            return

        if not hasattr(grimace, "PreparedMol"):
            self.skipTest("PreparedMol public API is not implemented yet")
        if self._testMethodName in self._PREPARED_MOL_ONLY_TESTS:
            return
        if not hasattr(grimace, "PrepareMol"):
            self.skipTest("PrepareMol public API is not implemented yet")

    def _prepare(self, smiles: str, **kwargs: object):
        return grimace.PrepareMol(parse_smiles(smiles), **kwargs)

    def _assert_writer_flags(self, prepared: object, expected: dict[str, bool]) -> None:
        writer_flags = prepared.writer_flags
        missing_flags = tuple(name for name in expected if not hasattr(writer_flags, name))
        self.assertEqual((), missing_flags, msg="writer_flags is missing expected flags")
        for flag_name, flag_value in expected.items():
            with self.subTest(flag_name=flag_name):
                self.assertEqual(flag_value, getattr(writer_flags, flag_name))

    def _assert_fragment_graph_matches_atom_indices(self, fragment: object) -> dict[str, object]:
        self.assertIsInstance(fragment.atom_indices, tuple)
        self.assertTrue(callable(fragment.prepared_graph.to_dict))

        graph_dict = fragment.prepared_graph.to_dict()
        self.assertIsInstance(graph_dict, dict)
        self.assertIn("atom_count", graph_dict)
        self.assertIn("atom_atomic_numbers", graph_dict)
        self.assertEqual(len(fragment.atom_indices), graph_dict["atom_count"])
        self.assertEqual(len(graph_dict["atom_atomic_numbers"]), graph_dict["atom_count"])
        return graph_dict

    def _assert_fragment_graph_uses_writer_flags(
        self,
        fragment: object,
        expected: dict[str, bool],
    ) -> None:
        graph_dict = self._assert_fragment_graph_matches_atom_indices(fragment)
        for outer_name, graph_name in GRAPH_WRITER_FLAG_KEYS.items():
            with self.subTest(graph_name=graph_name):
                self.assertEqual(expected[outer_name], graph_dict[graph_name])

    def _assert_prepared_shape_equal(self, prepared: object, restored: object) -> None:
        self.assertIsInstance(restored, grimace.PreparedMol)
        self.assertEqual(prepared.schema_version, restored.schema_version)
        self.assertEqual(prepared.writer_flags, restored.writer_flags)
        self.assertEqual(len(prepared.fragments), len(restored.fragments))

        for original_fragment, restored_fragment in zip(
            prepared.fragments,
            restored.fragments,
            strict=True,
        ):
            self.assertEqual(original_fragment.atom_indices, restored_fragment.atom_indices)
            self.assertEqual(
                original_fragment.prepared_graph.to_dict(),
                restored_fragment.prepared_graph.to_dict(),
            )

    def _assert_round_trip_preserves_shape(self, prepared: object) -> None:
        payload = prepared.to_bytes()
        self.assertIsInstance(payload, bytes)
        self.assertGreater(len(payload), 0)

        restored = grimace.PreparedMol.from_bytes(payload)
        self._assert_prepared_shape_equal(prepared, restored)

    def _assert_no_rdkit_mol_is_stored(self, prepared: object) -> None:
        from rdkit import Chem

        seen: set[int] = set()

        def walk(value: object) -> None:
            object_id = id(value)
            if object_id in seen:
                return
            seen.add(object_id)

            self.assertNotIsInstance(value, Chem.Mol)

            if isinstance(value, dict):
                for item in value.items():
                    walk(item)
                return
            if isinstance(value, (tuple, list, frozenset, set)):
                for item in value:
                    walk(item)
                return
            if is_dataclass(value):
                for field in fields(value):
                    walk(getattr(value, field.name))
                return

            attrs = getattr(value, "__dict__", None)
            if attrs is not None:
                walk(attrs)
                return

            slots = getattr(type(value), "__slots__", ())
            if isinstance(slots, str):
                slots = (slots,)
            for slot in slots:
                if slot.startswith("__") or not hasattr(value, slot):
                    continue
                walk(getattr(value, slot))

    def test_public_api_exposes_prepared_mol_surface(self) -> None:
        required_symbols = ("PreparedMol", "PrepareMol")
        missing_symbols = tuple(
            name for name in required_symbols if not hasattr(grimace, name)
        )
        self.assertEqual((), missing_symbols, msg="missing grimace public symbols")

        noncallable_symbols = tuple(
            name for name in required_symbols if not callable(getattr(grimace, name))
        )
        self.assertEqual((), noncallable_symbols, msg="non-callable grimace public symbols")

        prepared_mol_cls = grimace.PreparedMol
        missing_methods = tuple(
            name
            for name in ("from_bytes",)
            if not callable(getattr(prepared_mol_cls, name, None))
        )
        self.assertEqual((), missing_methods, msg="missing PreparedMol class methods")

    def test_prepare_mol_returns_prepared_mol_with_writer_flags(self) -> None:
        for name, prepare_kwargs, expected_flags in WRITER_FLAG_CASES:
            with self.subTest(name=name):
                prepared = self._prepare("F/C=C\\Cl", **prepare_kwargs)

                self.assertIsInstance(prepared, grimace.PreparedMol)
                self.assertIsInstance(prepared.schema_version, int)
                self.assertGreaterEqual(prepared.schema_version, 1)
                self.assertIsInstance(prepared.fragments, tuple)
                self.assertTrue(callable(prepared.to_bytes))
                self._assert_writer_flags(prepared, expected_flags)
                for fragment in prepared.fragments:
                    self._assert_fragment_graph_uses_writer_flags(fragment, expected_flags)

    def test_connected_molecule_has_one_prepared_fragment(self) -> None:
        prepared = self._prepare("CCO", isomericSmiles=False)

        self.assertEqual(1, len(prepared.fragments))
        self.assertEqual((0, 1, 2), prepared.fragments[0].atom_indices)
        graph_dict = self._assert_fragment_graph_matches_atom_indices(prepared.fragments[0])
        self.assertEqual((6, 6, 8), tuple(graph_dict["atom_atomic_numbers"]))

    def test_disconnected_molecule_preserves_ordered_fragments(self) -> None:
        prepared = self._prepare("CCO.N.Cl", isomericSmiles=False)

        self.assertEqual(3, len(prepared.fragments))
        self.assertEqual(
            ((0, 1, 2), (3,), (4,)),
            tuple(fragment.atom_indices for fragment in prepared.fragments),
        )
        self.assertEqual(
            ((6, 6, 8), (7,), (17,)),
            tuple(
                tuple(
                    self._assert_fragment_graph_matches_atom_indices(fragment)[
                        "atom_atomic_numbers"
                    ]
                )
                for fragment in prepared.fragments
            ),
        )

    def test_writer_surface_controls_atom_map_tokens(self) -> None:
        preserved = self._prepare(
            "[CH3:7]C",
            isomericSmiles=False,
            ignoreAtomMapNumbers=False,
        )
        ignored = self._prepare(
            "[CH3:7]C",
            isomericSmiles=False,
            ignoreAtomMapNumbers=True,
        )

        preserved_graph = preserved.fragments[0].prepared_graph.to_dict()
        ignored_graph = ignored.fragments[0].prepared_graph.to_dict()
        self.assertIn(":7", "".join(preserved_graph["atom_tokens"]))
        self.assertNotIn(":7", "".join(ignored_graph["atom_tokens"]))

    def test_writer_surface_controls_isotope_tokens(self) -> None:
        preserved = self._prepare("[13CH4]", isomericSmiles=True)
        ignored = self._prepare("[13CH4]", isomericSmiles=False)

        preserved_graph = preserved.fragments[0].prepared_graph.to_dict()
        ignored_graph = ignored.fragments[0].prepared_graph.to_dict()
        self.assertIn("13", preserved_graph["atom_tokens"][0])
        self.assertNotIn("13", ignored_graph["atom_tokens"][0])

    def test_runtime_options_are_not_part_of_prepare_mol(self) -> None:
        mol = parse_smiles("CCO")
        for runtime_kwarg in ("rootedAtAtom", "canonical", "doRandom"):
            with self.subTest(runtime_kwarg=runtime_kwarg):
                with self.assertRaises(TypeError):
                    grimace.PrepareMol(mol, **{runtime_kwarg: 0})

        prepared = grimace.PrepareMol(mol, isomericSmiles=False)
        self.assertFalse(hasattr(prepared.writer_flags, "rooted_at_atom"))
        self.assertFalse(hasattr(prepared.writer_flags, "canonical"))
        self.assertFalse(hasattr(prepared.writer_flags, "do_random"))

    def test_prepared_mol_does_not_store_rdkit_mol(self) -> None:
        prepared = self._prepare("CCO.N", isomericSmiles=False)

        self._assert_no_rdkit_mol_is_stored(prepared)

    def test_bytes_round_trip_preserves_prepared_shape(self) -> None:
        for name, prepare_kwargs, _expected_flags in WRITER_FLAG_CASES:
            with self.subTest(name=name):
                prepared = self._prepare("F/C=C\\Cl.N", **prepare_kwargs)

                self._assert_round_trip_preserves_shape(prepared)

    def test_malformed_payloads_fail_clearly(self) -> None:
        malformed_payloads = (
            b"",
            b"not a prepared molecule",
        )

        for payload in malformed_payloads:
            with self.subTest(payload=payload):
                with self.assertRaises(ValueError):
                    grimace.PreparedMol.from_bytes(payload)


if __name__ == "__main__":
    unittest.main()
