from __future__ import annotations

from dataclasses import fields, is_dataclass
import unittest

import grimace
from tests.helpers.mols import parse_smiles
from tests.helpers.public_runtime import (
    public_enum_support,
    public_token_inventory,
    supported_public_kwargs,
)


WRITER_KWARG_NAMES = (
    "isomericSmiles",
    "kekuleSmiles",
    "allBondsExplicit",
    "allHsExplicit",
    "ignoreAtomMapNumbers",
)


class PreparedMolContractTests(unittest.TestCase):
    def _prepare_kwargs(self, kwargs: dict[str, object]) -> dict[str, object]:
        return {name: kwargs[name] for name in WRITER_KWARG_NAMES if name in kwargs}

    def _prepare(self, smiles: str, **kwargs: object) -> object:
        return grimace.PrepareMol(parse_smiles(smiles), **kwargs)

    def _assert_public_runtime_equivalent(
        self,
        smiles: str,
        *,
        kwargs: dict[str, object],
    ) -> None:
        mol = parse_smiles(smiles)
        prepared = grimace.PrepareMol(mol, **self._prepare_kwargs(kwargs))
        payload = prepared.to_bytes()
        restored = grimace.PreparedMol.from_bytes(payload)

        self.assertIsInstance(prepared, grimace.PreparedMol)
        self.assertIsInstance(payload, bytes)
        self.assertGreater(len(payload), 0)
        self.assertIsInstance(restored, grimace.PreparedMol)
        self.assertEqual(
            public_enum_support(mol, **kwargs),
            public_enum_support(prepared, **kwargs),
        )
        self.assertEqual(
            public_enum_support(mol, **kwargs),
            public_enum_support(restored, **kwargs),
        )
        self.assertEqual(
            public_token_inventory(mol, **kwargs),
            public_token_inventory(prepared, **kwargs),
        )
        self.assertEqual(
            public_token_inventory(mol, **kwargs),
            public_token_inventory(restored, **kwargs),
        )

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

        walk(prepared)

    def test_public_api_exposes_only_preparation_and_bytes_surface(self) -> None:
        required_symbols = ("PreparedMol", "PrepareMol")
        missing_symbols = tuple(
            name for name in required_symbols if not hasattr(grimace, name)
        )
        self.assertEqual((), missing_symbols, msg="missing grimace public symbols")

        prepared = self._prepare("CCO", isomericSmiles=False)
        self.assertIsInstance(prepared, grimace.PreparedMol)
        self.assertTrue(callable(prepared.to_bytes))
        self.assertTrue(callable(getattr(grimace.PreparedMol, "from_bytes", None)))
        self.assertFalse(hasattr(prepared, "__dict__"))

        for leaked_name in ("schema_version", "writer_flags", "fragments"):
            with self.subTest(leaked_name=leaked_name):
                self.assertFalse(hasattr(prepared, leaked_name))

        with self.assertRaises(TypeError):
            grimace.PreparedMol()
        with self.assertRaises(AttributeError):
            prepared.extra = object()

    def test_bytes_round_trip_preserves_public_runtime_behavior(self) -> None:
        cases = (
            (
                "connected_nonstereo",
                "CCO",
                supported_public_kwargs(isomericSmiles=False, rootedAtAtom=0),
            ),
            (
                "disconnected_nonstereo",
                "CCO.N.Cl",
                supported_public_kwargs(isomericSmiles=False, rootedAtAtom=3),
            ),
            (
                "stereo",
                "F[C@H](Cl)Br",
                supported_public_kwargs(isomericSmiles=True, rootedAtAtom=1),
            ),
            (
                "writer_surface",
                "c1ccccc1",
                supported_public_kwargs(
                    isomericSmiles=False,
                    kekuleSmiles=True,
                    rootedAtAtom=0,
                ),
            ),
        )

        for name, smiles, kwargs in cases:
            with self.subTest(name=name):
                self._assert_public_runtime_equivalent(smiles, kwargs=kwargs)

    def test_writer_surface_controls_atom_map_tokens(self) -> None:
        preserved_kwargs = supported_public_kwargs(
            isomericSmiles=False,
            ignoreAtomMapNumbers=False,
            rootedAtAtom=0,
        )
        ignored_kwargs = supported_public_kwargs(
            isomericSmiles=False,
            ignoreAtomMapNumbers=True,
            rootedAtAtom=0,
        )

        preserved = self._prepare("[CH3:7]C", **self._prepare_kwargs(preserved_kwargs))
        ignored = self._prepare("[CH3:7]C", **self._prepare_kwargs(ignored_kwargs))

        self.assertIn(":7", "".join(public_token_inventory(preserved, **preserved_kwargs)))
        self.assertNotIn(":7", "".join(public_token_inventory(ignored, **ignored_kwargs)))

    def test_writer_surface_controls_isotope_tokens(self) -> None:
        preserved_kwargs = supported_public_kwargs(isomericSmiles=True, rootedAtAtom=0)
        ignored_kwargs = supported_public_kwargs(isomericSmiles=False, rootedAtAtom=0)

        preserved = self._prepare("[13CH4]", **self._prepare_kwargs(preserved_kwargs))
        ignored = self._prepare("[13CH4]", **self._prepare_kwargs(ignored_kwargs))

        self.assertIn("13", "".join(public_token_inventory(preserved, **preserved_kwargs)))
        self.assertNotIn("13", "".join(public_token_inventory(ignored, **ignored_kwargs)))

    def test_runtime_options_are_not_part_of_prepare_mol(self) -> None:
        mol = parse_smiles("CCO")
        for runtime_kwarg in ("rootedAtAtom", "canonical", "doRandom"):
            with self.subTest(runtime_kwarg=runtime_kwarg):
                with self.assertRaises(TypeError):
                    grimace.PrepareMol(mol, **{runtime_kwarg: 0})

    def test_prepare_mol_coerces_integral_writer_flags_like_rdkit(self) -> None:
        flag_cases = (
            ("isomericSmiles", 0, False),
            ("kekuleSmiles", 1, True),
            ("allBondsExplicit", None, False),
            ("allHsExplicit", 1, True),
            ("ignoreAtomMapNumbers", 0, False),
        )

        mol = parse_smiles("CCO")
        for flag_name, provided_value, expected_value in flag_cases:
            with self.subTest(flag_name=flag_name, provided_value=provided_value):
                prepared = grimace.PrepareMol(mol, **{flag_name: provided_value})
                kwargs = supported_public_kwargs(rootedAtAtom=0, **{flag_name: expected_value})
                self.assertEqual(
                    public_enum_support(mol, **kwargs),
                    public_enum_support(prepared, **kwargs),
                )

    def test_prepare_mol_rejects_non_integral_writer_flags(self) -> None:
        invalid_cases = (
            ("isomericSmiles", 0.0),
            ("kekuleSmiles", "false"),
            ("allBondsExplicit", 1.0),
            ("allHsExplicit", "true"),
            ("ignoreAtomMapNumbers", 1.0),
        )

        for flag_name, invalid_value in invalid_cases:
            with self.subTest(flag_name=flag_name, invalid_value=invalid_value):
                with self.assertRaisesRegex(TypeError, flag_name):
                    self._prepare("CCO", **{flag_name: invalid_value})

    def test_prepared_mol_does_not_store_rdkit_mol(self) -> None:
        self._assert_no_rdkit_mol_is_stored(self._prepare("CCO.N", isomericSmiles=False))

    def test_from_bytes_rejects_non_bytes_and_malformed_payloads(self) -> None:
        with self.assertRaises(TypeError):
            grimace.PreparedMol.from_bytes(bytearray())

        for payload in (b"", b"not a prepared molecule"):
            with self.subTest(payload=payload):
                with self.assertRaises(ValueError):
                    grimace.PreparedMol.from_bytes(payload)


if __name__ == "__main__":
    unittest.main()
