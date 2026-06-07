#!/usr/bin/env python3
"""Compatibility entry point for historical PreparedMol zstd manifests."""

from __future__ import annotations

from collections.abc import Callable
import importlib.util
from pathlib import Path
import sys


def _load_main() -> Callable[[list[str]], int]:
    generator_path = Path(__file__).with_name("prepared_mol_zstd_dictionary_generate.py")
    spec = importlib.util.spec_from_file_location(
        "prepared_mol_zstd_dictionary_generate",
        generator_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load generator script: {generator_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.main


if __name__ == "__main__":
    raise SystemExit(_load_main()(sys.argv[1:]))
