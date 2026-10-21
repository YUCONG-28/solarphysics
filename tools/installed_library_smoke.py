"""Exercise a wheel's scientific APIs outside the source checkout."""

from __future__ import annotations

import argparse
import importlib.abc
import runpy
import sys
from pathlib import Path


class NoApplicationImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".", 1)[0].casefold() in {
            "solar_apps",
            "pyqt5",
            "pyqt6",
            "pyside",
            "pyside2",
            "pyside6",
            "flask",
            "streamlit",
        }:
            raise AssertionError(f"Library attempted an application import: {fullname}")
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--examples", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--installed-root",
        type=Path,
        help="Optional pip --target directory to verify without changing the environment.",
    )
    args = parser.parse_args()
    prefix = (
        args.installed_root.resolve()
        if args.installed_root is not None
        else Path(sys.prefix).resolve()
    )
    if args.installed_root is not None:
        sys.path.insert(0, str(prefix))
    sys.meta_path.insert(0, NoApplicationImports())
    import solar_toolkit

    assert Path(solar_toolkit.__file__).resolve().is_relative_to(prefix)
    from solar_toolkit.radio.muser_comparison import nearest_index

    assert nearest_index([0.0, 2.0, 4.0], 2.1, 0.5) == 1
    examples = (
        ("public_api/time_matching_example.py", []),
        ("public_api/sequence_matching_example.py", []),
        ("public_api/gaussian_model_example.py", []),
        ("public_api/native_radio_fit_example.py", []),
        ("radio/synthetic_radio.py", ["--output-dir", str(args.output_dir / "radio")]),
        ("sxr/sxr_example.py", ["--output-dir", str(args.output_dir / "sxr")]),
    )
    for relative, argv in examples:
        module = runpy.run_path(str(args.examples / relative))
        assert module["main"](argv) == 0, relative
    for name, module in tuple(sys.modules.items()):
        if name == "solar_toolkit" or name.startswith("solar_toolkit."):
            assert Path(module.__file__).resolve().is_relative_to(prefix), name
    print(
        f"Installed library APIs and synthetic compositions passed: {solar_toolkit.__version__}"
    )


if __name__ == "__main__":
    main()
