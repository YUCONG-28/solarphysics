"""Build, inspect and rebuild distributions without publishing them."""

from __future__ import annotations

import argparse
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path
from zipfile import ZipFile

from public_source_policy import check_archive


def build(project: Path, output: Path) -> None:
    project = project.resolve()
    output = output.resolve()
    repo = Path(__file__).resolve().parents[1]
    if output.is_relative_to(repo) and not output.is_relative_to(repo / "Local"):
        raise ValueError("Build output must be outside public source or under Local")
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError("Use an empty distribution output directory")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "build",
            "--no-isolation",
            "--sdist",
            "--wheel",
            "--outdir",
            str(output),
            str(project),
        ],
        check=True,
    )
    archives = list(output.glob("*.whl")) + list(output.glob("*.tar.gz"))
    if len(archives) != 2:
        raise ValueError("Expected exactly one wheel and one source distribution")
    for archive in archives:
        errors = check_archive(archive, project.name)
        if errors:
            raise ValueError(f"{archive.name}: {errors}")
    source = next(output.glob("*.tar.gz"))
    with tempfile.TemporaryDirectory(prefix="solarphysics-sdist-") as temporary:
        root = Path(temporary)
        with tarfile.open(source) as bundle:
            # check_archive already rejected links, traversal and special files.
            for member in bundle.getmembers():
                destination = root / member.name
                if not destination.resolve().is_relative_to(root.resolve()):
                    raise ValueError(
                        "Source archive destination escapes temporary root"
                    )
                if member.isfile():
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_bytes(bundle.extractfile(member).read())
        extracted = next(root.iterdir())
        rebuilt = output / "rebuilt"
        subprocess.run(
            [
                sys.executable,
                "-m",
                "build",
                "--no-isolation",
                "--wheel",
                "--outdir",
                str(rebuilt),
                str(extracted),
            ],
            cwd=root,
            check=True,
        )
        wheels = list(rebuilt.glob("*.whl"))
        if len(wheels) != 1:
            raise ValueError("Expected exactly one rebuilt wheel")
        errors = check_archive(wheels[0], project.name)
        if errors:
            raise ValueError(f"Rebuilt wheel: {errors}")
        original = next(output.glob("*.whl"))
        prefix = "solar_apps/" if project.name == "Apps" else "solar_toolkit/"

        def payload(path):
            with ZipFile(path) as archive:
                return {
                    name: archive.read(name)
                    for name in archive.namelist()
                    if (name.startswith(prefix) or ".dist-info/licenses/" in name)
                    and not name.endswith("/")
                }

        if payload(original) != payload(wheels[0]):
            raise ValueError("Rebuilt wheel package content differs from direct build")
    print(f"Validated wheel, source distribution and rebuilt wheel: {project.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()
    build(arguments.project, arguments.output_dir)
