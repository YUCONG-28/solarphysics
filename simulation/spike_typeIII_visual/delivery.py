"""Build and re-verify one portable ZIP64 scientific delivery package."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .package_verify import verify

SIMULATION_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_DELIVERABLES = SIMULATION_ROOT / "deliverables"

STORE_SUFFIXES = {
    ".h5",
    ".hdf5",
    ".npz",
    ".mp4",
    ".gif",
    ".png",
    ".jpg",
    ".jpeg",
    ".pptx",
    ".zip",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def _copy_tree(
    source: Path,
    target: Path,
    *,
    exclude,
) -> None:
    for path in source.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(source)
        if exclude(relative, path):
            continue
        _copy(path, target / relative)


def _source_exclude(relative: Path, path: Path) -> bool:
    parts = set(relative.parts)
    return (
        "__pycache__" in parts
        or ".pytest_cache" in parts
        or "outputs" in parts
        or path.suffix in {".pyc", ".pyo"}
    )


def _event_exclude(relative: Path, path: Path) -> bool:
    parts = set(relative.parts)
    return (
        path.suffix.lower() == ".mkv"
        or path.name.startswith(".")
        or "__pycache__" in parts
        or path.suffix in {".pyc", ".pyo"}
    )


def _control_exclude(relative: Path, path: Path) -> bool:
    return (
        path.suffix.lower() == ".mkv"
        or relative.parts[0] in {"animations", "figures"}
        or path.name.startswith(".")
    )


def _baseline_exclude(relative: Path, path: Path) -> bool:
    return (
        path.suffix.lower() in {".h5", ".mkv", ".mp4", ".gif", ".png", ".pdf", ".svg"}
        or path.name == "mhd_snapshots.npz"
    )


def _write_environment(stage: Path) -> None:
    environment = stage / "environment"
    environment.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [os.sys.executable, "-m", "pip", "freeze"],
        text=True,
        capture_output=True,
        check=False,
    )
    (environment / "pip_freeze.txt").write_text(
        result.stdout,
        encoding="utf-8",
        newline="\n",
    )
    for path in sorted(SIMULATION_ROOT.glob("environment*.yml")):
        _copy(path, environment / path.name)


def _role(relative: Path) -> str:
    first = relative.parts[0]
    return {
        "source": "reproducible source and tests",
        "runs": "authoritative event/control scientific data",
        "media": "delivery media",
        "presentation": "final presentation and reconstruction inputs",
        "environment": "runtime and dependency evidence",
        "baseline": "historical t=2 diagnostic summary",
        "documentation": "project documentation",
    }.get(first, "verification or package metadata")


def _write_manifests(
    stage: Path,
    *,
    event_run: Path,
    control_run: Path,
) -> tuple[Path, Path]:
    candidates = sorted(
        path
        for path in stage.rglob("*")
        if path.is_file()
        and path.name not in {"manifest.json", "SHA256SUMS.txt"}
    )
    manifest = {
        "schema": "spike-typeiii-delivery-v1",
        "generated_utc": datetime.now(UTC).isoformat(),
        "event_run_id": event_run.name,
        "control_run_id": control_run.name,
        "package_tier": "research-complete",
        "lossless_master_policy": (
            "FFV1 masters are reproducible from float64 HDF5 and are excluded "
            "to keep the single transfer package practical."
        ),
        "file_count": len(candidates) + 1,
        "files": [
            {
                "path": path.relative_to(stage).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
                "role": _role(path.relative_to(stage)),
            }
            for path in candidates
        ],
    }
    manifest_path = stage / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    checksum_candidates = sorted([*candidates, manifest_path])
    checksums = stage / "SHA256SUMS.txt"
    checksums.write_text(
        "\n".join(
            f"{_sha256(path)}  {path.relative_to(stage).as_posix()}"
            for path in checksum_candidates
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return manifest_path, checksums


def _stage_package(
    stage: Path,
    event_run: Path,
    control_run: Path,
) -> None:
    _copy_tree(PACKAGE_ROOT, stage / "source" / "spike_typeIII_visual", exclude=_source_exclude)
    for name in ("README.md", "PLAN.md"):
        path = SIMULATION_ROOT / name
        if path.is_file():
            _copy(path, stage / "documentation" / name)
    launcher = SIMULATION_ROOT / "run_scientific_production_wsl.sh"
    if launcher.is_file():
        _copy(launcher, stage / "source" / launcher.name)
    ppt = SIMULATION_ROOT / "Spike_Topping_TypeIII_simulation.pptx"
    if ppt.is_file():
        _copy(ppt, stage / "presentation" / ppt.name)
    presentation_workspace = SIMULATION_ROOT / "presentation"
    if presentation_workspace.is_dir():
        _copy_tree(
            presentation_workspace,
            stage / "presentation" / "workspace",
            exclude=lambda relative, path: (
                "build" in relative.parts
                or "__pycache__" in relative.parts
                or path.suffix in {".pyc", ".png", ".jpg", ".jpeg"}
            ),
        )
    _copy_tree(
        event_run,
        stage / "runs" / "event",
        exclude=_event_exclude,
    )
    _copy_tree(
        control_run,
        stage / "runs" / "control",
        exclude=_control_exclude,
    )
    baseline = (
        PACKAGE_ROOT
        / "outputs"
        / "runs"
        / "cuda-fine_seed20260726"
    )
    if baseline.is_dir():
        _copy_tree(
            baseline / "data",
            stage / "baseline" / "cuda-fine-t2" / "data",
            exclude=_baseline_exclude,
        )
    _write_environment(stage)
    verifier = PACKAGE_ROOT / "package_verify.py"
    _copy(verifier, stage / "verify_package.py")
    (stage / "verify.ps1").write_text(
        "python .\\verify_package.py .\n",
        encoding="utf-8",
        newline="\n",
    )
    (stage / "verify.sh").write_text(
        "#!/usr/bin/env sh\npython3 ./verify_package.py .\n",
        encoding="utf-8",
        newline="\n",
    )
    reproduce = stage / "REPRODUCE.md"
    reproduce.write_text(
        "# Reproduce\n\n"
        "Activate the WSL `torch-cuda` environment, then run:\n\n"
        "```bash\n"
        "python -m spike_typeIII_visual.production "
        "--seed 20260726 --target scientific-4k --resume\n"
        "```\n\n"
        "FFV1 masters can be regenerated from the packaged float64 HDF5 "
        "using the scientific render profile; they are intentionally not "
        "duplicated in this transfer archive.\n",
        encoding="utf-8",
        newline="\n",
    )


def _zip(stage: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".partial")
    if temporary.exists():
        temporary.unlink()
    with zipfile.ZipFile(
        temporary,
        "w",
        allowZip64=True,
    ) as archive:
        for path in sorted(stage.rglob("*")):
            if not path.is_file():
                continue
            compression = (
                zipfile.ZIP_STORED
                if path.suffix.lower() in STORE_SUFFIXES
                else zipfile.ZIP_DEFLATED
            )
            archive.write(
                path,
                path.relative_to(stage).as_posix(),
                compress_type=compression,
                compresslevel=None if compression == zipfile.ZIP_STORED else 6,
            )
    os.replace(temporary, output)


def _verify_archive(path: Path) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="spike_typeiii_package_verify_") as tmp:
        extract = Path(tmp)
        with zipfile.ZipFile(path) as archive:
            bad = archive.testzip()
            if bad is not None:
                raise RuntimeError(f"ZIP CRC failed: {bad}")
            archive.extractall(extract)
        report = verify(extract)
        if not report["passed"]:
            raise RuntimeError("Package checksum verification failed.")
        event_h5 = extract / "runs" / "event" / "data" / "rmhd_fields.h5"
        control_h5 = extract / "runs" / "control" / "data" / "rmhd_fields.h5"
        if not event_h5.is_file() or not control_h5.is_file():
            raise RuntimeError("Package lacks authoritative event/control HDF5.")
        try:
            import h5py

            with h5py.File(event_h5, "r") as handle:
                event_snapshots = int(handle["state/time"].shape[0])
            with h5py.File(control_h5, "r") as handle:
                control_snapshots = int(handle["state/time"].shape[0])
        except (ImportError, OSError, KeyError) as exc:
            raise RuntimeError(f"Packaged HDF5 cannot be opened: {exc}") from exc
        media_root = extract / "runs" / "event" / "animations"
        video_reports: dict[str, dict[str, Any]] = {}
        for stem in (
            "causal_chain",
            "reconnection_topology",
            "bidirectional_outflow",
            "radio_event_control",
        ):
            video = media_root / f"{stem}.mp4"
            if not video.is_file():
                raise RuntimeError(f"Package lacks required video: {video.name}")
            probe = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-count_frames",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "stream=width,height,avg_frame_rate,nb_read_frames",
                    "-of",
                    "json",
                    str(video),
                ],
                text=True,
                capture_output=True,
                check=False,
            )
            if probe.returncode != 0:
                raise RuntimeError(
                    f"Packaged video cannot be decoded: {video.name}: "
                    f"{probe.stderr.strip()}"
                )
            stream = json.loads(probe.stdout)["streams"][0]
            width = int(stream["width"])
            height = int(stream["height"])
            numerator, denominator = (
                int(value) for value in stream["avg_frame_rate"].split("/")
            )
            fps = numerator / denominator
            frames = int(stream["nb_read_frames"])
            if (width, height) != (3840, 2160) or abs(fps - 30.0) > 1e-6:
                raise RuntimeError(
                    f"Packaged video format mismatch: {video.name}: "
                    f"{width}x{height} @ {fps:g} fps"
                )
            if frames < 401:
                raise RuntimeError(
                    f"Packaged video has too few frames: {video.name}: {frames}"
                )
            video_reports[stem] = {
                "width": width,
                "height": height,
                "fps": fps,
                "frames": frames,
            }
            preview = media_root / f"{stem}.gif"
            if not preview.is_file():
                raise RuntimeError(f"Package lacks required GIF: {preview.name}")
            try:
                from PIL import Image

                with Image.open(preview) as image:
                    gif_frames = int(image.n_frames)
                    image.seek(gif_frames - 1)
                    image.convert("RGB").load()
            except (ImportError, OSError, EOFError) as exc:
                raise RuntimeError(
                    f"Packaged GIF cannot be decoded: {preview.name}: {exc}"
                ) from exc
            if gif_frames < 30:
                raise RuntimeError(
                    f"Packaged GIF has too few frames: {preview.name}: "
                    f"{gif_frames}"
                )
            video_reports[stem]["gif_frames"] = gif_frames
        readme = extract / "documentation" / "README.md"
        presentation = (
            extract
            / "presentation"
            / "Spike_Topping_TypeIII_simulation.pptx"
        )
        if not readme.is_file() or not readme.read_text(encoding="utf-8").strip():
            raise RuntimeError("Packaged UTF-8 README is missing or empty.")
        if not presentation.is_file() or presentation.stat().st_size == 0:
            raise RuntimeError("Packaged presentation is missing or empty.")
        report.update(
            {
                "event_snapshots": event_snapshots,
                "control_snapshots": control_snapshots,
                "zip_crc": "passed",
                "media": video_reports,
                "readme": "passed",
                "presentation": "passed",
            }
        )
        return report


def build_package(
    event_run: Path,
    control_run: Path,
    output: Path,
) -> dict[str, Any]:
    event_run = Path(event_run).resolve()
    control_run = Path(control_run).resolve()
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(
        tempfile.mkdtemp(
            prefix=".spike_typeiii_stage_",
            dir=output.parent,
        )
    )
    try:
        _stage_package(stage, event_run, control_run)
        _write_manifests(stage, event_run=event_run, control_run=control_run)
        staged_report = verify(stage)
        if not staged_report["passed"]:
            raise RuntimeError(f"Staging verification failed: {staged_report}")
        _zip(stage, output)
        archive_report = _verify_archive(output)
    finally:
        shutil.rmtree(stage, ignore_errors=True)
    return {
        "schema": "spike-typeiii-package-result-v1",
        "path": str(output),
        "bytes": output.stat().st_size,
        "sha256": _sha256(output),
        "verification": archive_report,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event-run", type=Path, required=True)
    parser.add_argument("--control-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = build_package(args.event_run, args.control_run, args.output)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
