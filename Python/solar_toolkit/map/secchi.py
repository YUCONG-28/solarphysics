"""Run official SolarSoft SECCHI_PREP and verify its EUVI products.

This is an IDL bridge, not a Python approximation of instrument calibration.
An installed, licensed IDL and SolarSoft tree are required.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.time import Time

RECIPE = "euvi-dn-per-second-v1"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def provenance_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".secchi.json")


def verify_prepared(path: Path) -> dict:
    """Reject unverified products rather than guessing from exposure or filename."""
    path = Path(path)
    record = json.loads(provenance_path(path).read_text())
    if record.get("recipe") != RECIPE or record.get("status") != "verified":
        raise ValueError("Missing verified SECCHI_PREP recipe")
    if file_sha256(path) != record["output_sha256"]:
        raise ValueError("SECCHI_PREP output SHA-256 mismatch")
    with fits.open(path) as hdus:
        if hdus[0].header.get("SPPREP") != RECIPE:
            raise ValueError("SECCHI_PREP provenance/header mismatch")
    return record


def _idl_string(value: str | Path) -> str:
    value = str(value)
    if any(c in value for c in "\n\r\x00"):
        raise ValueError("Control characters are not valid IDL paths")
    return "'" + value.replace("'", "''") + "'"


def runtime_environment(ssw_root: Path, idl_executable: str) -> tuple[str, dict]:
    executable = shutil.which(idl_executable)
    if executable is None:
        raise RuntimeError("Official SECCHI_PREP requires an available IDL executable")
    root = Path(ssw_root).resolve(strict=True)
    required = [
        root / "gen/idl",
        root / "stereo/secchi/idl/prep/secchi_prep.pro",
        root / "stereo/secchi/calibration",
    ]
    if any(not p.exists() for p in required):
        raise RuntimeError(
            "Incomplete SolarSoft: gen, SECCHI programs and calibration data are required"
        )
    env = os.environ.copy()
    env.update(
        SSW=str(root),
        SSW_INSTR="secchi",
        SSW_SECCHI=str(root / "stereo/secchi"),
        SECCHI_CAL=str(root / "stereo/secchi/calibration"),
        SCC_DATA=str(root / "stereo/secchi/data"),
    )
    return executable, env


def prepare_euvi(
    path: Path,
    output_dir: Path,
    *,
    ssw_root: Path,
    idl_executable: str = "idl",
    timeout: float = 600,
) -> Path:
    """Calibrate one raw EUVI frame with official defaults except DN/open options.

    Bias, SEB corrections, exposure, calibration image, pointing and missing
    blocks remain enabled. No resize, image rotation, interpolation or fill is
    requested. Missing blocks are NaN. Original science pixels are never edited.
    """
    source = Path(path).resolve(strict=True)
    source_hash = file_sha256(source)
    output_dir = Path(output_dir)
    output = output_dir / (source.stem + "_secchi_prep.fits")
    if output.exists():
        record = verify_prepared(output)
        if record["source_sha256"] != source_hash:
            raise ValueError("Existing calibrated product belongs to a different input")
        return output
    with fits.open(source) as hdus:
        original = hdus[0].header.copy()
        if original.get("DETECTOR", "").strip() != "EUVI":
            raise ValueError("SECCHI EUVI input required")
        name = str(original.get("FILENAME", source.name))
        if original.get("SPPREP") or (len(name) > 16 and name[16].isdigit()):
            raise ValueError(
                "Input appears already processed; refusing double calibration"
            )
        if float(original.get("EXPTIME", 0)) <= 0:
            raise ValueError("Positive raw exposure is required")
    executable, env = runtime_environment(ssw_root, idl_executable)
    root = Path(ssw_root).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    # Retain failed runs for diagnosis; never publish their partial outputs.
    stage = Path(tempfile.mkdtemp(prefix="secchi_run_", dir=output_dir))
    script = stage / "run.pro"
    paths = [
        root / "gen/idl",
        root / "stereo/secchi/idl",
        root / "stereo/ssc/idl",
        root / "packages",
    ]
    path_expr = " + ':' + ".join(
        "expand_path(" + _idl_string("+" + str(p)) + ")" for p in paths if p.exists()
    )
    script.write_text(
        "pro app_secchi_run\n"
        "catch, error\nif error ne 0 then begin\n"
        " print, !error_state.msg\n exit, status=1\nendif\n"
        f"!path = {path_expr} + ':' + !path\n"
        "print, 'APP_IDL_VERSION ', !version.release\n"
        f"secchi_prep, {_idl_string(source)}, headers, images, /normal_off, /dn2p_off, fill_value=!values.f_nan\n"
        "if n_elements(headers) ne 1 then message, 'Expected one calibrated header'\n"
        "if total(finite(images)) eq 0 then message, 'No valid calibrated pixels'\n"
        # SCCWRITEFITS handles the official SECCHI structure, including history.
        f"sccwritefits, 'calibrated.fts', reform(images), headers[0], savepath={_idl_string(str(stage) + '/')}\n"
        "print, 'APP_SECCHI_COMPLETE'\nend\n"
    )
    batch = stage / "batch.pro"
    batch.write_text(f".compile {_idl_string(script)}\napp_secchi_run\nexit\n")
    try:
        result = subprocess.run(
            [executable, str(batch)],
            env=env,
            cwd=stage,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:

        def as_text(value):
            return (
                value.decode(errors="replace")
                if isinstance(value, bytes)
                else (value or "")
            )

        (stage / "idl.log").write_text(as_text(exc.stdout) + "\n" + as_text(exc.stderr))
        raise RuntimeError(f"SECCHI_PREP timed out; see {stage / 'idl.log'}") from exc
    log = result.stdout + "\n" + result.stderr
    (stage / "idl.log").write_text(log)
    if result.returncode != 0 or "APP_SECCHI_COMPLETE" not in [
        line.strip() for line in log.splitlines()
    ]:
        raise RuntimeError(f"SECCHI_PREP did not complete; see {stage / 'idl.log'}")
    products = list(stage.glob("*.fts"))
    if len(products) != 1:
        raise RuntimeError("SECCHI_PREP must produce exactly one FITS output")
    product = products[0]
    with fits.open(product, memmap=False) as hdus:
        hdus.verify("exception")
        header = hdus[0].header
        data = hdus[0].data
        if data is None or data.ndim != 2 or not np.isfinite(data).any():
            raise ValueError("Unreadable calibrated image")
        for key in ("DETECTOR", "OBSRVTRY", "WAVELNTH"):
            if str(header.get(key)) != str(original.get(key)):
                raise ValueError(f"Calibration changed identity: {key}")
        if abs((Time(header["DATE-OBS"]) - Time(original["DATE-OBS"])).sec) > 0.001:
            raise ValueError("Calibration changed observation time")
        for key in ("CRPIX1", "CRPIX2", "CRVAL1", "CRVAL2", "CDELT1", "CDELT2"):
            if not np.isfinite(float(header[key])):
                raise ValueError("Invalid calibrated WCS")
        # Unit is established by this exact official recipe, not by EXPTIME.
        # SECCHI_PREP may also normalize summed pixels; retain that qualifier.
        native_unit = str(header.get("BUNIT", ""))
        unit = "DN/s/CCDPIX" if "CCDPIX" in native_unit.upper() else "DN/s"
        header["BUNIT"] = unit
        header["SPPREP"] = RECIPE
        header.add_history(
            "Official SECCHI_PREP /NORMAL_OFF /DN2P_OFF; missing blocks NaN"
        )
        header.add_history("No additional exposure normalization after SECCHI_PREP")
        valid_fraction = float(np.isfinite(data).mean())
        if int(original.get("NMISSING", 0)) > 0 and valid_fraction == 1:
            raise ValueError(
                "Missing telemetry blocks were not retained as invalid pixels"
            )
        staged = stage / "verified.fits"
        hdus.writeto(staged, checksum=True)
    sources = {}
    for directory in (root / "stereo/secchi/idl", root / "stereo/secchi/calibration"):
        for item in sorted(directory.rglob("*")):
            if item.is_file():
                sources[str(item.relative_to(root))] = file_sha256(item)
    record = dict(
        recipe=RECIPE,
        status="verified",
        source=str(source),
        source_sha256=source_hash,
        output_sha256=file_sha256(staged),
        unit=unit,
        finite_fraction=valid_fraction,
        idl_executable=executable,
        log=str(stage / "idl.log"),
        ssw_files=sources,
        options=dict(
            normal_off=True,
            dn2p_off=True,
            missing_fill="NaN",
            pointing_off=False,
            bias_off=False,
            exptime_off=False,
            sebip_off=False,
            calimg_off=False,
            rotate_on=False,
        ),
    )
    # Exclusive creation preserves any product that appeared during calibration.
    with output.open("xb") as dst, staged.open("rb") as src:
        shutil.copyfileobj(src, dst)
    provenance_path(output).write_text(json.dumps(record, indent=2))
    verify_prepared(output)
    return output
