"""Private launcher configuration is bounded and native bundles never overwrite."""

import importlib.util
import json
from pathlib import Path
import plistlib
import sys

import pytest


_SOURCE = Path(__file__).resolve().parents[2] / "tools" / "build_jet_lab_app.py"
_SPEC = importlib.util.spec_from_file_location("jet_app_builder", _SOURCE)
builder = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(builder)


def _config(tmp_path, *, allowed=None):
    source = tmp_path / "image.fits"
    source.write_bytes(b"path validation fixture, not an observation")
    data = dict(
        schema="solarphysics.jet_lab.launcher",
        version=1,
        allowed_roots=allowed or [str(tmp_path)],
        runtime_root=str(tmp_path / "runtime"),
        startup={"left": str(source)},
    )
    path = tmp_path / "launcher.json"
    path.write_text(json.dumps(data))
    return path, data


def test_private_launcher_config_validates_inputs_and_output(tmp_path):
    path, data = _config(tmp_path)
    out = tmp_path / "runtime" / "Jet Lab.app"
    prepared = builder.read_launch_config(path, output=out)
    assert prepared["startup"]["left"] == str(tmp_path / "image.fits")
    out.mkdir(parents=True)
    with pytest.raises(FileExistsError, match="overwrite"):
        builder.read_launch_config(path, output=out)
    with pytest.raises(ValueError, match="inside runtime_root"):
        builder.read_launch_config(path, output=tmp_path / "Elsewhere.app")
    data["startup"]["left"] = str(_SOURCE)
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="outside allowed_roots"):
        builder.read_launch_config(path)
    data["startup"] = {"shell": "anything"}
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="Unsupported startup"):
        builder.read_launch_config(path)
    data["startup"] = {}
    for broad in (_SOURCE.parents[2], _SOURCE.parents[3]):
        data["allowed_roots"] = [str(broad)]
        path.write_text(json.dumps(data))
        with pytest.raises(ValueError, match="workspace"):
            builder.read_launch_config(path)


@pytest.mark.skipif(
    sys.platform != "darwin" or Path(sys.prefix).name != "solarphysics_env_latest",
    reason="requires the explicitly supported existing macOS environment",
)
def test_native_launcher_build_is_local_and_complete_without_launching(tmp_path):
    config, _ = _config(tmp_path)
    out = tmp_path / "runtime" / "Review.app"
    result = builder.build_app(
        config, out, name="Jet Lab Test", bundle_id="org.solarphysics.jetlab-test"
    )
    assert result == out
    plist = plistlib.loads((out / "Contents/Info.plist").read_bytes())
    assert plist["CFBundleExecutable"] == "JetLabNative"
    executable = out / "Contents/MacOS/JetLabNative"
    assert executable.read_bytes()[:4] in (
        b"\xcf\xfa\xed\xfe",
        b"\xfe\xed\xfa\xcf",
        b"\xca\xfe\xba\xbe",
    )
    resources = out / "Contents/Resources"
    checks = json.loads((resources / "COMPLETE.json").read_text())["sha256"]
    import hashlib

    assert all(
        hashlib.sha256((out / name).read_bytes()).hexdigest() == digest
        for name, digest in checks.items()
    )
    runtime = json.loads((resources / "bundle-launch.json").read_text())
    assert runtime["python_home"] == str(Path(sys.prefix).resolve())
    assert Path(runtime["library"]).is_file()
    assert (resources / "launch.py").read_text().find(
        "solar_apps.frontends.jet_lab.cli"
    ) >= 0
    # The validator and Python bootstrap are executable syntax without importing
    # the application or starting Qt. Actual GUI acceptance is separate.
    compile((resources / "launch.py").read_text(), "launch.py", "exec")
    before = executable.read_bytes()
    with pytest.raises(FileExistsError):
        builder.build_app(config, out)
    assert executable.read_bytes() == before
