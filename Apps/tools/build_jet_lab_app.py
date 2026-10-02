#!/usr/bin/env python3
"""Build a local macOS Jet Lab launcher using the selected Conda interpreter.

This is a launcher, not a redistributable Python application. The repository,
private launch configuration and existing ``solarphysics_env_latest`` must
remain installed. No observations, dependencies or private paths are bundled
in this source file. ``Apps/run.sh`` remains the public shell entry point.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import plistlib
import re
import subprocess
import sys
import sysconfig


_STARTUP_KEYS = ("manifest", "left", "right", "timeline", "event_profile", "annotation")


def read_launch_config(path, *, output=None, code_roots=()):
    """Resolve and validate private input/output locations before building."""
    path = Path(path).expanduser().resolve(strict=True)
    data = json.loads(path.read_text(encoding="utf-8"))
    if (
        data.get("schema") != "solarphysics.jet_lab.launcher"
        or data.get("version") != 1
    ):
        raise ValueError("Expected solarphysics.jet_lab.launcher version 1")
    roots = data.get("allowed_roots")
    if (
        not isinstance(roots, list)
        or not roots
        or any(not isinstance(r, str) for r in roots)
    ):
        raise ValueError(
            "allowed_roots must be a nonempty list of existing directories"
        )
    roots = [Path(r).expanduser().resolve(strict=True) for r in roots]
    if any(not r.is_dir() or r == Path(r.anchor) for r in roots):
        raise ValueError("Allow specific directories, not a filesystem root")
    repo = (
        Path(code_roots[0]).resolve().parent
        if code_roots
        else Path(__file__).resolve().parents[2]
    )
    # Use the same fail-closed policy as the GUI CLI: a workspace ancestor is
    # too broad even when it contains all requested input and runtime paths.
    sys.path.insert(0, str(repo / "Apps"))
    try:
        from solar_apps.platform.paths.allowed_roots import normalize_allowed_roots

        roots = list(normalize_allowed_roots(roots, workspace_root=repo))
    finally:
        sys.path.pop(0)

    def allowed(value, *, existing=True):
        result = Path(value).expanduser()
        if not result.is_absolute():
            result = path.parent / result
        result = result.resolve(strict=existing)
        if any(character in str(result) for character in ("\n", "\r", "\0")):
            raise ValueError("Paths must not contain control characters")
        if not any(result.is_relative_to(root) for root in roots):
            raise ValueError("Path is outside allowed_roots: " + str(result))
        return result

    allowed(path)
    runtime = allowed(data["runtime_root"], existing=False)
    if runtime.exists() and not runtime.is_dir():
        raise ValueError("runtime_root must be a directory")
    for root in code_roots:
        # Code comes from the fixed repository selected by this builder, not
        # from the observation allow-list. It need not be browsable as data.
        if not Path(root).is_dir():
            raise ValueError("Repository code directory is missing")
    startup = data.get("startup", {})
    if not isinstance(startup, dict) or set(startup) - set(_STARTUP_KEYS):
        raise ValueError("Unsupported startup option")
    normalized = {}
    for key, value in startup.items():
        if not isinstance(value, str) or not value:
            raise ValueError("Startup file paths must be nonempty strings")
        source = allowed(value)
        if not source.is_file():
            raise ValueError("Startup input must be a file: " + str(source))
        normalized[key] = str(source)
    if output is not None:
        destination = allowed(output, existing=False)
        if destination.suffix != ".app" or not destination.is_relative_to(runtime):
            raise ValueError("Output must be a new .app directory inside runtime_root")
        if destination.exists():
            raise FileExistsError(
                "Refusing to overwrite an existing application: " + str(destination)
            )
    return dict(
        data,
        allowed_roots=[str(r) for r in roots],
        runtime_root=str(runtime),
        startup=normalized,
    )


def python_runtime():
    """Discover the current environment only; never silently switch Python."""
    if sys.platform != "darwin":
        raise RuntimeError("This launcher builder is for macOS")
    prefix = Path(sys.prefix).resolve()
    if prefix.name != "solarphysics_env_latest":
        raise RuntimeError("Run with the existing solarphysics_env_latest interpreter")
    library_dir = Path(sysconfig.get_config_var("LIBDIR") or prefix / "lib")
    # Conda's LDLIBRARY may name its static archive even when the public dylib
    # is present. Derive its ABI suffix from this interpreter's sysconfig.
    abi = sysconfig.get_config_var("LDVERSION") or sysconfig.get_python_version()
    library = (library_dir / f"libpython{abi}.dylib").resolve(strict=True)
    if not library.is_relative_to(prefix):
        raise RuntimeError("libpython must belong to the selected environment")
    return dict(
        python_home=str(prefix),
        library=str(library),
        python_executable=str(Path(sys.executable).resolve()),
    )


_NATIVE_SOURCE = r"""
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <unistd.h>
#include <dlfcn.h>
#include <mach-o/dyld.h>
static int line(FILE *f, char *buffer, size_t size) {
    if (!fgets(buffer, (int)size, f)) return 0;
    buffer[strcspn(buffer, "\r\n")] = '\0';
    return buffer[0] != '\0';
}
int main(int argc, char **argv) {
    (void)argc;
    char executable[PATH_MAX], real[PATH_MAX], resources[PATH_MAX];
    uint32_t size = sizeof(executable);
    if (_NSGetExecutablePath(executable, &size) || !realpath(executable, real)) return 2;
    char *slash = strrchr(real, '/'); if (!slash) return 2; *slash = '\0';
    if (snprintf(resources, sizeof(resources), "%s/../Resources", real) >= sizeof(resources)) return 2;
    char file[PATH_MAX], home[PATH_MAX], library[PATH_MAX], bootstrap[PATH_MAX];
    if (snprintf(file, sizeof(file), "%s/python-runtime.txt", resources) >= sizeof(file)) return 2;
    FILE *config = fopen(file, "r");
    if (!config) { perror("Jet Lab runtime"); return 2; }
    int valid = line(config, home, sizeof(home)) && line(config, library, sizeof(library));
    fclose(config); if (!valid) return 2;
    if (setenv("PYTHONHOME", home, 1)) return 2;
    void *handle = dlopen(library, RTLD_NOW | RTLD_GLOBAL);
    if (!handle) { fprintf(stderr, "Jet Lab: %s\n", dlerror()); return 2; }
    int (*run_python)(int, char **) = (int (*)(int, char **))dlsym(handle, "Py_BytesMain");
    if (!run_python) { fprintf(stderr, "Jet Lab: Py_BytesMain is unavailable\n"); return 2; }
    if (snprintf(bootstrap, sizeof(bootstrap), "%s/launch.py", resources) >= sizeof(bootstrap)) return 2;
    char *args[] = {argv[0], "-I", bootstrap, NULL};
    return run_python(3, args);
}
"""


_BOOTSTRAP = '''"""Runtime bootstrap; launch paths come from private validated metadata."""
import importlib.util
import json
import os
from pathlib import Path
import sys

resources = Path(__file__).resolve().parent
metadata = json.loads((resources / "bundle-launch.json").read_text(encoding="utf-8"))
spec = importlib.util.spec_from_file_location("jet_launcher_validation", resources / "config_validation.py")
validation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validation)
config = validation.read_launch_config(metadata["configuration"], code_roots=metadata["code_roots"])
runtime = Path(config["runtime_root"])
runtime.mkdir(parents=True, exist_ok=True)
for name, relative in (("SUNPY_CONFIGDIR", "sunpy"), ("MPLCONFIGDIR", "matplotlib"),
                       ("XDG_CACHE_HOME", "cache")):
    directory = runtime / relative
    directory.mkdir(parents=True, exist_ok=True)
    os.environ[name] = str(directory)
os.environ["SOLAR_APPS_LOCAL_ROOT"] = str(runtime)
logs = runtime / "logs"
logs.mkdir(parents=True, exist_ok=True)
stream = (logs / f"jet-lab-{os.getpid()}.log").open("a", encoding="utf-8", buffering=1)
sys.stdout = sys.stderr = stream
sys.executable = metadata["python_executable"]
sys.path[:0] = metadata["code_roots"]
args = ["--allowed-roots", os.pathsep.join(config["allowed_roots"])]
for key, value in config["startup"].items():
    args.extend(["--" + key.replace("_", "-"), value])
from solar_apps.frontends.jet_lab.cli import main
raise SystemExit(main(args))
'''


def build_app(
    config_path, output, *, name="Jet Lab", bundle_id="org.solarphysics.jet-lab"
):
    if not name.strip() or any(character in name for character in ("\n", "\r", "\0")):
        raise ValueError("Application name is empty or contains a control character")
    if not re.fullmatch(r"[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+", bundle_id):
        raise ValueError("Use a reverse-domain bundle ID")
    repo = Path(__file__).resolve().parents[2]
    code_roots = [repo / "Apps", repo / "Python"]
    config_path = Path(config_path).expanduser().resolve(strict=True)
    output = Path(output).expanduser().resolve()
    read_launch_config(config_path, output=output, code_roots=code_roots)
    runtime = python_runtime()
    compiler = Path("/usr/bin/clang")
    if not compiler.is_file():
        raise RuntimeError(
            "Existing macOS command-line compiler is unavailable; no installation was attempted"
        )
    # Exclusive creation preserves every existing .app, including incomplete builds.
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(exist_ok=False)
    contents = output / "Contents"
    resources = contents / "Resources"
    macos = contents / "MacOS"
    resources.mkdir(parents=True)
    macos.mkdir()
    metadata = dict(
        schema="solarphysics.jet_lab.native_launcher",
        version=1,
        configuration=str(config_path),
        code_roots=[str(p) for p in code_roots],
        configuration_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
        **runtime,
    )
    (resources / "bundle-launch.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    (resources / "python-runtime.txt").write_text(
        runtime["python_home"] + "\n" + runtime["library"] + "\n", encoding="utf-8"
    )
    (resources / "launch.py").write_text(_BOOTSTRAP, encoding="utf-8")
    # Bundle the exact validator, so launch-time checks need no current cwd.
    (resources / "config_validation.py").write_text(
        Path(__file__).read_text(encoding="utf-8"), encoding="utf-8"
    )
    source = resources / "native_main.c"
    source.write_text(_NATIVE_SOURCE, encoding="utf-8")
    executable = macos / "JetLabNative"
    result = subprocess.run(
        [str(compiler), str(source), "-O2", "-o", str(executable)],
        text=True,
        capture_output=True,
        timeout=60,
    )
    (resources / "build.log").write_text(
        result.stdout + result.stderr, encoding="utf-8"
    )
    if result.returncode:
        raise RuntimeError(
            "Native launcher compilation failed; see " + str(resources / "build.log")
        )
    executable.chmod(0o755)
    info = dict(
        CFBundleExecutable="JetLabNative",
        CFBundleIdentifier=bundle_id,
        CFBundleName=name,
        CFBundleDisplayName=name,
        CFBundlePackageType="APPL",
        CFBundleVersion="1",
        CFBundleShortVersionString="1.0",
        NSHighResolutionCapable=True,
    )
    with (contents / "Info.plist").open("wb") as stream:
        plistlib.dump(info, stream)
    files = {
        str(path.relative_to(output)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in output.rglob("*")
        if path.is_file()
    }
    (resources / "COMPLETE.json").write_text(
        json.dumps(dict(version=1, sha256=files), indent=2), encoding="utf-8"
    )
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", required=True, help="Private launch JSON inside allowed roots"
    )
    parser.add_argument(
        "--output", required=True, help="New .app under configuration runtime_root"
    )
    parser.add_argument("--name", default="Jet Lab")
    parser.add_argument("--bundle-id", default="org.solarphysics.jet-lab")
    args = parser.parse_args(argv)
    try:
        print(
            build_app(
                args.config, args.output, name=args.name, bundle_id=args.bundle_id
            )
        )
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        parser.exit(2, str(error) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
