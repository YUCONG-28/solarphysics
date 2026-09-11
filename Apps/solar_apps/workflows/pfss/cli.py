"""PFSS workflow CLI for an explicitly selected Miniforge compute host."""

import argparse
import json
from pathlib import Path

from solar_apps.platform.paths import validate_allowed_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--allowed-roots", nargs="+", required=True)
    args = parser.parse_args(argv)
    config_path = validate_allowed_path(
        args.config, allowed_roots=args.allowed_roots, kind="file"
    )
    config = json.loads(config_path.read_text())
    for key in [
        "boundary_fits",
        "aia_fits",
        "sources_csv",
        "burst_matches_csv",
        "jet_axis_csv",
        "drifts_csv",
    ]:
        if config.get(key):
            config[key] = str(
                validate_allowed_path(
                    config[key], allowed_roots=args.allowed_roots, kind="file"
                )
            )
    if config.get("reuse_run"):
        config["reuse_run"] = str(
            validate_allowed_path(
                config["reuse_run"], allowed_roots=args.allowed_roots, kind="directory"
            )
        )
    config["output_dir"] = str(
        validate_allowed_path(
            config["output_dir"],
            allowed_roots=args.allowed_roots,
            kind="output_directory",
        )
    )
    from .workflow import run
    import sys
    import signal

    if sys.platform.startswith("linux"):
        import resource

        current = resource.getrlimit(resource.RLIMIT_AS)
        limit = 8 * 1024**3
        if current[0] not in (-1, resource.RLIM_INFINITY):
            limit = min(limit, current[0])
        resource.setrlimit(resource.RLIMIT_AS, (limit, current[1]))

    def timed_out(_signum, _frame):
        raise TimeoutError("PFSS run exceeded the 30-minute resource budget")

    signal.signal(signal.SIGALRM, timed_out)
    signal.alarm(1800)
    try:
        run(config)
    except Exception as exc:
        output = Path(config["output_dir"])
        if not (output / "manifest.json").exists():
            output.mkdir(parents=True, exist_ok=True)
            (output / "failure.json").write_text(
                json.dumps(
                    {"error_type": type(exc).__name__, "message": str(exc)}, indent=2
                )
            )
        raise
    finally:
        signal.alarm(0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
