"""Launch Jet Lab using the interpreter already selected by the public launcher."""

import argparse
import json
from pathlib import Path
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Jet Lab — native AIA/EUVI tie-point 3D reconstruction"
    )
    parser.add_argument(
        "--allowed-roots",
        help="Allowed local roots, separated by the OS path separator",
    )
    parser.add_argument("--manifest", help="Completed paired-sample manifest JSON")
    parser.add_argument("--left", help="Left observation FITS")
    parser.add_argument("--right", help="Right observation FITS")
    parser.add_argument("--timeline", help="Completed native-frame timeline JSON")
    parser.add_argument(
        "--event-profile", help="JSON event bands and UTC browsing interval"
    )
    parser.add_argument(
        "--annotation", help="Existing annotation to preserve on opening"
    )
    args = parser.parse_args(argv)
    from solar_apps.platform.paths.allowed_roots import configured_allowed_roots

    roots = configured_allowed_roots(cli_value=args.allowed_roots)
    if not roots:
        parser.error("Configure --allowed-roots before opening files")
    if any(name in sys.modules for name in ("PyQt5", "PySide6")):
        parser.error("Jet Lab requires a dedicated PyQt6 process")
    from PyQt6.QtWidgets import QApplication
    from .window import JetLabWindow

    app = QApplication.instance() or QApplication(sys.argv[:1])
    app.setApplicationName("Jet Lab")
    app.setOrganizationName("SolarPhysics")
    profile = None
    if args.event_profile:
        # Use the same allow-list policy as observation loading.
        profile_path = Path(args.event_profile).expanduser().resolve()
        if not any(profile_path.is_relative_to(Path(r).resolve()) for r in roots):
            parser.error("Event profile is outside allowed roots")
        profile = json.loads(profile_path.read_text())
    window = JetLabWindow(roots, event_profile=profile)
    try:
        if args.manifest:
            window.load_manifest(args.manifest)
        else:
            if args.left:
                window.open_image(0, args.left)
            if args.right:
                window.open_image(1, args.right)
        if args.annotation:
            window.restore(args.annotation)
        if args.timeline:
            window.common.load_timeline(args.timeline)
            window.set_view_mode("native")
            available = app.primaryScreen().availableGeometry()
            window.resize(
                min(1500, available.width() - 40), min(1000, available.height() - 60)
            )
            window.common.request()
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
