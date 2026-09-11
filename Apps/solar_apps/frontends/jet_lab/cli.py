"""Launch Jet Lab using the interpreter already selected by the public launcher."""

import argparse
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Jet Lab — independent AIA/EUVI annotation prototype"
    )
    parser.add_argument(
        "--allowed-roots",
        help="Allowed local roots, separated by the OS path separator",
    )
    parser.add_argument("--manifest", help="Completed paired-sample manifest JSON")
    parser.add_argument("--left", help="Left observation FITS")
    parser.add_argument("--right", help="Right observation FITS")
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
    window = JetLabWindow(roots)
    try:
        if args.manifest:
            window.load_manifest(args.manifest)
        else:
            if args.left:
                window.open_image(0, args.left)
            if args.right:
                window.open_image(1, args.right)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
