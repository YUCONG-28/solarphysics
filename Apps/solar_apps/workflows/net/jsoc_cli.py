"""Download caller-selected AIA records and write their URL manifest."""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.request import Request, urlopen

OUT = None
BASE = "https://jsoc1.stanford.edu"
SERIES = "aia.lev1_euv_12s"
TIMERANGE = None
WAVES = [211, 304]
MAX_WORKERS = 2

__all__ = [
    "BASE",
    "MAX_WORKERS",
    "OUT",
    "SERIES",
    "TIMERANGE",
    "WAVES",
    "build_parser",
    "collect_records",
    "download_one",
    "main",
]


def collect_records(
    *,
    output_dir: str | Path = OUT,
    base_url: str = BASE,
    series: str = SERIES,
    timerange: str = TIMERANGE,
    waves: list[int] | tuple[int, ...] = tuple(WAVES),
) -> list[tuple[str, Path]]:
    import drms

    if output_dir is None or not timerange:
        raise ValueError("Supply output_dir and an explicit JSOC timerange.")
    output_dir = Path(output_dir)
    client = drms.Client()
    records = []
    for wave in waves:
        recset = f"{series}[{timerange}][{wave}]"
        keys, segs = client.query(recset, key="T_REC,WAVELNTH,QUALITY", seg="image")
        for row, seg in zip(
            keys.to_dict("records"), segs["image"].tolist(), strict=False
        ):
            if int(row.get("QUALITY", 0)) != 0:
                print(
                    f"skip quality {row['QUALITY']}: {row['T_REC']} {wave}", flush=True
                )
                continue
            trec = str(row["T_REC"]).replace("-", "").replace(":", "")
            fname = f"aia_lev1_euv_12s_{trec}_{int(row['WAVELNTH'])}.fits"
            records.append((base_url + seg, output_dir / fname))
    return records


def download_one(item: tuple[str, Path]) -> tuple[str, str, int | str]:
    url, path = item
    if path.exists() and path.stat().st_size > 1024:
        return ("exists", path.name, path.stat().st_size)

    tmp = path.with_suffix(path.suffix + ".part")
    last_error = ""
    for attempt in range(1, 4):
        try:
            req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urlopen(req, timeout=120) as resp, tmp.open("wb") as handle:
                while True:
                    chunk = resp.read(1024 * 1024)
                    if not chunk:
                        break
                    handle.write(chunk)
            if tmp.stat().st_size <= 1024:
                raise RuntimeError(f"download too small: {tmp.stat().st_size}")
            tmp.replace(path)
            return ("done", path.name, path.stat().st_size)
        except Exception as exc:
            last_error = repr(exc)
            try:
                tmp.unlink()
            except FileNotFoundError:
                pass
            time.sleep(2 * attempt)
    return ("failed", path.name, last_error)


def build_parser() -> argparse.ArgumentParser:
    """Build the event-compatible JSOC downloader parser."""
    parser = argparse.ArgumentParser(
        prog="solar-apps workflow net jsoc",
        description="Download selected SDO/AIA level-1 EUV FITS records.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--base-url", default=BASE)
    parser.add_argument("--series", default=SERIES)
    parser.add_argument("--timerange", required=True)
    parser.add_argument("--waves", type=int, nargs="+", default=list(WAVES))
    parser.add_argument("--max-workers", type=int, default=MAX_WORKERS)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = collect_records(
        output_dir=output_dir,
        base_url=args.base_url,
        series=args.series,
        timerange=args.timerange,
        waves=args.waves,
    )
    manifest = output_dir / "manifest_urls.txt"
    manifest.write_text(
        "\n".join(f"{url} {path.name}" for url, path in records) + "\n",
        encoding="utf-8",
    )
    print(f"records: {len(records)}", flush=True)
    print(f"manifest: {manifest}", flush=True)

    seen = set()
    jobs = []
    for url, path in records:
        if path.name not in seen:
            seen.add(path.name)
            jobs.append((url, path))
    print(f"unique files: {len(jobs)}", flush=True)

    start = time.time()
    done = exists = failed = 0
    failures = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [executor.submit(download_one, job) for job in jobs]
        for i, future in enumerate(as_completed(futures), 1):
            status, name, info = future.result()
            if status == "done":
                done += 1
            elif status == "exists":
                exists += 1
            else:
                failed += 1
                failures.append((name, info))

            if i == 1 or i % 10 == 0 or status == "failed":
                elapsed = time.time() - start
                print(
                    f"progress {i}/{len(jobs)} done={done} exists={exists} "
                    f"failed={failed} elapsed={elapsed:.1f}s last={status}:{name}",
                    flush=True,
                )

    if failures:
        failfile = output_dir / "failed_downloads.txt"
        failfile.write_text(
            "\n".join(f"{name}\t{err}" for name, err in failures) + "\n",
            encoding="utf-8",
        )
        print(f"failures written: {failfile}", flush=True)

    print(f"finished done={done} exists={exists} failed={failed}", flush=True)
    print(
        f"files in dir: {sum(1 for path in output_dir.glob('*.fits'))}",
        flush=True,
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
