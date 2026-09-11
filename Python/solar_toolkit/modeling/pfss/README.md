# Global PFSS and conditional radio diagnostics

This optional package solves a **global** radial magnetic boundary. A visible
disk LOS magnetogram is not a global Br boundary. Importing the namespace, or
reading a completed result, does not import the solver. Computation requires
both `sunkit-magex` and `streamtracer`; PythonTracer is a numerical cross-check,
not a missing-dependency fallback.

## Scientific contract

- `prepare_boundary` rejects missing cells, incomplete latitude/longitude
  coverage, unknown units, known LOS products and unsupported WCS. It converts
  field units to G and canonicalizes HMI's special latitude units exactly once.
  Axis-aligned global CAR maps use conservative cell overlap in sin(latitude)
  to reproject values to CEA. CEA downsampling also conserves signed flux.
- Synoptic assembly start/stop and the effective boundary time remain in the
  audit. A frozen Carrington pattern is an explicit assumption, not a change
  of the original observation time. Input mean subtraction is opt-in. The
  backend independently excludes the monopole mode from its solution; an
  uncorrected input does **not** mean the resulting field retains its net flux.
- `solve_pfss` calls public `Input` and `pfss` APIs. `trace_fieldlines` uses
  public PerformanceTracer, retaining failed traces, warnings, polarity,
  step-limit failures and boundary-termination checks. Endpoints are checked
  against a fixed 0.01 R_sun tolerance; step-size sensitivity is required.
- `project_fieldlines` uses actual 3D points, observation time and observer,
  with finite observer-to-source occultation. It does not place coronal
  points on the photosphere. Photospheric seed boxes are separate assumptions.
- `radio.fieldline_association` retains each local projected branch minimum,
  axis closest point, LOS closest point and separation. Without a supplied
  angular gate, there are **no accepted source positions**. A supplied angular
  gate is exploratory and is not a 3-sigma test. No FWHM-to-centroid-error
  substitution is made. A nearest finite endpoint is only a diagnostic.
- Candidate lines are fixed by EUV projection before reading radio records or
  applying Newkirk. All shell intersections remain in the tables. Rank models
  within the same line and common source sample; compare lines only after
  checking coverage. Minima over shell intersections are conditional fits.
- `h = norm(position)/R_sun - 1`. PFSS positions remain conditional even if
  their projections fit a source. `1x/H2` and `4x/H1` are exactly degenerate.
  Scattering/refraction, clock ambiguity and unverified EUV association remain
  limitations; no other event's correction is applied.

## Entry points and artifacts

The application workflow is `solar_apps.workflows.pfss.cli`, exposed by the
existing Apps command router as `workflow pfss`. Supply `--config` and explicit
`--allowed-roots`. Private run JSON contains `boundary_fits`, `aia_fits`,
`event_utc`, `sources_csv`, `burst_matches_csv`, `drifts_csv`, `jet_axis_csv`,
`window_start_utc`, `window_end_utc`, `output_dir`, optional `seed_hpc_box`, and
`pfss` (`nphi`, `ns`, `nr`, `rss`, `max_seeds`). No machine path belongs here.
The local seed box means candidate photospheric starts, never a measured jet
footpoint. The default trace step is 0.25 cells with 4096 maximum steps.

The workflow is limited to 512 seeds, 8 GiB virtual memory on Linux and 30
minutes. Failed runs retain `failure.json`; complete runs are immutable.
`manifest.json` is written last and authenticates every file by size and
SHA-256. NPZ contains numerical arrays, with explicit JSON coordinate and
schema metadata; no pickled solver object is loaded.

`reuse_run` can reuse a checksum-verified field/trace bundle when its magnetic
cache key matches. Changing display, source time window or density parameters
does not invalidate magnetic geometry. Changing seeds requires a fresh solve:
the backend currently has no public, lossless solver-state serialization
adapter here. Public field samples support inspection, not exact solver restart.
This is a documented remaining cache limitation.

The native App 1.0 **Global PFSS** page reads synchronized bundles, displays the
AIA FITS cutout using WCS, rotates the 3D view, selects lines and filters radio
time/frequency. Its table contains precomputed whole-window model statistics.
Parameter export creates a new configuration for the existing compute-host
execution path. The viewer neither starts remote jobs nor requires the backend.

## Validation and dependency evidence

`Python/tests/test_pfss.py` covers harmonic source surfaces, open flux, interior
dipole vectors and refinement, tracer cross-checks, global-map conservation,
CAR reprojection, idempotent HMI metadata, occultation, exact projection, branch
ambiguity, Newkirk degeneracy and incomplete/tampered bundles. Existing source
geometry tests cover timestamps, burst crossings and weak intersections.

The optional profile is `environment/pfss-requirements.txt`. Linux has its own
`environment/locks/linux-64-py314` wheel hashes; the existing Mac lock remains
unchanged. A matching installed version list alone is not a replay receipt.
Do not claim the main compute environment was replaced by the sealed replay.

The adapter is original project code. The external backend is licensed
GPL-3.0-or-later (sunkit-magex); streamtracer declares GPLv3 in its package classifiers. This does not
relicense unrelated project code; distribution of a combined program requires
observing the external licenses. Backend source is not copied into this package.

Method references: [public tracer API](https://docs.sunpy.org/projects/sunkit-magex/en/stable/_modules/sunkit_magex/pfss/tracing.html),
[analytic PFSS tests](https://arxiv.org/abs/2201.07783),
[Thompson coordinates](https://fits.gsfc.nasa.gov/wcs/coordinates.pdf).
