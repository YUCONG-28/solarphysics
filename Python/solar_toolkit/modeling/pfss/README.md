# Global PFSS and conditional radio diagnostics

Purpose: state what this optional package owns, which inputs it refuses, and what
a solved or traced result may and may not be used to claim.

Reader: whoever runs a global-field solve, reads a completed result bundle, or
compares modelled field lines with an observed radio source. Terms are owned by
the [magnetic models guide](../../../../docs/magnetic_models.md).

This optional package solves a **global** radial magnetic boundary. A visible
disk LOS magnetogram is not a global Br boundary. Importing the namespace, or
reading a completed result, does not import the solver. Computation requires
both `sunkit-magex` and `streamtracer`; PythonTracer is a numerical cross-check,
not a missing-dependency fallback.

## Purpose and boundary

| Owned here | Belongs elsewhere |
| --- | --- |
| Global radial boundary validation, solving, tracing, projection and result bundles | Workflow configuration and the Global PFSS page: `solar_apps.workflows.pfss.cli` and the Apps router |
| Conditional band diagnostics over a completed bundle | Radio-source measurement, calibration and fitting: `solar_toolkit.radio` |
| Bundle validation and magnetic cache keys | Backend version pins and replay receipts: [the environment guide](../../../../environment/README.md) |

## Layout

| Module | Public names | Role |
| --- | --- | --- |
| `boundary.py` | `prepare_boundary` | Validate a global radial boundary without repairing missing observations |
| `solver.py` | `PFSSConfig`, `backend_status`, `solve_pfss` | Lazy public-backend adapter; no density or application dependency |
| `tracing.py` | `trace_fieldlines`, `uniform_seeds` | Trace through the public backend and validate each returned line |
| `projection.py` | `project_fieldlines` | Project genuinely 3D field lines, explicitly freezing their Carrington pattern |
| `bundle.py` | `load_bundle`, `save_bundle` | Portable arrays plus JSON, with the completion manifest written last |
| `cache.py` | Stage keys | Keep display and density parameters outside magnetic geometry |
| `observations.py` | `header_time` | Header-based observation times without filename or photospheric assumptions |

### Entry points and artifacts

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

### Frequency bands

Pass `sample_field_strength=True` to `trace_fieldlines` or use
`"sample_field_strength": true` in the workflow configuration to retain along-line
field strength in Gauss. Sampling errors are separate from trace validity.
The [magnetic models guide](../../../../docs/magnetic_models.md) describes the
backend-free band analysis for completed bundles and the conditional physics.

## Dependency directions

Solving and tracing depend on the external backend; nothing else does:

```text
solar_toolkit.modeling.pfss  ->  sunkit-magex, streamtracer  (solve and trace only)
solar_toolkit.modeling.pfss  -X-> solar_apps
```

The forbidden directions are the `-X->` arrow and the backend at import time: the
namespace, the readers and the band analysis import no solver, and no application
module is imported here.

## Run and verify

Run the checks through Miniforge `solarphysics_env_latest`, the environment the
library is developed in.

`Python/tests/test_pfss.py` covers harmonic source surfaces, open flux, interior
dipole vectors and refinement, tracer cross-checks, global-map conservation,
CAR reprojection, idempotent HMI metadata, occultation, exact projection, branch
ambiguity, Newkirk degeneracy and incomplete/tampered bundles. Existing source
geometry tests cover timestamps, burst crossings and weak intersections.

Optional backend requirements are listed in
[`environment/pfss-requirements.txt`](../../../../environment/pfss-requirements.txt).
Use the lock for the actual target platform, following the
[environment guide](../../../../environment/README.md). A matching installed
version list alone is not a replay receipt; exact-environment verification also
requires the hash-verified installation and detached runtime receipt.

## Rules that must not be violated

- `prepare_boundary` rejects missing or explicitly masked cells, incomplete latitude/longitude
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

The adapter is original project code. The external backend is licensed
GPL-3.0-or-later (sunkit-magex); streamtracer declares GPLv3 in its package classifiers. This does not
relicense unrelated project code; distribution of a combined program requires
observing the external licenses. Backend source is not copied into this package.

## Related documents

- [magnetic models guide](../../../../docs/magnetic_models.md) — backend-free band analysis and the conditional physics.
- [environment guide](../../../../environment/README.md) — platform lock and replay evidence.
- [Python/README.md](../../../README.md) — library boundary, install and verification.
- [Python/docs/FUNCTION_MAP.md](../../../docs/FUNCTION_MAP.md) — the PFSS row and its neighbours.
- [magnetic models example](../../../examples/public_api/README.md) — synthetic PFSS, SHARP and local grid files.
- [public tracer API](https://docs.sunpy.org/projects/sunkit-magex/en/stable/_modules/sunkit_magex/pfss/tracing.html) — the tracer entry point the adapter calls.
- [analytic PFSS tests](https://arxiv.org/abs/2201.07783) — analytic cases behind the tracer cross-checks.
- [Thompson coordinates](https://fits.gsfc.nasa.gov/wcs/coordinates.pdf) — the coordinate definition the projection assumes.
