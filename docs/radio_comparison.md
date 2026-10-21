# Original-image radio comparison and localization

Reusable calculations live in `solar_toolkit.radio`: `raw_comparison` reads
native MUSER, DART, and AIA images with observation-time provenance;
`cso_comparison` computes memory-bounded finite RCP+LCP spectral and light-curve
means; `muser_localization` fits an explicitly selected native-pixel source ROI.
Scientific inputs and output locations must be supplied by the caller. Keep
event configurations, FITS observations, figures, and run receipts in private
runtime storage.

## Comparison workflows

Run through `Apps/run.sh workflow radio muser-raw-preview --config <private.json>
--output <new-directory>`, or use `muser-raw-video` with the same configuration
and a required `display_window_utc` pair. Windows uses `Apps/run.ps1`.
Both commands require the repository's Miniforge environment.

The configuration supplies `event_id` (a caller-defined label), `reference_utc`,
`muser_directory`, `aia_directory` (171/304 subdirectories), `dart_mode` (`I` or
`RR+LL`), and `cso_files`. Supply either `frequency_mhz` and `dart_directory`, or
one/two explicit `radio_bands` objects, each containing `dart_mhz`, `muser_mhz`,
and `dart_directory`. There is no default observing frequency. Native DART I is
read directly; the explicit `RR+LL` mode is a sum without a factor of one half.
MUSER I is the file's designation; this workflow does not verify upstream
polarization synthesis.

Optional settings are `window_seconds` (half-window, default 300),
`half_bandwidth_mhz` (default 1), `fov_arcsec` (default
`[-1200, 1200, -1200, 1200]`), `radio_tolerance_seconds` (default 0.25), and
`aia_tolerance_seconds` (default 12). The image grid has four panels for one
band, or three columns by two rows for two bands. Native WCS positions and
signed radio pixels are preserved; missing matches remain visible as NO DATA.
Large images may use a recorded integer pixel stride for rendering.

CSO output uses finite native R+L samples, with sample-count weighting rather
than averages of already binned means. Spectrum bins are 0.1 seconds by
0.25 MHz; uncovered cells remain missing. Selected-band flux retains source
SFU units. The display spectrum uses a per-frequency positive P20 background
and `10 log10(I/B)`. It is a separate display transform, not a cross-instrument
intensity calibration. Source axes, counts, time offsets, and input hashes are
recorded in the output JSON/CSV.

The complete physical time axis is preserved by default. Setting
`remove_internal_missing_times=true` explicitly compresses wholly empty
internal spectral intervals; the figure labels and metadata record this, and
CSV rows retain their original UTC and a `displayed` flag. No interpolation or
zero filling is performed. `display_window_utc` crops existing bins without
changing the extraction window or its background calculation.

The movie workflow uses the existing hierarchical sequence planner, restricting
each stream before RR/LL pairing. It preserves full in-window coverage, permits
frame reuse, and visibly labels offsets above tolerance. Reference image scales
and the spectral background remain fixed. Every PNG and planned frame is
recorded; an MP4 is promoted from its partial filename only after complete
decoding confirms its frame count. Playback FPS does not imply uniform physical
time sampling. When a panel has no reference-time match, its first usable
planned image establishes a fixed movie scale; the source frame is recorded
separately from the original preview. A source missing throughout the plan
keeps its NO DATA panel and an empty scale. `--plan-only` writes the sequence
plan without rendering.

## Explicit localization jobs

Run `Apps/run.sh workflow radio muser-localization --config <private.json>
--output <new-directory>`. The JSON contains a nonempty `jobs` array. Each job
requires `path`, zero-based `plane_index`, integer half-open `roi_xyxy`
`[x0, y0, x1, y1]`, and a nonempty `source_id`. Relative FITS paths resolve beside
the configuration. Optional `fit_bg` is `constant` or `plane`, globally or per
job. There is no implicit source or plane selection.

The fitter reuses the shared rotated Gaussian model, retains finite native
pixels with equal weights, and transforms the numerical center through the
supplied HPLN/HPLT WCS. Apparent widths use the local WCS Jacobian and are not
beam-deconvolved. Covariance assumes independent residuals and does not include
calibration, beam correlation, or model uncertainty. Multiple peaks, large
residuals, low coverage, ROI truncation, and unsupported WCS remain diagnostics.

Each job produces a record, including failures. `sources.json`/`sources.csv`
collect them; input hashes are checked again before `COMPLETE.json` records
output hashes and success/review counts. Exit status 2 indicates failed jobs;
review flags remain relevant even when all numerical fits succeed. Storage
failures or changed inputs do not produce a completion marker. Use a new output
directory for retry.
