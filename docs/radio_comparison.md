# Original-image radio comparison and localization

## Purpose and reader

Purpose: fix what the radio comparison workflows compute from native images, and
which part of their output is a measurement rather than a display transform.

Reader: a caller reading a CSO spectrum, a comparison preview or a localization
fit, and the reviewer checking that a documentation change kept the boundary.

Reusable calculations live in `solar_toolkit.radio`: `raw_comparison` reads
native MUSER, DART, and AIA images with observation-time provenance;
`cso_comparison` computes memory-bounded finite RCP+LCP spectral and light-curve
means; `muser_localization` fits an explicitly selected native-pixel source ROI.
Scientific inputs and output locations must be supplied by the caller. Keep
event configurations, FITS observations, figures, and run receipts in private
runtime storage.

The one claim this guide does not support: that the display spectrum is a
cross-instrument intensity calibration. It is a separate display transform, and
apparent widths are not beam-deconvolved; [Assumptions and
validity](#assumptions-and-validity) lists the remaining boundaries.

Vocabulary: [terminology](terminology.md). File structure:
[writing-standard](writing-standard.md).

## Scope

Covered:

- Native CSO polarization extraction from FITS, including multiple files and gzip
  inputs.
- The `muser-raw-preview` and `muser-raw-video` comparison workflows, their
  configuration and their outputs.
- `muser-localization` fits on an explicitly selected native-pixel ROI.

Deliberately out of scope:

- Event configurations, FITS observations, figures and run receipts; they stay in
  private runtime storage.
- An observing frequency or an implicit source and plane selection: neither is
  supplied.
- Upstream MUSER polarization synthesis.
- An absolute cross-instrument intensity calibration.

## Definitions and notation

| Symbol or field | Unit | Meaning |
| --- | --- | --- |
| `sum_rcp`, `sum_lcp` | SFU | summed channel intensity per frequency bin, finite pairs only |
| `N_sample` | count | channels with a finite RR/LL pair, the only ones entering both sums |
| `I_mean` | SFU | mean total intensity, Equation 1 |
| `P_circ` | percent | circular polarization, Equation 2 |
| `I`, `B` | SFU | plotted flux and its per-frequency positive P20 background |
| `S_display` | logarithmic display scale | display spectrum, Equation 3 |
| spectrum bin | 0.1 s by 0.25 MHz | CSO spectral cell; uncovered cells stay missing |
| `window_seconds` | s | half-window, default 300 |
| `half_bandwidth_mhz` | MHz | default 1 |
| `fov_arcsec` | arcsec | default `[-1200, 1200, -1200, 1200]` |
| `radio_tolerance_seconds` | s | default 0.25 |
| `aia_tolerance_seconds` | s | default 12 |
| `roi_xyxy` | native pixel indices | zero-based integer half-open `[x0, y0, x1, y1]` |
| `plane_index` | index | zero-based image plane |
| `fit_bg` | flag: `constant` or `plane` | background model, globally or per job |
| `display_window_utc` | UTC | display crop that leaves the extraction window unchanged |

## Method

### Native CSO polarization

The following example reads the synthetic observation produced by the
[radio example](../Python/examples/radio/README.md). Run it from the repository
root, replacing the input and output paths with your local paths. On Windows use
`Apps/run.ps1` with the same workflow arguments.

```bash
./Apps/run.sh workflow radio cso --native-polarization \
  --file-path Local/outputs/example/synthetic.fits \
  --time-start 2000-01-01T00:00:00Z --time-end 2000-01-01T00:00:00.800Z \
  --frequency-start 100 --frequency-end 200 --frequency-bin-mhz 100 \
  --output-dir Local/outputs/native-example
```

Supply multiple files with `--file-paths`. For gzip inputs, `--cache-root`
selects the temporary directory used for random reads. Allow enough free space
for the expanded FITS files. The workflow writes `native_polarization.npz`,
`native_polarization.json` and `native_polarization.png`.

Each time column represents an observed instant. For each frequency bin, only
channels with a finite pair enter both sums. Mean total intensity in SFU is
Equation 1; circular polarization in percent is Equation 2.

```math
I_{\mathrm{mean}} = \frac{\mathrm{sum}_{\mathrm{rcp}} + \mathrm{sum}_{\mathrm{lcp}}}{N_{\mathrm{sample}}} \tag{1}
```

```math
P_{\mathrm{circ}}[\%] = 100\cdot\frac{\mathrm{sum}_{\mathrm{rcp}} - \mathrm{sum}_{\mathrm{lcp}}}{\mathrm{sum}_{\mathrm{rcp}} + \mathrm{sum}_{\mathrm{lcp}}} \tag{2}
```

Symbols and units are those of [Definitions and
notation](#definitions-and-notation).

The figure plots elapsed SI seconds from the labeled UTC start and draws each
coverage segment separately so gaps remain visible. The Python API returns the
arrays and metadata directly through
`solar_toolkit.radio.cso_extract.extract_native_polarization`.

### Comparison workflows

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

Optional settings are `window_seconds` (half-window, default 300),
`half_bandwidth_mhz` (default 1), `fov_arcsec` (default
`[-1200, 1200, -1200, 1200]`), `radio_tolerance_seconds` (default 0.25), and
`aia_tolerance_seconds` (default 12). The image grid has four panels for one
band, or three columns by two rows for two bands. Native WCS positions and
signed radio pixels are preserved; missing matches remain visible as NO DATA.
Large images may use a recorded integer pixel stride for rendering.

CSO output uses finite native R+L samples, with sample-count weighting rather
than averages of already binned means. Spectrum bins are 0.1 seconds by
0.25 MHz; uncovered cells remain missing. Selected-band flux retains source SFU
units. The display spectrum uses a per-frequency positive P20 background and
`10 log10(I/B)`, Equation 3.

```math
S_{\mathrm{display}} = 10\log_{10}\!\left(\frac{I}{B}\right) \tag{3}
```

It is a separate display transform, not a cross-instrument intensity
calibration. Source axes, counts, time offsets, and input hashes are recorded in
the output JSON/CSV.

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
decoding confirms its frame count. When a panel has no reference-time match, its
first usable planned image establishes a fixed movie scale; the source frame is
recorded separately from the original preview. A source missing throughout the plan
keeps its NO DATA panel and an empty scale. `--plan-only` writes the sequence
plan without rendering.

### Explicit localization jobs

Run `Apps/run.sh workflow radio muser-localization --config <private.json>
--output <new-directory>`. The JSON contains a nonempty `jobs` array. Each job
requires `path`, zero-based `plane_index`, integer half-open `roi_xyxy`
`[x0, y0, x1, y1]`, and a nonempty `source_id`. Relative FITS paths resolve
beside the configuration. Optional `fit_bg` is `constant` or `plane`, globally
or per job. There is no implicit source or plane selection.

The fitter reuses the shared rotated Gaussian model, retains finite native
pixels with equal weights, and transforms the numerical center through the
supplied HPLN/HPLT WCS. Multiple peaks, large residuals, low coverage, ROI
truncation, and unsupported WCS remain diagnostics.

Each job produces a record, including failures. `sources.json`/`sources.csv`
collect them; input hashes are checked again before `COMPLETE.json` records
output hashes and success/review counts. Exit status 2 indicates failed jobs;
review flags remain relevant even when all numerical fits succeed. Storage
failures or changed inputs do not produce a completion marker. Use a new output
directory for retry.

## Assumptions and validity

The method holds under the conditions below; each one concerns the input or the
upstream product, not something these workflows verify.

- Empty bins, nonpositive totals and negative aggregated channel intensities have
  undefined polarization.
- Uncovered spectrum cells remain missing; compressing an internal empty interval
  is explicit, and CSV rows keep their original UTC with a `displayed` flag.
- Apparent widths use the local WCS Jacobian and are not beam-deconvolved.
- Covariance assumes independent residuals and does not include calibration, beam
  correlation, or model uncertainty.
- Playback FPS does not imply uniform physical time sampling.
- MUSER I is the file's designation; this workflow does not verify upstream
  polarization synthesis.

## Failure modes and uncertainty

| Trigger | Detection | Consequence | Retained record |
| --- | --- | --- | --- |
| A frequency bin has no finite RR/LL pair, a nonpositive total or a negative aggregated channel | the sums exclude the channel pair | polarization is undefined for that bin | the per-bin sample count in the output JSON/CSV |
| gzip inputs larger than the free space below `--cache-root` | the expanded FITS files do not fit | the random reads cannot be served | the cache root named on the command line |
| A panel has no reference-time match | the plan reports no match for that instant | its first usable planned image fixes the movie scale | the source frame, recorded apart from the original preview |
| A source is missing throughout the plan | the plan contains no frame for it | its panel stays NO DATA with an empty scale | the NO DATA panel |
| Multiple peaks, large residuals, low coverage, ROI truncation or unsupported WCS | the job record carries the diagnostic | the fit is flagged for review | `sources.json`/`sources.csv` and the review flag |
| An MP4 is not completely decodable | the frame count does not match the plan | the file stays at its partial filename | every PNG and planned frame already recorded |

## Verification

Run the command of the section that matches the product, then check the files it
names: `native_polarization.npz`, `native_polarization.json` and
`native_polarization.png` for the CSO extraction; `sources.json`,
`sources.csv` and `COMPLETE.json` for the localization jobs; an MP4 promoted
from its partial filename only after complete decoding confirms its frame count.
`--plan-only` writes the sequence plan without rendering, which separates a
planning defect from a rendering one. Review flags remain relevant even when all
numerical fits succeed, and input hashes are checked again before
`COMPLETE.json` records output hashes and success/review counts.

## Code anchors

| Step | Module or entry point |
| --- | --- |
| Native image reading with provenance | `solar_toolkit.radio.raw_comparison` |
| Spectral and light-curve means | `solar_toolkit.radio.cso_comparison` |
| Native CSO polarization | `solar_toolkit.radio.cso_extract.extract_native_polarization`, `workflow radio cso --native-polarization` |
| ROI Gaussian fit | `solar_toolkit.radio.muser_localization`, `workflow radio muser-localization` |
| Comparison preview and movie | `workflow radio muser-raw-preview`, `workflow radio muser-raw-video` |
