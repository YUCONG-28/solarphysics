# MUSER contour preview

Purpose: specify the preview workflow that lays a MUSER contour over DART, CSO and
AIA panels at one reference time, and optionally renders an approved preview into a
synchronized movie.

Reader: whoever runs `workflow radio muser-preview` or sets `make_video: true`, and
interprets the resulting PNG and provenance JSON. Vocabulary is in
[terminology](terminology.md); the raw-image companion workflow is documented in
[radio comparison](radio_comparison.md); the composite frontend is
[aia radio composite](aia_radio_composite.md).

The claim this workflow does not support: the composite compares instruments
observationally. It is not an absolute cross-instrument intensity calibration.

## 1. Purpose and reader

One PNG plus one provenance JSON per run, at one chosen reference time. The
workflow performs no source fit and no polarization synthesis of its own.

## 2. Scope

```console
Apps/run.sh workflow radio muser-preview --config <private.json> --output <new-directory>
```

Run in `solarphysics_env_latest`. The output directory must be new and no input
observation is modified. A movie is produced only from an already approved
configuration with `make_video: true`.

## 3. Definitions and notation

| Field | Meaning |
| --- | --- |
| `label` | Text label carried into the PNG title and the JSON |
| `muser_directory` | One event containing `img_fits` and a `PWR_I` spectrum |
| `dart_directory` | `frequency/LL` and `frequency/RR` subdirectories |
| `aia_directory` | `171` and `304` subdirectories |
| `spectrum_directory` | The standard DART four-file spectrum |
| `frequencies_mhz` | Nonempty subset of the supported MUSER bands; omitted selects all |
| `cso_files` | Ordered, non-overlapping original-UTC RCP/LCP CSO FITS replacing the lower DART display |
| `spectrum_frequency_bounds_mhz` | Frequency bounds for the CSO panel |
| `relative_background_db` | Switch to `10 log10(I/B(f))` with a per-channel positive-sample P20 background |
| `remove_missing_times` | Remove entirely empty CSO time bins that are bounded by observations |
| `display_gap_seconds` | Gap handling for the spectrum display; `0` disables interpolation |
| `fov_arcsec` | `[xmin, xmax, ymin, ymax]`, finite and increasing |
| `fov_reference` | Optional private record naming a reference video and the inspected frame |
| `legend_location` | Legend placement so it does not obscure contours |
| `make_video`, `video_fps` | Movie switch and playback rate (default `10`) |

Record the source checkout revision in the private configuration when running a
code snapshot.

## 4. Method

### 4.1 Reference frame selection

The reference frame maximizes the mean DART source-dB intensity across 2 MHz bands
centered on the selected frequencies:

```math
f^{*} = \arg\max_{f \in F}\ \frac{1}{|W_f|}\sum_{t \in W_f} S_{\mathrm{DART}}(t, f)
\tag{1}
```

`W_f` is the 2 MHz band centered on `f` and `S_DART` the source-dB intensity.
Ranking retains native time samples. The selected sample must match MUSER, each DART
polarization frame (0.1 s) and both AIA channels (12 s); a missing pair fails
explicitly rather than being substituted. Image indexing uses FITS observation times,
never directory labels.

### 4.2 Contours and coordinates

Contours are 90% of each finite positive image peak, drawn on the observed angular
HPLN/HPLT WCS grid. There is no source fitting, no solar-radius rescaling, no
empirical shift, and no interpolation between times. LL/RR image grids must match
before summation. Contours beyond two apparent solar radii are retained and flagged.

### 4.3 Panels

AIA is display-resampled to at most 1024 pixels per dimension. Both AIA panels share
the solar-disk/contour bounding box with a 10% margin. Both spectra share UTC and
frequency limits but keep independent color limits. Gaps larger than 1.5 times the
median cadence or channel spacing are left blank. MUSER `FMHZ` and beam extensions
are read per plane; spectrum channels are sorted together with their data because
real SPW ordering need not be monotonic. XX/YY are not reinterpreted as circular
polarization.

### 4.4 CSO spectral replacement

With `cso_files`, finite native R+L samples are averaged into 0.1 s / 0.25 MHz bins
and displayed as `log10` of SFU from the source header; empty bins stay blank. The
frame-selection score remains the original DART score. `relative_background_db`
removes the instrument's frequency response, not its absolute flux scale:

```math
\mathrm{RB}(t, f) = 10\log_{10}\!\left(\frac{I(t, f)}{B(f)}\right),
\qquad B(f) = P_{20}\bigl\{I(t, f) > 0\bigr\}
\tag{2}
```

`remove_missing_times` compresses the time axis and labels ticks in original UTC;
removed intervals are recorded, and no drift-rate inference may use compressed plot
distances. `display_gap_seconds = 0` disables interpolation. Original FITS files
are never modified.

### 4.5 Movie mode

Each video frame is one valid native DART149 observation whose requested radio pairs
are within 0.1 s and whose AIA frames are within 12 s. Unmatched and
removed-interval frames are skipped and recorded. Constant playback fps is not a
uniform physical cadence, and UTC is shown per frame. Spectral background, color
limits, AIA color limits and the field of view stay fixed for the whole movie.

## 5. Assumptions and validity

- The reference time is a display anchor chosen by a spectral criterion; it is not
  a claim that all instruments integrated simultaneously.
- Contours describe apparent structure at the observed resolution; the beam is not
  deconvolved here.
- Every actual time difference is recorded rather than absorbed into a tolerance.
- A compressed time axis changes the apparent slope of anything plotted on it.

## 6. Failure modes and uncertainty

| Trigger | Detection | Consequence | Retained record |
| --- | --- | --- | --- |
| A required pair is missing at the reference time | Explicit failure | No PNG produced | Requested and available samples |
| LL/RR grids differ | Pre-summation check | Summation refused | Grid shapes |
| Contour beyond two apparent solar radii | Flag | Contour kept and marked | Flag and value |
| Gap above 1.5× median cadence | Blank band | No interpolation | Gap bounds |
| Removed CSO intervals | Removal log | Compressed axis | Interval list and duration |
| AIA beyond the requested FOV | Crop boundary | Structures outside are not moved | Exact angular bounds |

The JSON records selected input hashes, spectral input sizes and timestamps, actual
matching differences, channel frequencies, beam metadata, contour levels, color
limits, the WCS policy, implementation hashes and the PNG hash. Intensity comparison
across panels carries each instrument's full calibration uncertainty; it is not a
measured ratio [inferred].

## 7. Verification

A preview is acceptable when the recorded matching differences are inside the stated
tolerances, the contour level equals 90% of the recorded image peak, the WCS policy
matches the intended convention, and every quality note has been read. A movie
additionally requires its frame manifest and skip list to be inspected; playback
smoothness is not evidence.

## 8. Code anchors

| Behavior | Implementation |
| --- | --- |
| Workflow entry point and panel assembly | `Apps/solar_apps/workflows/radio/muser_preview.py` |
| MUSER readers and plane metadata | `Python/solar_toolkit/radio/muser.py` |
| Frame matching and comparison | `Python/solar_toolkit/radio/muser_comparison.py` |
| Relative-background spectrum display | `Python/solar_toolkit/radio/spectrum_display.py` |
| Dispatcher alias | `Apps/solar_apps/workflows/radio/dispatcher.py` (`muser-preview`) |
